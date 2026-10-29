"""Deep forecast for stock_forecast_app_1.5.0: machine-learning + statistical models, validated out of sample.

Pure functions (no Streamlit), so they can be tested on their own.

    run(ohlcv, horizon)  ->  dict with
        models    {"Random Forest": {...}, "XGBoost": {...}}   expected horizon return, target price, 80% range,
                                                               Bullish/Neutral/Bearish probabilities, walk-forward scores
        ensemble  average of the models above
        sarimax   statistical forecast path with an 80% confidence band
        importance, warnings, ...

What the models predict
    regressor   the forward `horizon`-day log return                (price target = last close x exp(return))
    classifier  Bearish / Neutral / Bullish, the same three states the app's Forecast Engine uses
                (Neutral = |return| within `band_sigma` x horizon sigma)
Honesty rules built in
    * features use only data up to the prediction day (no look-ahead)
    * validation is walk-forward (TimeSeriesSplit) with `gap = horizon`, so overlapping targets do not leak across folds
    * every model is compared with a naive baseline; with no out-of-sample skill a warning is returned
    * prediction ranges come from the out-of-fold residuals, not from the in-sample fit
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX

try:
    from xgboost import XGBClassifier, XGBRegressor
    XGB_OK = True
except Exception:          # xgboost not installed: the other models still work
    XGB_OK = False

MIN_HISTORY = 60           # bars needed before the first usable feature row
MIN_SAMPLES = 150          # labelled rows needed to train and validate
CLASS_NAMES = ("bear", "side", "bull")      # label 0, 1, 2


# --------------------------------------------------------------------------- features
def _rsi(close, n=14):
    d = close.diff()
    au = d.clip(lower=0).ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    ad = (-d).clip(lower=0).ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    rsi = 100 - 100 / (1 + au / ad.replace(0, np.nan))
    return rsi.where(ad != 0, 100.0).where(au.notna())


def build_features(ohlcv):
    """Technical features from daily OHLCV. Row t uses only data up to and including day t."""
    c, h, l = ohlcv["Close"].astype(float), ohlcv["High"].astype(float), ohlcv["Low"].astype(float)
    v = ohlcv["Volume"].astype(float).replace(0, np.nan) if "Volume" in ohlcv else pd.Series(np.nan, index=ohlcv.index)
    lr = np.log(c / c.shift(1))
    X = pd.DataFrame(index=ohlcv.index)
    for n in (1, 5, 10, 20, 60):
        X[f"ret_{n}d"] = np.log(c / c.shift(n))
    for n in (10, 20, 60):
        X[f"vol_{n}d"] = lr.rolling(n).std()
    X["vol_ratio_10_60"] = X["vol_10d"] / X["vol_60d"]
    X["rsi14"] = _rsi(c) / 100.0
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    X["macd_hist_pct"] = (macd - macd.ewm(span=9, adjust=False).mean()) / c
    for n in (20, 50, 200):
        X[f"close_vs_sma{n}"] = c / c.rolling(n).mean() - 1
    mid, sd = c.rolling(20).mean(), c.rolling(20).std(ddof=0)
    X["bollinger_pctb"] = (c - (mid - 2 * sd)) / (4 * sd).replace(0, np.nan)
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    X["atr_pct"] = tr.ewm(alpha=1.0 / 14, adjust=False, min_periods=14).mean() / c
    X["volume_5_20"] = np.log(v.rolling(5).mean() / v.rolling(20).mean())
    X["dist_52w_high"] = c / h.rolling(252, min_periods=120).max() - 1
    X["dist_52w_low"] = c / l.rolling(252, min_periods=120).min() - 1
    hi20, lo20 = h.rolling(20).max(), l.rolling(20).min()
    X["range_pos_20d"] = (c - lo20) / (hi20 - lo20).replace(0, np.nan)
    return X.replace([np.inf, -np.inf], np.nan)


# --------------------------------------------------------------------------- models
def _pipe(est, scale=True):
    steps = [("imp", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("sc", StandardScaler()))
    return Pipeline(steps + [("m", est)])


def make_models(n_estimators=300, horizon=20, seed=42):
    """name -> (regressor pipeline factory, classifier pipeline factory).
    Consecutive rows share most of their `horizon`-day target, so a leaf must span roughly half a horizon of rows;
    smaller leaves just memorise overlapping windows and give over-confident probabilities."""
    leaf = max(10, int(horizon) // 2)
    rf_kw = dict(n_estimators=n_estimators, max_depth=5, min_samples_leaf=leaf, max_features="sqrt", n_jobs=-1, random_state=seed)
    models = {"Random Forest": (lambda: _pipe(RandomForestRegressor(**rf_kw)),
                                lambda: _pipe(RandomForestClassifier(class_weight="balanced_subsample", **rf_kw)))}
    if XGB_OK:
        xk = dict(n_estimators=n_estimators, max_depth=3, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8,
                  min_child_weight=max(5, leaf // 2), reg_lambda=5.0, random_state=seed, n_jobs=-1)
        models["XGBoost"] = (lambda: _pipe(XGBRegressor(objective="reg:squarederror", **xk), scale=False),
                             lambda: _pipe(XGBClassifier(objective="multi:softprob", **xk), scale=False))
    return models


def _fit_clf(factory, X, lab):
    """Fit a classifier on labels that may miss a class; returns (model, classes present)."""
    classes = np.unique(lab)
    m = factory()
    m.fit(X, np.searchsorted(classes, lab))
    return m, classes


def _proba3(model, classes, X):
    """Probabilities as a (n, 3) array ordered bear / side / bull, zero for classes unseen in training."""
    p = model.predict_proba(X)
    out = np.zeros((len(X), 3))
    out[:, classes] = p
    return out


def _walk_forward(reg_f, clf_f, X, y, lab, horizon, n_splits):
    """Out-of-sample scores. gap = horizon keeps the overlapping forward-return targets from leaking between train and test."""
    n = len(X)
    while n_splits >= 2 and n - horizon - n_splits * (n // (n_splits + 1)) <= 0:     # a long gap leaves no room for many folds: use fewer
        n_splits -= 1
    if n_splits < 2:
        return None
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=horizon)
    oof_pred, oof_true, rows = [], [], []
    for tr, te in tscv.split(X):
        if len(tr) < 60 or len(te) < 10:
            continue
        reg = reg_f().fit(X.iloc[tr], y.iloc[tr])
        p = reg.predict(X.iloc[te])
        yt = y.iloc[te].to_numpy()
        base = np.full(len(te), y.iloc[tr].mean())
        row = {"mae": mean_absolute_error(yt, p), "base_mae": mean_absolute_error(yt, base),
               "dir_acc": float(np.mean(np.sign(p) == np.sign(yt))), "n": len(te)}
        try:
            clf, classes = _fit_clf(clf_f, X.iloc[tr], lab.iloc[tr].to_numpy())
            pc = classes[np.argmax(_proba3(clf, classes, X.iloc[te]), axis=1)]
            maj = np.bincount(lab.iloc[tr].to_numpy(), minlength=3).argmax()
            row["clf_acc"] = float(np.mean(pc == lab.iloc[te].to_numpy()))
            row["clf_base_acc"] = float(np.mean(lab.iloc[te].to_numpy() == maj))
        except Exception:
            pass
        rows.append(row)
        oof_pred.append(p)
        oof_true.append(yt)
    if not rows:
        return None
    w = np.array([r["n"] for r in rows], dtype=float)
    avg = lambda k: float(np.average([r[k] for r in rows if k in r], weights=[r["n"] for r in rows if k in r])) if any(k in r for r in rows) else None
    cv = {"folds": len(rows), "mae": avg("mae"), "base_mae": avg("base_mae"), "dir_acc": avg("dir_acc"),
          "clf_acc": avg("clf_acc"), "clf_base_acc": avg("clf_base_acc")}
    cv["skill_pct"] = (1 - cv["mae"] / cv["base_mae"]) * 100 if cv["base_mae"] else None
    resid = np.concatenate(oof_true) - np.concatenate(oof_pred)
    return cv, resid


def _importance(final_pipes, names):
    imp = np.zeros(len(names))
    n = 0
    for p in final_pipes:
        fi = getattr(p.named_steps["m"], "feature_importances_", None)
        if fi is not None and fi.sum() > 0:
            imp += fi / fi.sum()
            n += 1
    if not n:
        return []
    imp /= n
    return sorted(zip(names, imp.tolist()), key=lambda t: -t[1])


# --------------------------------------------------------------------------- statistical model
def sarimax_forecast(close, horizon, order=(1, 1, 1), seasonal=(0, 0, 0, 0), alpha=0.2, max_obs=750, holdout_h=None):
    """SARIMAX on the log price. order=(p,d,q); seasonal=(P,D,Q,s). Returns the price path, an (1-alpha) band
    and a one-off hold-out check (fit without the last `horizon` days, forecast them, compare with a flat forecast)."""
    y = np.log(close.dropna().astype(float)).iloc[-max_obs:].reset_index(drop=True)
    if len(y) < max(120, horizon * 2):
        return {"error": f"need at least {max(120, horizon * 2)} daily closes, have {len(y)}"}
    trend = "t" if order[1] >= 1 else "c"      # a drift term; with d >= 1 the linear trend is the drift per day
    SC = 100.0                                 # fit on 100 x log price: a drift of ~1e-4 per day is too small for the optimiser otherwise

    def fit(series):
        """Fit with the drift term; if that fit does not converge or forecasts nonsense, refit without drift (random-walk-type forecast)."""
        for tr in ([trend, "n"] if trend != "n" else ["n"]):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    r = SARIMAX(series, order=order, seasonal_order=seasonal, trend=tr).fit(disp=False, maxiter=300)
                    m = r.get_forecast(horizon).predicted_mean.to_numpy() / SC
                except Exception:
                    continue
            if (bool(r.mle_retvals.get("converged", True)) and np.all(np.isfinite(r.params)) and np.all(np.isfinite(m))
                    and abs(m[-1] - series.iloc[-1] / SC) < np.log(3)):
                return r, tr
        raise RuntimeError("the fit did not converge to a sensible forecast; try another (p, d, q)")

    try:
        res, trend_used = fit(y * SC)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fc = res.get_forecast(horizon)
            mean, ci = fc.predicted_mean.to_numpy() / SC, fc.conf_int(alpha=alpha).to_numpy() / SC
        out = {"path": np.exp(mean).tolist(), "lo": np.exp(ci[:, 0]).tolist(), "hi": np.exp(ci[:, 1]).tolist(),
               "ret": float(mean[-1] - y.iloc[-1]), "order": order, "seasonal": seasonal, "aic": float(res.aic) + 2 * len(y) * np.log(SC),   # likelihood of the unscaled log price
               "coverage": 1 - alpha, "drift": trend_used != "n"}
        if not np.all(np.isfinite(out["lo"] + out["hi"])):
            return {"error": "the confidence band is not finite (unstable fit); try another (p, d, q)"}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"[:200]}
    hh = int(holdout_h or horizon)
    try:    # hold-out: forecast the last `hh` days from the earlier data
        res_h, _ = fit(y.iloc[:-hh] * SC)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mh = res_h.get_forecast(hh).predicted_mean.to_numpy() / SC
        actual, flat = np.exp(y.iloc[-hh:].to_numpy()), np.exp(y.iloc[-hh - 1])
        out["holdout"] = {"model_mae_pct": float(np.mean(np.abs(np.exp(mh) - actual) / actual) * 100),
                          "flat_mae_pct": float(np.mean(np.abs(flat - actual) / actual) * 100)}
    except Exception:
        pass
    return out


# --------------------------------------------------------------------------- main entry
def run(ohlcv, horizon, band_sigma=0.5, n_estimators=300, n_splits=5, use_rf=True, use_xgb=True, use_sarimax=True,
        sarimax_order=(1, 1, 1), sarimax_seasonal=(0, 0, 0, 0)):
    """Train, validate and predict. `ohlcv`: daily Open/High/Low/Close/Volume indexed by date (about 3 years)."""
    horizon = int(horizon)
    res = {"ok": False, "horizon": horizon, "warnings": [], "models": {}}
    if ohlcv is None or len(ohlcv) < MIN_HISTORY + horizon + MIN_SAMPLES:
        res["error"] = (f"Not enough daily history: need at least {MIN_HISTORY + horizon + MIN_SAMPLES} bars for a {horizon}-day horizon, "
                        f"have {0 if ohlcv is None else len(ohlcv)}.")
        return res
    ohlcv = ohlcv.sort_index()
    close = ohlcv["Close"].astype(float)
    last = float(close.iloc[-1])
    X_all = build_features(ohlcv).iloc[MIN_HISTORY:]
    lr = np.log(close / close.shift(1))
    sig_t = (lr.expanding(min_periods=MIN_HISTORY).std() * np.sqrt(horizon)).reindex(X_all.index)
    y_all = np.log(close.shift(-horizon) / close).reindex(X_all.index)
    ok = y_all.notna() & sig_t.notna()
    X, y, sig = X_all[ok], y_all[ok], sig_t[ok]
    lab = pd.Series(np.where(y > band_sigma * sig, 2, np.where(y < -band_sigma * sig, 0, 1)), index=y.index)
    if len(X) < MIN_SAMPLES:
        res["error"] = f"Only {len(X)} labelled samples after removing the last {horizon} days; need {MIN_SAMPLES}."
        return res
    x_now = X_all.iloc[[-1]]
    sigma_h = float(lr.iloc[-250:].std() * np.sqrt(horizon))
    res.update(ok=True, n_samples=int(len(X)), n_features=int(X.shape[1]), last_price=last, last_date=str(ohlcv.index[-1])[:10],
               sigma_h=sigma_h, band=band_sigma * sigma_h, features=list(X.columns),
               class_share={n: float((lab == i).mean()) for i, n in enumerate(CLASS_NAMES)})
    eff = max(1, len(X) // horizon)
    if eff < 12:
        res["warnings"].append(f"Only about {eff} independent {horizon}-day windows exist in the history (overlapping targets), "
                               "so the scores below are noisy. Treat them as a rough guide.")

    wanted = [n for n, flag in (("Random Forest", use_rf), ("XGBoost", use_xgb)) if flag]
    if use_xgb and not XGB_OK:
        res["warnings"].append("xgboost is not installed (pip install xgboost): XGBoost was skipped.")
    final_pipes = []
    for name, (reg_f, clf_f) in make_models(n_estimators, horizon).items():
        if name not in wanted:
            continue
        try:
            wf = _walk_forward(reg_f, clf_f, X, y, lab, horizon, n_splits)
            reg = reg_f().fit(X, y)
            ret = float(reg.predict(x_now)[0])
            clf, classes = _fit_clf(clf_f, X, lab.to_numpy())
            pr = _proba3(clf, classes, x_now)[0]
            final_pipes.append(reg)
            q10, q90 = (np.quantile(wf[1], [0.1, 0.9]) if wf is not None and len(wf[1]) >= 30 else (-1.28 * sigma_h, 1.28 * sigma_h))
            # shrink the class probabilities toward the historical base rate by the model's out-of-sample classification skill
            cv = wf[0] if wf is not None else None
            if cv and cv.get("clf_acc") is not None and cv.get("clf_base_acc") is not None and cv["clf_base_acc"] < 1:
                wgt = float(np.clip((cv["clf_acc"] - cv["clf_base_acc"]) / (1 - cv["clf_base_acc"]), 0, 1))
            else:
                wgt = 0.0
            share = np.array([res["class_share"][k] for k in CLASS_NAMES])
            ps = wgt * pr + (1 - wgt) * share
            res["models"][name] = {"ret": ret, "target": last * np.exp(ret), "lo": last * np.exp(ret + q10), "hi": last * np.exp(ret + q90),
                                   "probs": dict(zip(CLASS_NAMES, ps.tolist())), "probs_raw": dict(zip(CLASS_NAMES, pr.tolist())),
                                   "prob_weight": wgt, "cv": cv}
            if wf is None:
                res["warnings"].append(f"{name}: too little data for walk-forward validation; no out-of-sample scores.")
        except Exception as e:
            res["warnings"].append(f"{name} failed: {type(e).__name__}: {e}"[:200])
    if res["models"]:
        ms = list(res["models"].values())
        er = float(np.mean([m["ret"] for m in ms]))
        res["ensemble"] = {"ret": er, "target": last * np.exp(er), "lo": float(np.mean([m["lo"] for m in ms])), "hi": float(np.mean([m["hi"] for m in ms])),
                           "probs": {k: float(np.mean([m["probs"][k] for m in ms])) for k in CLASS_NAMES},
                           "prob_weight": float(np.mean([m["prob_weight"] for m in ms])),
                           "agree": len({np.sign(m["ret"]) for m in ms}) == 1}
        res["importance"] = _importance(final_pipes, list(X.columns))[:10]
        skills = [m["cv"]["skill_pct"] for m in ms if m["cv"] and m["cv"].get("skill_pct") is not None]
        if skills and max(skills) <= 0:
            res["warnings"].append("No model beat the naive baseline out of sample (skill <= 0%): the predicted returns carry no demonstrated "
                                   "edge for this stock and horizon. Do not rely on them.")
    if use_rf or use_xgb:   # ML term structure for the forecast line: independent of the selected horizon and of the horizon models above
        try:
            res["curve"] = ml_curve(ohlcv, use_rf=use_rf, use_xgb=use_xgb, n_estimators=max(50, n_estimators // 2))
        except Exception as e:
            res["warnings"].append(f"ML term structure failed: {type(e).__name__}: {e}"[:200])
    if use_sarimax:
        # path over >= 130 days and a fixed 60-day hold-out: the SARIMAX component of the forecast line must not change with the horizon
        res["sarimax"] = sarimax_forecast(close, max(horizon, 130), sarimax_order, sarimax_seasonal, holdout_h=60)
        if "path" in res["sarimax"]:
            res["sarimax"]["ret"] = float(np.log(res["sarimax"]["path"][horizon - 1] / last))      # the return at the chosen horizon
        ho = res["sarimax"].get("holdout")
        if ho and ho["model_mae_pct"] > ho["flat_mae_pct"]:
            res["warnings"].append(f"SARIMAX hold-out: its path was off by {ho['model_mae_pct']:.1f}% on the last 60 days versus "
                                   f"{ho['flat_mae_pct']:.1f}% for simply holding the price flat. A single check, but do not over-trust its drift.")
    return res


# --------------------------------------------------------------------------- combine with the Forecast Engine
VOTE_WEIGHTS = {"engine": 0.4, "model": 0.2, "sarimax": 0.2}


def combine(engine, deep, max_weight=0.4):
    """Blend the Deep Forecast into the Forecast Engine's scenario probabilities and score how well the models agree.

    engine = {"probs": {"bull", "side", "bear"}, "targets": {"bull", "side", "bear"} (prices or None), "last_price": float,
              "range": (low, high) or None}
    deep   = the dict returned by run()

    BLEND  p = (1 - w) * p_engine + w * p_ML, with  w = max_weight * q.
           q is the average classification advantage (out-of-sample, over always guessing the majority class) of the models
           that also beat the naive baseline for the return; no such model -> q = 0 -> the engine is left untouched.
           The ML probabilities are already shrunk toward the historical base rate, so a weak model is discounted twice.
    CONVICTION  weighted vote of the engine (40%), each ML model (20%) and SARIMAX (20%). A voter is Bullish / Bearish when its
           expected return is beyond the neutral band, otherwise Neutral; a voter without out-of-sample skill counts a quarter.
           score = 100 x |weighted vote| (directional), or the weight share voting Neutral (neutral consensus).
    The ML models and the engine read the same price history, so agreement is robustness, not independent confirmation."""
    if not deep or not deep.get("ok") or not deep.get("models"):
        return None
    last, band = float(engine["last_price"]), float(deep["band"])
    pe = {k: float(engine["probs"][k]) for k in ("bull", "side", "bear")}
    models, ens = deep["models"], deep["ensemble"]

    skilled = [m for m in models.values() if m.get("cv") and (m["cv"].get("skill_pct") or 0) > 0]
    q = float(np.clip(np.mean([m.get("prob_weight", 0.0) for m in skilled]), 0, 1)) if skilled else 0.0
    w = float(max_weight) * q
    if not skilled:
        reason = "no ML model beat the naive baseline out of sample, so the engine probabilities are left unchanged"
    elif q <= 0:
        reason = "the classifiers show no advantage over the base rate, so the engine probabilities are left unchanged"
    else:
        reason = f"ML weight {w:.0%} = cap {max_weight:.0%} x out-of-sample classification advantage {q:.0%}"
    pm = {k: float(ens["probs"][k]) for k in ("bull", "side", "bear")}
    pb = {k: (1 - w) * pe[k] + w * pm[k] for k in pe}
    tot = sum(pb.values()) or 1.0
    pb = {k: v / tot for k, v in pb.items()}

    tg = engine.get("targets") or {}
    exp_engine = sum(pe[k] * float(tg[k] if tg.get(k) else last) for k in pe)
    expected = (1 - w) * exp_engine + w * float(ens["target"])
    rng = engine.get("range")
    rng_b = ((1 - w) * rng[0] + w * float(ens["lo"]), (1 - w) * rng[1] + w * float(ens["hi"])) if rng else None

    def sgn(r):
        return 1 if r > band else -1 if r < -band else 0

    d_eng = 1 if pe["bull"] > pe["bear"] + 0.05 else -1 if pe["bear"] > pe["bull"] + 0.05 else 0
    votes = [{"name": "Forecast Engine", "dir": d_eng, "weight": VOTE_WEIGHTS["engine"], "note": f"P(bull) {pe['bull']:.0%} vs P(bear) {pe['bear']:.0%}"}]
    for name, m in models.items():
        sk = (m.get("cv") or {}).get("skill_pct")
        votes.append({"name": name, "dir": sgn(m["ret"]), "weight": VOTE_WEIGHTS["model"] * (1.0 if (sk or 0) > 0 else 0.25),
                      "note": f"expected {m['ret']:+.1%}" + ("" if (sk or 0) > 0 else " · no skill, quarter weight")})
    sx = deep.get("sarimax")
    if sx and "path" in sx:
        ho = sx.get("holdout")
        good = not ho or ho["model_mae_pct"] <= ho["flat_mae_pct"]
        votes.append({"name": "SARIMAX", "dir": sgn(sx["ret"]), "weight": VOTE_WEIGHTS["sarimax"] * (1.0 if good else 0.25),
                      "note": f"expected {np.exp(sx['ret']) - 1:+.1%}" + ("" if good else " · worse than flat on hold-out, quarter weight")})
    wsum = sum(v["weight"] for v in votes)
    V = sum(v["weight"] * v["dir"] for v in votes) / wsum
    neutral_share = sum(v["weight"] for v in votes if v["dir"] == 0) / wsum
    direction = "bull" if V >= 0.15 else "bear" if V <= -0.15 else "side"
    score = round(100 * abs(V)) if direction != "side" else round(100 * neutral_share)
    strength = "Strong" if score >= 75 else "Moderate" if score >= 50 else "Weak" if score >= 25 else "Mixed"
    name = {"bull": "bullish", "bear": "bearish", "side": "neutral"}[direction]
    label = f"{strength} {name}" if strength != "Mixed" else "Mixed signals"
    return {"weight": w, "weight_reason": reason, "probs": pb, "probs_engine": pe, "probs_ml": pm,
            "expected_engine": exp_engine, "expected": expected, "range": rng_b,
            "conviction": score, "direction": direction, "label": label, "votes": votes,
            "dominant": max(pb, key=pb.get)}


# --------------------------------------------------------------------------- ML term structure + scenario paths
CURVE_KNOTS = (5, 10, 20, 30, 40, 60, 80, 100, 130, 160, 200, 260)   # horizons (trading days) the term-structure models are trained for


def ml_curve(ohlcv, use_rf=True, use_xgb=True, n_estimators=100, knots=CURVE_KNOTS, seed=42):
    """ML expected cumulative log-return curve: a regressor per horizon in `knots` (Random Forest / XGBoost, same features and pipelines as
    run()), each predicting the forward return from the latest bar, linearly interpolated to every trading day.
    It does not depend on the horizon chosen in the app, so changing 'weeks ahead' never changes it. Returns
    {"days": [0..n], "rf": [...], "xgb": [...], "mean": [...], "knots": [...]}; the lists hold cumulative log returns (0 on day 0)."""
    close = ohlcv["Close"].astype(float)
    X_all = build_features(ohlcv).iloc[MIN_HISTORY:]
    x_now = X_all.iloc[[-1]]
    series = {"Random Forest": [], "XGBoost": []}
    used = []
    for h in knots:
        y = np.log(close.shift(-h) / close).reindex(X_all.index)
        ok = y.notna()
        if int(ok.sum()) < MIN_SAMPLES:
            break
        used.append(h)
        for name, (reg_f, _) in make_models(n_estimators, h, seed).items():
            if (name == "Random Forest" and not use_rf) or (name == "XGBoost" and not use_xgb):
                continue
            series[name].append(float(reg_f().fit(X_all[ok], y[ok]).predict(x_now)[0]))
    if not used:
        return None
    days = np.arange(0, used[-1] + 1)
    out = {"days": days.tolist(), "knots": used}
    curves = []
    for key, name in (("rf", "Random Forest"), ("xgb", "XGBoost")):
        if len(series[name]) == len(used):
            c = np.interp(days, [0] + used, [0.0] + series[name])
            out[key] = c.tolist()
            curves.append(c)
    if not curves:
        return None
    out["mean"] = np.mean(curves, axis=0).tolist()
    out["skill"] = {"rf": None, "xgb": None}
    try:                       # walk-forward skill vs the naive baseline at the reference horizon: the weights of the forecast line use this
        ref = 40
        y = np.log(close.shift(-ref) / close).reindex(X_all.index)
        ok = y.notna()
        if int(ok.sum()) >= MIN_SAMPLES:
            Xr, yr = X_all[ok], y[ok]
            sig = (np.log(close / close.shift(1)).expanding(min_periods=MIN_HISTORY).std() * np.sqrt(ref)).reindex(Xr.index)
            lab = pd.Series(np.where(yr > 0.5 * sig, 2, np.where(yr < -0.5 * sig, 0, 1)), index=yr.index)
            for key, name in (("rf", "Random Forest"), ("xgb", "XGBoost")):
                if (name == "Random Forest" and not use_rf) or (name == "XGBoost" and not use_xgb):
                    continue
                reg_f, clf_f = make_models(n_estimators, ref, seed)[name]
                wf = _walk_forward(reg_f, clf_f, Xr, yr, lab, ref, 4)
                if wf is not None:
                    out["skill"][key] = wf[0].get("skill_pct")
    except Exception:
        pass
    return out


def _smooth(a, k=2):
    """Centred moving average over 2k+1 days (the window shrinks at the ends): turns daily noise into visible swings."""
    a = np.asarray(a, dtype=float)
    return np.array([a[max(0, i - k): i + k + 1].mean() for i in range(len(a))])


# --------------------------------------------------------------------------- ONE consensus forecast line + projected timing
LINE_WEIGHTS = {"engine": 0.30, "fib": 0.20, "rf": 0.15, "xgb": 0.15, "sarimax": 0.20}
LOW_SKILL = 0.25                 # a component without out-of-sample skill counts a quarter (same rule as the conviction vote)


def fib_pattern_target(last, primary, secondaries, levels, snap=0.04, cap=0.35):
    """Where the chart patterns say the price is heading, snapped to a Fibonacci level when one is close.

    primary     {"bias": +1/-1/0, "height": flagpole height (price), "upper": pattern upper line now, "lower": lower line now, "days": flagpole length}
    secondaries [{"bias": +1/-1, "target": measured-move price, "confirmed": bool}]   (double/triple tops and bottoms, head and shoulders)
    levels      Fibonacci prices (retracements and extensions)
    Primary pattern: measured move = the flagpole height added to the breakout side (upper line for a bullish bias, lower line for a bearish one).
    Reversal patterns: their own measured move (confirmed: weight 1, not yet confirmed: 0.5). The target is the weighted mean, limited to +/-35%
    in log terms, and replaced by the nearest Fibonacci level when that level is within 4% of it and on the same side of the price.
    Returns None when no pattern has a direction (the Fibonacci component is then left out of the line)."""
    parts = []
    if primary and primary.get("bias"):
        b = primary["bias"]
        base = primary["upper"] if b > 0 else primary["lower"]
        parts.append({"name": "Primary pattern", "target": base + b * primary["height"], "weight": 1.0, "bias": b})
    for s in secondaries or []:
        if s.get("bias"):
            parts.append({"name": "Reversal pattern", "target": s["target"], "weight": 1.0 if s.get("confirmed") else 0.5, "bias": s["bias"]})
    parts = [p for p in parts if p["target"] and p["target"] > 0]
    if not parts:
        return None
    T = sum(p["weight"] * p["target"] for p in parts) / sum(p["weight"] for p in parts)
    T = last * float(np.exp(np.clip(np.log(T / last), -cap, cap)))
    snapped = None
    if levels:
        near = min(levels, key=lambda v: abs(v - T))
        if abs(near - T) / T <= snap and (near - last) * (T - last) > 0:
            snapped = near
    days = int(primary["days"]) if primary and primary.get("days") else 20
    return {"target": snapped if snapped else T, "raw": T, "snapped": snapped, "days": days, "parts": parts}


def consensus_line(last, H, engine_ret, fib=None, deep=None, smooth_k=2, ml_only=False):
    """ONE forecast line: the weighted mean, day by day, of the cumulative log-return paths of
        Forecast engine (0.30)   its evidence tilt spread evenly over the horizon (the engine has no timing information)
        Fibonacci pattern (0.20) smooth move to the pattern / Fibonacci target within the flagpole duration, flat afterwards
        Random Forest (0.15), XGBoost (0.15)   regressors trained for horizons 5-260 days (deep["curve"]): the models' own timing
        SARIMAX (0.20)           the statistical path
    A component without out-of-sample skill (ML models) or worse than a flat price on hold-out (SARIMAX) counts a quarter. Components that
    are not available are left out and the weights renormalised. The result is smoothed over 5 days so the line shows swings, not noise."""
    t = np.arange(1, H + 1)
    parts = [] if ml_only else [("Forecast engine", LINE_WEIGHTS["engine"], engine_ret * t / H)]
    if fib and not ml_only:
        D = max(10, int(fib["days"]))       # not clipped to H: the line for a shorter horizon is the same line, cut earlier
        x = np.minimum(t / D, 1.0)
        parts.append(("Fibonacci pattern", LINE_WEIGHTS["fib"], np.log(fib["target"] / last) * x * x * (3 - 2 * x)))
    if deep and deep.get("ok"):
        curve, models = deep.get("curve"), deep.get("models", {})
        for key, name in (("rf", "Random Forest"), ("xgb", "XGBoost")):
            if curve and key in curve and name in models:
                sk = (curve.get("skill") or {}).get(key)
                skill = (sk if sk is not None else (models[name].get("cv") or {}).get("skill_pct") or 0) > 0
                arr = np.asarray(curve[key], dtype=float)
                parts.append((name, LINE_WEIGHTS[key] * (1.0 if skill else LOW_SKILL), arr[np.minimum(t, len(arr) - 1)]))
        sx = deep.get("sarimax")
        if sx and "path" in sx:
            ho = sx.get("holdout")
            good = not ho or ho["model_mae_pct"] <= ho["flat_mae_pct"]
            path = np.asarray(sx["path"], dtype=float)
            parts.append(("SARIMAX", LINE_WEIGHTS["sarimax"] * (1.0 if good else LOW_SKILL), np.log(path[np.minimum(t, len(path)) - 1] / last)))
    if not parts:                      # ml_only and no ML result
        return None
    W = sum(w for _, w, _ in parts)
    L = sum(w * c for _, w, c in parts) / W
    if smooth_k:
        L = _smooth(L, smooth_k)
    return {"line": (last * np.exp(L)).tolist(), "ret": float(L[-1]),
            "parts": [{"name": n, "weight": float(w / W), "end_ret": float(c[-1]), "end_price": float(last * np.exp(c[-1])),
                       "prices": (last * np.exp(c)).tolist()} for n, w, c in parts]}


# Fibonacci wave path to a target (5 waves, like an impulse): wave 1 = 40% of the move, wave 2 retraces 50% of wave 1, wave 3 extends 1.618 x wave 1,
# wave 4 retraces 38.2% of wave 3, wave 5 = wave 1 and ends exactly at the target. Time fractions of the flagpole duration.
# After the target an A-B-C correction follows: A gives back 23.6% of the move, B recovers 50% of A, C ends at the 38.2% retracement of the whole move.
WAVE_T = (0.0, 0.15, 0.27, 0.57, 0.75, 1.0, 1.25, 1.40, 1.70)
WAVE_P = (0.0, 0.4, 0.2, 0.847, 0.6, 1.0, 0.764, 0.882, 0.618)


def fib_wave(x):
    """Fraction of the move to the target reached at x = elapsed / flagpole duration (array). Each wave eases in and out, so the turning points are round."""
    x = np.clip(np.asarray(x, dtype=float), 0.0, WAVE_T[-1])
    i = np.minimum(np.searchsorted(WAVE_T, x, side="right") - 1, len(WAVE_T) - 2)
    T, P = np.asarray(WAVE_T), np.asarray(WAVE_P)
    s = (x - T[i]) / (T[i + 1] - T[i])
    return P[i] + (P[i + 1] - P[i]) * s * s * (3 - 2 * s)


def pattern_line(last, H, engine_ret, fib=None):
    """The forecast line: the PATTERN projection. A move to the Fibonacci / measured-move target of the detected pattern within the flagpole
    duration, as five Fibonacci waves with pullbacks of 50% and 38.2% on the way, then an A-B-C correction to the 38.2% retracement; flat afterwards (see fib_wave). Without a detected pattern it falls back to the engine's tilt spread evenly.
    The ML models and the other methods do not change this line (see consensus_line for the all-methods reference)."""
    t = np.arange(1, H + 1)
    if fib:
        D = max(10, int(fib["days"]))
        c, name = np.log(fib["target"] / last) * fib_wave(t / D), "Fibonacci pattern"
    else:
        c, name = engine_ret * t / H, "Forecast engine"
    return {"line": (last * np.exp(c)).tolist(), "ret": float(c[-1]),
            "parts": [{"name": name, "weight": 1.0, "end_ret": float(c[-1]), "end_price": float(last * np.exp(c[-1])), "prices": (last * np.exp(c)).tolist()}]}


def timing_events(line, levels, last, min_swing=0.015):
    """When does the forecast line reach something? `levels` = {name: constant price or a per-day list}.
    A level is crossed on the first day the line is on the other side of it than today's price. Also the turning points of the line
    (peaks / troughs that move at least `min_swing` from the previous turning point or the start).
    Returns [{"kind": "cross"|"peak"|"trough", "name", "day" (1 = tomorrow), "price", "dir": +1/-1}] sorted by day."""
    line = np.asarray(line, dtype=float)
    H = len(line)
    out = []
    for name, lv in levels.items():
        arr = np.full(H, float(lv)) if np.isscalar(lv) else np.asarray(lv, dtype=float)[:H]
        if len(arr) < H or not np.all(np.isfinite(arr)):
            continue
        side0 = np.sign(last - arr[0])
        if side0 == 0:
            continue
        hit = np.nonzero(np.sign(line - arr) == -side0)[0]
        if len(hit):
            out.append({"kind": "cross", "name": name, "day": int(hit[0]) + 1, "price": float(line[hit[0]]), "dir": int(-side0)})
    ref = last
    d = np.diff(np.concatenate([[last], line]))
    for i in range(1, H - 1):
        if (d[i] > 0 >= d[i + 1]) or (d[i] < 0 <= d[i + 1]):
            kind = "peak" if d[i] > 0 else "trough"
            if abs(np.log(line[i] / ref)) >= min_swing:
                out.append({"kind": kind, "name": "Peak" if kind == "peak" else "Trough", "day": i + 1, "price": float(line[i]), "dir": 1 if kind == "peak" else -1})
                ref = line[i]
    return sorted(out, key=lambda e: e["day"])


def _mulberry32(seed):
    """Small seeded PRNG that the HTML page implements identically, so both UIs draw the same candles."""
    M = 0xFFFFFFFF
    a = seed & M

    def nxt():
        nonlocal a
        a = (a + 0x6D2B79F5) & M
        t = ((a ^ (a >> 15)) * (a | 1)) & M
        t = (((t + (((t ^ (t >> 7)) * (t | 61)) & M)) & M) ^ t) & M
        return ((t ^ (t >> 14)) & M) / 4294967296
    return nxt


def forecast_candles(line, last, open_, high, low, close, seed, phi=0.85, shock=0.8, n=250):
    """Candles that FOLLOW the consensus line but move like a real chart (ups and downs, different bodies and wicks).
    close_i = line_i * exp(x_i), where x is a mean-reverting random walk (x_i = phi*x_(i-1) + shock*r) around the line, r being a
    day-to-day return drawn from the last `n` real daily returns (so the volatility is the stock's own). Open = previous close.
    The wicks are drawn from the real wicks of the last `n` candles. The draws are seeded with the last bar's date and consumed one
    day at a time, so the first days do not change when the horizon changes. The candle path is ONE plausible path, not a forecast:
    the consensus line is the expected value, and the candle closes oscillate around it."""
    line = np.asarray(line, dtype=float)
    o_, h_, l_, c_ = (np.asarray(v, dtype=float) for v in (open_, high, low, close))
    r = np.diff(np.log(c_[-(n + 1):]))
    r = r[np.isfinite(r)]
    m = len(r)
    r = r - r.mean() if m else r
    hi_, lo_, op_, cl_ = h_[-m:], l_[-m:], o_[-m:], c_[-m:]
    up = (hi_ - np.maximum(op_, cl_)) / cl_
    dn = (np.minimum(op_, cl_) - lo_) / cl_
    rnd = _mulberry32(int(seed))
    out, x, prev = [], 0.0, float(last)
    for i in range(len(line)):
        k = int(rnd() * m) if m else 0
        k2 = int(rnd() * m) if m else 0
        x = phi * x + (shock * r[k] if m else 0.0)
        c = float(line[i] * np.exp(x))
        o = prev
        out.append({"o": o, "h": float(max(o, c) + (up[k2] * c if m else 0.0)), "l": float(min(o, c) - (dn[k2] * c if m else 0.0)), "c": c})
        prev = c
    return out


def event_label(e):
    """Short chart / table label for a timing_events() entry."""
    if e["kind"] == "peak":
        return "Peak ▲"
    if e["kind"] == "trough":
        return "Trough ▼"
    n = e["name"]
    if n.startswith(("Cycle resistance", "Macro resistance")):
        return "Breakout ▲" if e["dir"] > 0 else "Back below resistance ▼"
    if n.startswith(("Cycle support", "Macro support")):
        return "Breakdown ▼" if e["dir"] < 0 else "Back above support ▲"
    return n.replace("Fibonacci ", "Fib ") + (" ▲" if e["dir"] > 0 else " ▼")
