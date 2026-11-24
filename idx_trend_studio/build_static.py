"""Build a static copy of the Web UI (for GitHub Pages) with its data as JSON files.

The page normally gets its data from server.py (/api/bars, /api/deep, /api/idx, /api/status).
GitHub Pages cannot run Python, so this script does the server's work once and writes the answers to files:

    site/index.html             the Web UI, switched to static mode (reads data/ instead of /api/)
    site/data/bars.json         daily OHLCV for the watchlist (same shape as /api/bars)
    site/data/deep/<CODE>.json  Deep Forecast with the UI's default settings (same shape as /api/deep when done)
    site/data/idx.json          IDX-API information, only when --db is given (empty otherwise)
    site/data/status.json       build time and steps, shown in the page's status bar

Usage:
    python build_static.py                       # whole watchlist, with Deep Forecast (about 10-20 minutes)
    python build_static.py --only BBRI,BBCA      # a few tickers (quick local test)
    python build_static.py --no-deep             # prices only
    python build_static.py --db PATH             # also include IDX-API information from a local database
    python build_static.py --min-last-bar 2026-11-20   # publish nothing if Yahoo's data is older than that
    python build_static.py --deep-from https://USER.github.io/REPO/data/deep/
                                                 # prices only; copy the Deep Forecast already online (intraday update)
"""
import argparse
import json
import os
import shutil
import sys
import time
import traceback
import urllib.request
from datetime import datetime
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import server            # noqa: E402  (reuses WATCH, YEARS, download(); also imports deep_forecast)
import idx_source        # noqa: E402

# Must match the page's defaults (PAR_DEF.weeks / .band and DS in static/index.html),
# so the stored result is the one the page asks for when nobody has changed the settings.
DEEP_DEFAULTS = dict(weeks=16, band=0.5, trees=200, folds=5, rf=True, xgb=True, sx=True, p=1, d=1, q=1, sea=False)


def bars_frame(bars):
    df = pd.DataFrame(bars)
    df.index = pd.to_datetime(df.pop("t"))
    return df.rename(columns={"o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"})[["Open", "High", "Low", "Close", "Volume"]]


def run_deep(bars):
    o = DEEP_DEFAULTS
    res = server.deep_forecast.run(bars_frame(bars), o["weeks"] * 5, band_sigma=o["band"], n_estimators=o["trees"], n_splits=o["folds"],
                                   use_rf=o["rf"], use_xgb=o["xgb"], use_sarimax=o["sx"], sarimax_order=(o["p"], o["d"], o["q"]),
                                   sarimax_seasonal=(1, 0, 0, 5) if o["sea"] else (0, 0, 0, 0))
    return idx_source.clean(res)


def fetch_deep(src, code):
    """The Deep Forecast file already published for a ticker, or None when it is missing or unreadable."""
    try:
        if src.startswith(("http://", "https://")):
            with urllib.request.urlopen(src.rstrip("/") + f"/{code}.json", timeout=30) as r:
                obj = json.load(r)
        else:
            obj = json.loads((Path(src) / f"{code}.json").read_text(encoding="utf-8"))
        return obj if obj.get("state") == "done" else None
    except Exception as e:
        print(f"deep {code}: no copy to keep ({type(e).__name__}); computing it")
        return None


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, separators=(",", ":")), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(HERE.parent / "site"), help="output folder (default: ../site)")
    ap.add_argument("--only", help="comma-separated tickers instead of the whole watchlist")
    ap.add_argument("--no-deep", action="store_true", help="skip the Deep Forecast models")
    ap.add_argument("--db", help="IDX-API database.sqlite to read dividends, ratios and foreign flow from")
    ap.add_argument("--deep-from", help="URL (or folder) holding the Deep Forecast JSON files already online: copy them instead "
                                        "of running the models (quick intraday price update); a ticker missing there is computed")
    ap.add_argument("--min-last-bar", help="YYYY-MM-DD of the site now online: if Yahoo's newest bar is older, publish nothing "
                                           "(Yahoo sometimes drops the latest day for a while)")
    a = ap.parse_args()

    out = Path(a.out)
    data = out / "data"
    if out.exists():
        shutil.rmtree(out)
    data.mkdir(parents=True)
    codes = [c.strip().upper() for c in a.only.split(",")] if a.only else list(server.WATCH)
    started = server.now_wib()
    steps = []

    # 1. prices
    t0 = time.time()
    bars = {}
    for i in range(0, len(codes), 50):
        bars.update(server.download(codes[i:i + 50], server.YEARS))
    missing = [c for c in codes if c not in bars]
    if not bars:
        sys.exit("Yahoo Finance returned no data; nothing to publish.")
    last = max(b[-1]["t"] for b in bars.values())
    if a.min_last_bar and last < a.min_last_bar:
        msg = f"Yahoo's newest bar is {last}, older than the {a.min_last_bar} already online; keeping the current site."
        print(f"::warning::{msg}")
        if os.environ.get("GITHUB_OUTPUT"):
            with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as fh:
                fh.write("stale=true\n")
        return
    write(data / "bars.json", {"bars": bars, "missing": missing, "fetched": started.strftime("%Y-%m-%d %H:%M WIB")})
    steps.append({"id": "prices", "label": "1 · yfinance", "state": "ok" if not missing else "warn",
                  "detail": f"{len(bars)}/{len(codes)} tickers to {last}" + (f" · no data: {', '.join(missing)}" if missing else ""),
                  "at": started.strftime("%H:%M:%S")})
    print(f"prices: {len(bars)} tickers in {time.time() - t0:.0f}s, last bar {last}")

    # 2. Deep Forecast
    if a.no_deep or server.deep_forecast is None:
        why = "skipped (--no-deep)" if a.no_deep else f"unavailable ({server.DEEP_ERR})"
        steps.append({"id": "deep", "label": "2 · Deep Forecast", "state": "skip", "detail": why, "at": ""})
    else:
        done, failed, reused = 0, [], 0
        for code in [c for c in codes if c in bars and c != "JKSE"] + (["JKSE"] if "JKSE" in bars else []):
            t1 = time.time()
            if a.deep_from:
                old = fetch_deep(a.deep_from, code)
                if old is not None:
                    write(data / "deep" / f"{code}.json", old)
                    reused += 1
                    continue
            try:
                res = run_deep(bars[code])
                write(data / "deep" / f"{code}.json", {"state": "done", "result": res, "built": started.strftime("%Y-%m-%d %H:%M WIB")})
                done += 1
                print(f"deep {code}: {'ok' if res.get('ok') else res.get('error', 'not ok')} ({time.time() - t1:.0f}s)")
            except Exception as e:
                failed.append(code)
                print(f"deep {code}: FAILED {type(e).__name__}: {e}")
                traceback.print_exc()
        detail = (f"{done} tickers" if done or not reused else "") \
            + (f"{' · ' if done else ''}{reused} kept from the last full build" if reused else "") \
            + (f" · failed: {', '.join(failed)}" if failed else "")
        steps.append({"id": "deep", "label": "2 · Deep Forecast", "state": "ok" if not failed else "warn",
                      "detail": detail, "at": server.now_wib().strftime("%H:%M:%S")})
        print(f"deep: {done} computed, {reused} kept, {len(failed)} failed")

    # 3. IDX-API information (only from a local database)
    idx = {"info": {}, "market": {}, "db": {}}
    if a.db:
        try:
            idx = idx_source.snapshot(a.db, [c for c in codes if c != "JKSE"])
            steps.append({"id": "idx", "label": "3 · IDX info", "state": "ok", "detail": f"DB prices to {idx['db'].get('latest')}", "at": server.now_wib().strftime("%H:%M:%S")})
        except Exception as e:
            steps.append({"id": "idx", "label": "3 · IDX info", "state": "err", "detail": f"{type(e).__name__}: {e}"[:140], "at": ""})
    else:
        steps.append({"id": "idx", "label": "3 · IDX info", "state": "skip", "detail": "not in the online version", "at": ""})
    write(data / "idx.json", idx_source.clean(idx))

    write(data / "status.json", {"static": True, "built": started.strftime("%Y-%m-%d %H:%M WIB"), "last_bar": last,
                                 "steps": steps, "versions": {"prices": 1, "idx": 1}, "deep_defaults": DEEP_DEFAULTS})

    # 4. the page, switched to static mode
    html = (HERE / "static" / "index.html").read_text(encoding="utf-8")
    marker = '<meta charset="utf-8">'
    if marker not in html:
        sys.exit("index.html: <meta charset> not found; cannot switch the page to static mode")
    html = html.replace(marker, marker + '<meta name="idx-static" content="data/">', 1)
    (out / "index.html").write_text(html, encoding="utf-8")
    for f in (HERE / "static").iterdir():                # other page files (icon, ...)
        if f.is_file() and f.name != "index.html":
            shutil.copy2(f, out / f.name)
    (out / ".nojekyll").write_text("", encoding="utf-8")
    print(f"site written to {out} in {time.time() - t0:.0f}s ({datetime.now():%H:%M:%S})")


if __name__ == "__main__":
    main()
