import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.signal import find_peaks
from scipy.stats import linregress

# Konfigurasi Halaman Web
st.set_page_config(page_title="Macro Pattern & Fibonacci Analyzer", layout="wide")

# ============================================================
# SIDEBAR CUSTOMIZATION
# ============================================================
st.sidebar.header("⚙️ Parameters & Customization")

# Default saat pertama kali dibuka / restart: BBRI.JK, histori 6 bulan
# terakhir (rolling, bukan tanggal tetap) s/d hari ini, interval harian —
# mereplikasi tampilan "siap pakai" yang biasa dipakai, supaya tidak perlu
# setup ulang toggle satu-satu tiap buka aplikasi.
default_end = pd.Timestamp.now().normalize()
default_start = default_end - pd.DateOffset(months=6)

ticker = st.sidebar.text_input("Stock Ticker", value="BBRI.JK")

# Ambil nama bersih tanpa suffix .JK (misal BMRI.JK -> BMRI; ^JKSE -> JKSE)
clean_ticker = ticker.split('.')[0].replace('^', '').upper() if ticker else "STOCK"

col_date1, col_date2 = st.sidebar.columns(2)
start_date = col_date1.date_input("Start Date", default_start)
end_date = col_date2.date_input("End Date", default_end)

interval = st.sidebar.selectbox(
    "Data Interval", 
    options=["1d", "1wk", "1mo"], 
    index=0,
    format_func=lambda x: {"1d": "Daily (1d)", "1wk": "Weekly (1wk)", "1mo": "Monthly (1mo)"}[x]
)

forecast_weeks = st.sidebar.slider("Forecast Weeks Ahead", min_value=4, max_value=52, value=16)

# FIBONACCI SETTINGS (AUTO / MANUAL OVERRIDE)
st.sidebar.subheader("📐 Fibonacci & Levels")
show_fibo = st.sidebar.checkbox("Show Fibonacci Levels", value=True)
use_custom_fibo = st.sidebar.checkbox("Custom High / Low Levels", value=False)

if use_custom_fibo:
    ath_level = float(st.sidebar.number_input("Custom High Level", value=6450))
    crash_low_level = float(st.sidebar.number_input("Custom Low Level", value=2540))
else:
    ath_level = 0.0
    crash_low_level = 0.0

pivot_distance = st.sidebar.number_input("Pivot Distance (Lower-High Peaks)", min_value=1, max_value=20, value=6)
bep_level = float(st.sidebar.number_input("BEP Position (Optional)", value=0))

# TOGGLE MACRO RESISTANCE / SUPPORT (each line can be hidden independently;
# the underlying calculations keep running regardless, since BREAKOUT/BREAKDOWN
# status and other metrics depend on them)
st.sidebar.subheader("📏 Macro Trendlines")
show_macro_resistance = st.sidebar.checkbox("Show Macro Resistance Trendline", value=False)
show_macro_support = st.sidebar.checkbox("Show Macro Support Trendline", value=False)

# HISTORICAL SUPPORT/RESISTANCE ZONES (polarity principle: old support becomes
# new resistance, and vice versa)
st.sidebar.subheader("🔁 Historical S/R Zones")
show_sr_zones = st.sidebar.checkbox("Show Historical S/R Zones (per date range)", value=True)
sr_tolerance_pct = st.sidebar.slider("Zone Price Tolerance (%)", min_value=0.5, max_value=5.0, value=1.5, step=0.1) / 100
sr_min_touches = st.sidebar.number_input("Min Touches per Zone", min_value=2, max_value=10, value=2)
sr_max_span_bars = st.sidebar.number_input("Max Bars per Zone (split if wider)", min_value=10, max_value=200, value=40)

# CHART PATTERN DETECTOR (within the last Fibonacci cycle, always uses DAILY data)
st.sidebar.subheader("📐 Chart Pattern Detector")
show_flag_pennant = st.sidebar.checkbox("Detect Chart Pattern (Last Fibonacci Cycle)", value=True)
st.sidebar.caption("Covers all 20 classic patterns — reversal (Double/Triple Top/Bottom, "
                   "Head & Shoulders) and continuation (Flag, Pennant, Triangles, Wedges, "
                   "Expanding/Broadening). Always uses **daily** data regardless of the Data "
                   "Interval above — these patterns are too short-lived for weekly/monthly resolution.")

date_format_option = st.sidebar.selectbox("X-Axis Format", options=["%b %Y", "%d-%b-%Y", "%Y-%m-%d"], index=0)

# ============================================================
# CORPORATE & MACRO EVENT CALENDAR
# These events have WELL-ESTABLISHED causal mechanisms on price (unlike the
# optional astrology overlay elsewhere): dividend ex-dates cause a mechanical
# price adjustment, MSCI/JPM rebalancing moves real passive-fund flows, and
# central-bank rate decisions directly move bank/rate-sensitive stocks.
# ============================================================
st.sidebar.subheader("📅 Corporate & Macro Event Calendar")
show_dividend_cal = st.sidebar.checkbox("Show Dividend History", value=True)
dividend_source = "yfinance"
dividend_manual_input = ""
if show_dividend_cal:
    dividend_source = st.sidebar.radio(
        "Dividend data source", options=["yfinance (automatic)", "Manual input"],
        index=0, horizontal=True,
        help="If yfinance fails or returns no data for this ticker, switch to Manual "
             "input and paste known ex-dates/amounts from IDX or the broker app you use."
    )
    if dividend_source == "Manual input":
        dividend_manual_input = st.sidebar.text_area(
            "Dividends (one per line: YYYY-MM-DD,amount)",
            placeholder="2025-03-14,360\n2025-08-20,418",
            help="Check IDX keterbukaan informasi (www.idx.co.id) or your broker app's "
                 "dividend/corporate action history for exact figures."
        )
show_index_rebalance = st.sidebar.checkbox("Show MSCI / Index Rebalance Dates", value=True)
show_bi_rdg = st.sidebar.checkbox("Show BI RDG / Policy Rate Decision Dates", value=True)
show_rups = st.sidebar.checkbox("Show RUPS (Shareholders Meeting) Dates — manual input", value=False)
rups_dates_input = ""
if show_rups:
    rups_dates_input = st.sidebar.text_area(
        "RUPS dates (one per line, YYYY-MM-DD)",
        placeholder="2026-04-15\n2026-11-20",
        help="No free public API provides RUPS schedules. Paste dates from IDX "
             "keterbukaan informasi (www.idx.co.id) or the company's investor "
             "relations page."
    )
st.sidebar.caption("MSCI and BI RDG dates below are the **officially published 2026 schedule** "
                   "(verified against MSCI.com and Bank Indonesia press releases as of this app's "
                   "build date) — update the tables in code for other years once new schedules "
                   "are released.")

# ============================================================
# MACRO & FUNDAMENTAL FACTORS (feeds a transparent forecast tilt)
# ============================================================
# Long-standing LQ45 & IDX30 members — used ONLY to pre-tick the membership
# checkboxes below. Verify against the official IDX list (revised every
# Feb/Aug) and untick/tick manually if needed.
KNOWN_CORE_MEMBERS = {"BBCA", "BBRI", "BMRI", "BBNI", "TLKM", "ASII"}

# Proxy basket for the advance/decline calculation. IDX publishes no free
# breadth API, so we compute breadth from this basket via yfinance. Edit it to
# match the current LQ45/IDX30 constituents for a closer proxy.
DEFAULT_BREADTH_BASKET = (
    "BBCA.JK,BBRI.JK,BMRI.JK,BBNI.JK,TLKM.JK,ASII.JK,UNVR.JK,ICBP.JK,INDF.JK,KLBF.JK,"
    "ADRO.JK,ANTM.JK,MDKA.JK,PTBA.JK,CPIN.JK,SMGR.JK,INKP.JK,AMRT.JK,MAPI.JK,EXCL.JK,"
    "TOWR.JK,JSMR.JK,PGAS.JK,ITMG.JK,BRPT.JK,GOTO.JK,MIKA.JK,ACES.JK,INCO.JK,TBIG.JK"
)

st.sidebar.subheader("🧮 Macro & Fundamental Factors")
show_macro_factors = st.sidebar.checkbox("Enable macro & fundamental scorecard", value=True)
macro_tilt_strength = st.sidebar.slider(
    "Evidence tilt strength (%)", 0, 100, 50,
    help="How far the evidence (macro/fundamental scorecard + technical structure) may shift the "
         "Bullish/Sideways/Bearish probabilities away from the no-view base rate. 0% = no tilt. "
         "At 100%, fully supportive evidence shifts the expected horizon return by +0.5 sigma "
         "(roughly 31/38/31 -> 50/35/15; at the default 50% strength, 40/37/23). The scenario price targets never change - only their odds."
) / 100

macro_cfg = {
    "ticker": ticker, "ihsg_symbol": "^JKSE", "lq45_symbol": "^JKLQ45", "idx30_symbol": "",
    "is_lq45": clean_ticker in KNOWN_CORE_MEMBERS, "is_idx30": clean_ticker in KNOWN_CORE_MEMBERS,
    "basket": tuple(t.strip() for t in DEFAULT_BREADTH_BASKET.split(",") if t.strip()),
    "bi_rate": 5.75, "bi_expect_bps": 0,
    "fed_manual": 0.0, "ust10_manual": 0.0, "id10y_manual": 0.0, "id10y_prior_manual": 0.0,
    "roe_manual": 0.0, "roa_manual": 0.0, "pbv_manual": 0.0, "evebitda_manual": 0.0,
    "erp": 5.5, "g": 5.0,
}
if show_macro_factors:
    with st.sidebar.expander("Data sources & manual inputs", expanded=False):
        st.caption("Values left at 0 use the automatic source. Anything you type overrides it.")
        macro_cfg["ihsg_symbol"] = st.text_input("IHSG symbol", value="^JKSE")
        macro_cfg["lq45_symbol"] = st.text_input(
            "LQ45 index symbol", value="^JKLQ45",
            help="Yahoo Finance coverage of Indonesian sub-indices is patchy. Clear it if it returns no data.")
        macro_cfg["idx30_symbol"] = st.text_input(
            "IDX30 index symbol (optional)", value="",
            help="Left blank on purpose — no IDX30 symbol is verified on Yahoo. Fill one in if you find a working symbol.")
        macro_cfg["is_lq45"] = st.checkbox("Stock is an LQ45 constituent", value=macro_cfg["is_lq45"])
        macro_cfg["is_idx30"] = st.checkbox("Stock is an IDX30 constituent", value=macro_cfg["is_idx30"])
        basket_text = st.text_area("Breadth proxy basket (comma-separated Yahoo tickers)", value=DEFAULT_BREADTH_BASKET, height=120)
        macro_cfg["basket"] = tuple(t.strip() for t in basket_text.split(",") if t.strip())

        st.markdown("**Rates & policy**")
        macro_cfg["bi_rate"] = st.number_input("BI Rate (%) — last known: 5.75% held at the Sep 2026 RDG; update after each RDG", value=5.75, step=0.25)
        macro_cfg["bi_expect_bps"] = st.number_input(
            "BI-Rate expectation: expected change over next ~6 months (bps; negative = cuts)", value=0, step=25,
            help="No free API publishes the consensus. Use the latest Bloomberg/Reuters poll, broker research, or BI forward guidance.")
        macro_cfg["fed_manual"] = st.number_input("Fed Funds Rate (%) — 0 = auto (FRED: DFF)", value=0.0, step=0.25)
        macro_cfg["ust10_manual"] = st.number_input("US 10Y Treasury yield (%) — 0 = auto (^TNX)", value=0.0, step=0.05)
        macro_cfg["id10y_manual"] = st.number_input(
            "Indonesia 10Y yield (%) — 0 = auto (FRED OECD monthly series, lagged)", value=0.0, step=0.05)
        macro_cfg["id10y_prior_manual"] = st.number_input(
            "Indonesia 10Y yield ~3 months ago (%) — optional, used for the trend", value=0.0, step=0.05)

        st.markdown("**Fundamentals** (0 = auto from yfinance)")
        macro_cfg["roe_manual"] = st.number_input("ROE (%)", value=0.0, step=0.5)
        macro_cfg["roa_manual"] = st.number_input("ROA (%)", value=0.0, step=0.1)
        macro_cfg["pbv_manual"] = st.number_input("PBV (x)", value=0.0, step=0.1)
        macro_cfg["evebitda_manual"] = st.number_input("EV/EBITDA (x) — not meaningful for banks", value=0.0, step=0.5)
        macro_cfg["erp"] = st.number_input("Equity risk premium (%) — for the justified-PBV check", value=5.5, step=0.5)
        macro_cfg["g"] = st.number_input("Long-term growth g (%) — for the justified-PBV check", value=5.0, step=0.5)

st.sidebar.subheader("🎲 Scenario Probabilities")
show_probabilities = st.sidebar.checkbox("Show Bullish / Sideways / Bearish probabilities", value=True)
prob_band_sigma = st.sidebar.slider(
    "Sideways band (± x horizon sigma)", 0.20, 1.00, 0.50, 0.05,
    help="'Sideways' = the price ends the forecast horizon within +/- this many standard deviations "
         "of the horizon return. Wider band = more probability mass in Sideways. The band is shown "
         "in rupiah in the probability panel.")

plotly_date_format_map = {
    "%b %Y": "%b %Y",
    "%d-%b-%Y": "%d-%b-%Y",
    "%Y-%m-%d": "%Y-%m-%d"
}

# ============================================================
# DYNAMIC ENGLISH TITLE
# ============================================================
st.title(f"📈 {clean_ticker} Macro Pattern, Fibonacci & Cycle Projection")
st.caption(f"Automated Macro Pattern Detection, Dynamic Fibonacci Retracement, and Wave-Based Projection App for {clean_ticker}")

# ============================================================
# LOAD STOCK DATA
# ============================================================
@st.cache_data(ttl=3600)
def load_stock_data(symbol, start, end, tf):
    df = yf.download(symbol, start=start, end=end, interval=tf, auto_adjust=False, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    return df


# ============================================================
# FUNCTION: HISTORICAL S/R ZONES (polarity principle)
# ============================================================
def detect_sr_zones(df, pivot_distance=4, tolerance_pct=0.015, min_touches=2,
                     extend_bars=3, max_span_bars=40):
    """
    Deteksi zona support/resistance horizontal mengikuti prinsip POLARITY
    (old support -> new resistance, dst — lihat referensi gambar "Garis Support
    Berubah menjadi Resistance"):

    1. Cari semua pivot high & low dlm data.
    2. Cluster berdasar KEDEKATAN HARGA (bukan waktu) -> satu "zona harga" yg
       disentuh berkali-kali dari waktu ke waktu, lalu dipecah lagi per
       max_span_bars supaya tiap zona hanya mewakili satu periode aktif
       (bukan satu garis raksasa merentang bertahun-tahun).
    3. Tiap zona digambar HANYA sepanjang rentang tanggal dia aktif disentuh
       (+ sedikit extend_bars ke kanan) -- persis gaya gambar referensi:
       segmen horizontal pendek per zona, bukan satu garis penuh.
    4. Role (support/resistance) ditentukan dari median Close SETELAH zona
       terbentuk: mayoritas di atas -> SUPPORT, mayoritas di bawah ->
       RESISTANCE. If the SAME level reappears in another cluster at a different
       period with a different role, that is the polarity flip being captured
       automatically.

    Return: list of dict {price, start, end, role, touches}
    """
    n = len(df)
    high_idx, _ = find_peaks(df['High'].values, distance=pivot_distance)
    low_idx, _ = find_peaks(-df['Low'].values, distance=pivot_distance)
    high_idx_set = set(high_idx)

    pivots = [(df.index[i], float(df['High'].iloc[i]), i) for i in high_idx]
    pivots += [(df.index[i], float(df['Low'].iloc[i]), i) for i in low_idx]
    if len(pivots) == 0:
        return []
    pivots.sort(key=lambda p: p[1])

    price_clusters = []
    current = [pivots[0]]
    for p in pivots[1:]:
        ref_price = np.median([x[1] for x in current])
        if ref_price > 0 and abs(p[1] - ref_price) / ref_price <= tolerance_pct:
            current.append(p)
        else:
            price_clusters.append(current)
            current = [p]
    price_clusters.append(current)

    zones = []
    for cluster in price_clusters:
        idxs = sorted(set(p[2] for p in cluster))
        # pecah cluster jadi sub-grup berdasar waktu kalau span > max_span_bars
        time_groups = [[idxs[0]]]
        for ix in idxs[1:]:
            if ix - time_groups[-1][-1] <= max_span_bars:
                time_groups[-1].append(ix)
            else:
                time_groups.append([ix])
        for g in time_groups:
            if len(g) < min_touches:
                continue
            actual_prices = []
            for i in g:
                hv, lv = float(df['High'].iloc[i]), float(df['Low'].iloc[i])
                med_ref = np.median([float(df['High'].iloc[j]) if j in high_idx_set
                                      else float(df['Low'].iloc[j]) for j in g])
                actual_prices.append(hv if abs(hv - med_ref) < abs(lv - med_ref) else lv)
            price_level = float(np.median(actual_prices))
            start_idx, end_idx = g[0], min(g[-1] + extend_bars, n - 1)
            start_date, end_date = df.index[start_idx], df.index[end_idx]
            seg_close = df['Close'].iloc[start_idx:end_idx + 1]
            above = int((seg_close > price_level).sum())
            below = int((seg_close < price_level).sum())
            role = 'support' if above >= below else 'resistance'
            zones.append({'price': price_level, 'start': start_date, 'end': end_date,
                          'role': role, 'touches': len(g)})
    zones.sort(key=lambda z: z['start'])
    return zones


# ============================================================
# COMPREHENSIVE CHART PATTERN DETECTOR (20-pattern taxonomy)
# Covers: Reversal (Double/Triple Top/Bottom, Head & Shoulders, Inverse H&S)
#         Continuation (Flag, Pennant, Triangles, Wedges, Expanding/Broadening)
# All within the LAST FIBONACCI CYCLE (from the most recent extreme swing to now)
# ============================================================

def classify_channel_shape(h_slope, l_slope, ref_price, flat_thresh_ratio=0.08,
                            width_flat_thresh_ratio=0.08):
    """
    Classify the GEOMETRIC shape of a two-line channel (upper=resistance from
    swing highs, lower=support from swing lows) using two independent criteria:
      1. Direction of each line (rising / flat / falling), threshold scaled
         to ref_price so it works across any price level.
      2. Whether the channel is CONVERGING / PARALLEL / DIVERGING, from
         width_rate = h_slope - l_slope (negative=narrowing, ~0=parallel,
         positive=widening) — also scaled to ref_price.
    Returns: (shape_name, convergence, h_sign, l_sign)
    """
    flat_thresh = flat_thresh_ratio * ref_price / 100
    width_thresh = width_flat_thresh_ratio * ref_price / 100

    def sign(s):
        if abs(s) < flat_thresh:
            return 0
        return 1 if s > 0 else -1

    hs, ls = sign(h_slope), sign(l_slope)
    width_rate = h_slope - l_slope
    if abs(width_rate) < width_thresh:
        convergence = 'parallel'
    elif width_rate < 0:
        convergence = 'converging'
    else:
        convergence = 'diverging'

    if convergence == 'parallel':
        return 'Flag/Channel', convergence, hs, ls
    if convergence == 'converging':
        if hs == 0 and ls == 1:
            return 'Ascending Triangle', convergence, hs, ls
        if hs == -1 and ls == 0:
            return 'Descending Triangle', convergence, hs, ls
        if hs == 1 and ls == 1:
            return 'Rising Wedge', convergence, hs, ls
        if hs == -1 and ls == -1:
            return 'Falling Wedge', convergence, hs, ls
        if hs == -1 and ls == 1:
            return 'Symmetrical Triangle', convergence, hs, ls
        return 'Converging Channel (mixed)', convergence, hs, ls
    else:
        if hs == 1 and ls == -1:
            return 'Symmetrical Expanding Triangle', convergence, hs, ls
        if hs == 1 and ls == 1:
            return 'Expanding Triangle (Ascending Bias)', convergence, hs, ls
        if hs == -1 and ls == -1:
            return 'Expanding Triangle (Descending Bias)', convergence, hs, ls
        if hs == 0:
            return 'Broadening Formation (flat top)', convergence, hs, ls
        if ls == 0:
            return 'Broadening Formation (flat bottom)', convergence, hs, ls
        return 'Broadening Formation', convergence, hs, ls


def detect_peak_reversal_patterns(cycle, pivot_distance=3, tol_pct=0.03):
    """
    Detect peak/trough-counting reversal patterns within the cycle:
    Double/Triple Top (bearish), Double/Triple Bottom (bullish),
    Head & Shoulders (bearish), Inverse Head & Shoulders (bullish).
    Uses the LAST k comparable swing extremes (k=3 tried before k=2, so
    Head & Shoulders / Triple Top take priority over a looser Double Top
    reading of the same two outer peaks).
    Returns a list of dicts (can be empty, or contain BOTH a top- and a
    bottom-pattern simultaneously if both are present — patterns can be
    nested/overlapping in real price action).
    """
    high_idx, _ = find_peaks(cycle['High'].values, distance=pivot_distance)
    low_idx, _ = find_peaks(-cycle['Low'].values, distance=pivot_distance)
    results = []

    if len(high_idx) >= 2:
        for k in [3, 2]:
            if len(high_idx) >= k:
                last_k_idx = high_idx[-k:]
                last_k_h = cycle['High'].iloc[last_k_idx].values
                ref = np.mean(last_k_h)
                if ref > 0 and np.all(np.abs(last_k_h - ref) / ref <= tol_pct):
                    neckline = float(cycle['Low'].iloc[last_k_idx[0]:last_k_idx[-1] + 1].min())
                    if k == 3:
                        mid_higher = (last_k_h[1] > last_k_h[0] * (1 + tol_pct * 0.5) and
                                      last_k_h[1] > last_k_h[2] * (1 + tol_pct * 0.5))
                        pname = 'Bearish Head and Shoulders' if mid_higher else 'Bearish Triple Top'
                    else:
                        pname = 'Bearish Double Top'
                    results.append({
                        'pattern': pname, 'category': 'reversal', 'direction': 'bearish',
                        'pivot_idx': [int(i) for i in last_k_idx], 'neckline': neckline,
                        'level': float(ref),
                        'dates': [cycle['Date'].iloc[i] for i in last_k_idx],
                        'prices': [float(cycle['High'].iloc[i]) for i in last_k_idx],
                    })
                    break

    if len(low_idx) >= 2:
        for k in [3, 2]:
            if len(low_idx) >= k:
                last_k_idx = low_idx[-k:]
                last_k_l = cycle['Low'].iloc[last_k_idx].values
                ref = np.mean(last_k_l)
                if ref > 0 and np.all(np.abs(last_k_l - ref) / ref <= tol_pct):
                    neckline = float(cycle['High'].iloc[last_k_idx[0]:last_k_idx[-1] + 1].max())
                    if k == 3:
                        mid_lower = (last_k_l[1] < last_k_l[0] * (1 - tol_pct * 0.5) and
                                     last_k_l[1] < last_k_l[2] * (1 - tol_pct * 0.5))
                        pname = 'Bullish Inverse Head and Shoulders' if mid_lower else 'Bullish Triple Bottom'
                    else:
                        pname = 'Bullish Double Bottom'
                    results.append({
                        'pattern': pname, 'category': 'reversal', 'direction': 'bullish',
                        'pivot_idx': [int(i) for i in last_k_idx], 'neckline': neckline,
                        'level': float(ref),
                        'dates': [cycle['Date'].iloc[i] for i in last_k_idx],
                        'prices': [float(cycle['Low'].iloc[i]) for i in last_k_idx],
                    })
                    break
    return results


def detect_chart_pattern(df_daily, cycle_start_idx, search_frac=0.88, min_consol_bars=6,
                          reversal_pivot_distance=3, reversal_tol_pct=0.03):
    """
    MASTER detector combining both pattern families within the last Fibonacci
    cycle (from the most recent extreme swing to now):

      A) Continuation/geometric family (flagpole + consolidation channel):
         Bull/Bear Flag, Bull/Bear Pennant, Ascending/Descending/Symmetrical
         Triangle, Rising/Falling Wedge, (Symmetrical) Expanding Triangle.
         Flagpole = move from cycle start to the most extreme swing point
         found in the first search_frac of the cycle. Consolidation = the
         remaining bars, channel-fitted exactly like the macro channel.

      B) Peak-counting reversal family (Double/Triple Top/Bottom, Head &
         Shoulders / Inverse H&S) — detected independently on the SAME
         cycle, since these can be nested inside the consolidation leg of
         pattern A (e.g. a Double Top forming within a larger Bull Flag).

    Returns a dict with keys 'primary' (pattern A result) and 'secondary'
    (list of pattern B matches, may be empty) so the caller can display both.
    """
    cycle = df_daily.iloc[cycle_start_idx:].reset_index()
    n = len(cycle)
    out = {'primary': {'pattern': None, 'reason': None}, 'secondary': []}
    if n < 10:
        out['primary']['reason'] = f'Cycle too short ({n} days of daily data, need >= 10)'
        return out

    # --- Pattern family B: peak-counting reversal (checked on full cycle) ---
    out['secondary'] = detect_peak_reversal_patterns(cycle, pivot_distance=reversal_pivot_distance,
                                                       tol_pct=reversal_tol_pct)

    # --- Pattern family A: flagpole + consolidation channel ---
    search_end = max(5, int(n * search_frac))
    search_seg = cycle.iloc[:search_end]
    idx_max = int(search_seg['High'].values.argmax())
    idx_min = int(search_seg['Low'].values.argmin())
    val_max, val_min = search_seg['High'].iloc[idx_max], search_seg['Low'].iloc[idx_min]
    start_price = cycle['Close'].iloc[0]

    ret_up = np.log(val_max / start_price)
    ret_down = np.log(val_min / start_price)
    if abs(ret_up) >= abs(ret_down):
        fp_end_idx, fp_ret, fp_dir = idx_max, ret_up, 'UP'
    else:
        fp_end_idx, fp_ret, fp_dir = idx_min, ret_down, 'DOWN'

    fp_start_date, fp_end_date = cycle['Date'].iloc[0], cycle['Date'].iloc[fp_end_idx]
    consol = cycle.iloc[fp_end_idx:].reset_index(drop=True)

    primary = {
        'flagpole_start': fp_start_date, 'flagpole_end': fp_end_date,
        'flagpole_return_pct': fp_ret * 100, 'flagpole_direction': fp_dir,
        'flagpole_start_price': float(start_price),
        'flagpole_end_price': float(val_max if fp_dir == 'UP' else val_min),
        'n_consol_bars': len(consol), 'pattern': None,
    }
    if len(consol) < min_consol_bars:
        primary['reason'] = f'Consolidation leg too short ({len(consol)} days < {min_consol_bars})'
        out['primary'] = primary
        return out

    h_idx, _ = find_peaks(consol['High'].values, distance=2)
    l_idx, _ = find_peaks(-consol['Low'].values, distance=2)
    if len(h_idx) < 2 or len(l_idx) < 2:
        primary['reason'] = 'Not enough pivots in consolidation leg (<2) to fit a channel'
        out['primary'] = primary
        return out

    h_fit = linregress(h_idx, consol['High'].iloc[h_idx].values)
    l_fit = linregress(l_idx, consol['Low'].iloc[l_idx].values)
    ref_price = float(consol['Close'].iloc[0])
    shape, convergence, hs, ls = classify_channel_shape(h_fit.slope, l_fit.slope, ref_price)

    is_short = len(consol) <= 15
    if shape == 'Flag/Channel':
        pattern_name = 'Bullish Flag' if fp_dir == 'UP' else 'Bearish Flag'
        implication = 'bullish continuation' if fp_dir == 'UP' else 'bearish continuation'
    elif shape == 'Symmetrical Triangle' and is_short:
        pattern_name = 'Bullish Pennant' if fp_dir == 'UP' else 'Bearish Pennant'
        implication = 'bullish continuation' if fp_dir == 'UP' else 'bearish continuation'
    elif shape in ('Ascending Triangle',):
        pattern_name = shape
        implication = 'bullish continuation' if fp_dir == 'UP' else 'bullish reversal watch (counter-trend)'
    elif shape in ('Descending Triangle',):
        pattern_name = shape
        implication = 'bearish continuation' if fp_dir == 'DOWN' else 'bearish reversal watch (counter-trend)'
    elif shape == 'Falling Wedge':
        pattern_name = shape
        implication = 'bullish continuation' if fp_dir == 'UP' else 'bullish reversal watch (classic falling-wedge bias)'
    elif shape == 'Rising Wedge':
        pattern_name = shape
        implication = 'bearish continuation' if fp_dir == 'DOWN' else 'bearish reversal watch (classic rising-wedge bias)'
    else:
        pattern_name = shape
        implication = f'ambiguous / watch both directions (after a {fp_dir.lower()} flagpole)'

    primary.update({
        'pattern': f'{pattern_name} ({implication})', 'pattern_short': pattern_name,
        'shape': shape, 'convergence': convergence, 'implication': implication,
        'consol_start': consol['Date'].iloc[0], 'consol_end': consol['Date'].iloc[-1],
        'h_slope': float(h_fit.slope), 'h_intercept': float(h_fit.intercept), 'h_r2': float(h_fit.rvalue ** 2),
        'l_slope': float(l_fit.slope), 'l_intercept': float(l_fit.intercept), 'l_r2': float(l_fit.rvalue ** 2),
        'consol_dates': consol['Date'].values,
    })
    out['primary'] = primary
    return out

# ============================================================
# FUNCTION: DIVIDEND HISTORY (real data via yfinance)
# ============================================================
@st.cache_data(ttl=3600)
def load_dividend_history(symbol):
    """Fetch REAL historical ex-dividend dates and amounts via yfinance.
    Returns a pandas Series indexed by ex-date (may be empty if the ticker
    has never paid a dividend, e.g. a pure index like ^JKSE).

    IMPORTANT: yfinance often returns dividends with a TIMEZONE-AWARE index
    (e.g. Asia/Jakarta for .JK tickers), while the main OHLCV dataframe from
    load_stock_data() is timezone-NAIVE. Comparing the two directly raises
    `TypeError: Invalid comparison between dtype=datetime64[tz] and
    Timestamp`. We strip tz info here so the rest of the app can safely
    compare/filter dividend dates against the chart's date range.
    """
    try:
        div = yf.Ticker(symbol).dividends
        if div is not None and len(div) > 0 and div.index.tz is not None:
            div.index = div.index.tz_localize(None)
        return div if div is not None else pd.Series(dtype=float)
    except Exception as e:
        st.sidebar.warning(f"Could not fetch dividend data for {symbol}: {e}")
        return pd.Series(dtype=float)


def infer_dividend_season(dividend_series, lookback_years=3):
    """From real historical ex-dates, infer the typical MONTH(S) this stock
    tends to pay dividends (e.g. Indonesian banks often pay once in Q1/Q2
    after the annual RUPS). This is a PATTERN from real data, not a
    guaranteed future date."""
    if dividend_series is None or len(dividend_series) == 0:
        return []
    recent = dividend_series[dividend_series.index >= dividend_series.index.max() - pd.DateOffset(years=lookback_years)]
    months = sorted(set(recent.index.month))
    return months


# ============================================================
# OFFICIAL 2026 MACRO CALENDAR (verified via web search against official
# sources — Bank Indonesia press releases and MSCI.com announcements).
# Update this table when BI / MSCI publish schedules for other years.
# ============================================================
BI_RDG_2026 = [
    # (decision announcement date — day 2 of the 2-day RDG, when BI Rate is announced)
    ("2026-01-21", "RDG Jan (Quarterly)"), ("2026-02-19", "RDG Feb"),
    ("2026-03-17", "RDG Mar"), ("2026-04-22", "RDG Apr (Quarterly)"),
    ("2026-05-20", "RDG May"), ("2026-06-18", "RDG Jun"),
    ("2026-07-22", "RDG Jul (Quarterly)"), ("2026-08-19", "RDG Aug"),
    ("2026-09-23", "RDG Sep"), ("2026-10-21", "RDG Oct (Quarterly+Annual)"),
    ("2026-11-18", "RDG Nov"), ("2026-12-16", "RDG Dec"),
]

MSCI_REVIEW_2026 = [
    # (announcement date, effective date, label)
    ("2026-02-10", "2026-02-27", "MSCI Feb Review"),
    ("2026-05-12", "2026-05-29", "MSCI May Review"),
    ("2026-08-12", "2026-08-31", "MSCI Aug Review"),
    ("2026-11-11", "2026-12-01", "MSCI Nov Review"),
]


def parse_dividend_manual_input(text):
    """Parse manually-pasted dividend lines ('YYYY-MM-DD,amount') into the
    same Series format load_dividend_history() returns, so downstream code
    (chart markers, season inference) works identically regardless of source."""
    dates, amounts = [], []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d, a = line.split(",")
            dates.append(pd.Timestamp(d.strip()))
            amounts.append(float(a.strip()))
        except Exception:
            st.sidebar.warning(f"Could not parse dividend line: '{line}' — use 'YYYY-MM-DD,amount'.")
    if not dates:
        return pd.Series(dtype=float)
    return pd.Series(amounts, index=pd.DatetimeIndex(dates)).sort_index()


def parse_rups_input(text):
    dates = []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            dates.append(pd.Timestamp(line))
        except Exception:
            st.sidebar.warning(f"Could not parse RUPS date: '{line}' — use YYYY-MM-DD format.")
    return dates


# ============================================================
# MACRO & FUNDAMENTAL FACTOR ENGINE
# Data layer (cached loaders) -> pure metric helpers -> scorecard -> forecast tilt.
# All loaders fail soft: a missing source yields None and that factor is
# simply left out of the composite (never silently replaced by a made-up value).
# ============================================================
def _tz_naive(idx):
    idx = pd.DatetimeIndex(idx)
    return idx.tz_localize(None) if idx.tz is not None else idx


@st.cache_data(ttl=3600)
def load_market_series(symbol, days_back=420):
    """Daily OHLCV for an index/ticker over a fixed look-back, independent of
    the chart's date range (volatility/volume/relative-strength need ~1 year)."""
    if not symbol:
        return pd.DataFrame()
    try:
        end = pd.Timestamp.now().normalize() + pd.Timedelta(days=1)
        start = end - pd.Timedelta(days=days_back)
        d = yf.download(symbol, start=start, end=end, interval="1d", auto_adjust=False, progress=False)
        if d is None or len(d) == 0:
            return pd.DataFrame()
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = d.columns.get_level_values(0)
        d = d.dropna(subset=["Close"]).copy()
        d.index = _tz_naive(d.index)
        return d
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def load_basket_closes(tickers, days_back=150):
    """Close prices for a basket of tickers (used as the advance/decline proxy)."""
    if not tickers:
        return pd.DataFrame()
    try:
        end = pd.Timestamp.now().normalize() + pd.Timedelta(days=1)
        start = end - pd.Timedelta(days=days_back)
        d = yf.download(list(tickers), start=start, end=end, interval="1d", auto_adjust=False, progress=False)
        if d is None or len(d) == 0:
            return pd.DataFrame()
        close = d["Close"] if isinstance(d.columns, pd.MultiIndex) else d[["Close"]]
        close = close.copy()
        close.index = _tz_naive(close.index)
        return close.dropna(axis=1, how="all")
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def load_fundamentals(symbol):
    """ROE / ROA / PBV / EV-EBITDA from yfinance .info. Fields are frequently
    missing or stale for Indonesian tickers, and EV/EBITDA is generally not
    reported for banks — anything unavailable is returned as None."""
    out = {"roe": None, "roa": None, "pbv": None, "ev_ebitda": None}
    try:
        info = yf.Ticker(symbol).info or {}
    except Exception:
        return out

    def num(k):
        v = info.get(k)
        return float(v) if isinstance(v, (int, float)) and np.isfinite(v) else None

    roe, roa = num("returnOnEquity"), num("returnOnAssets")
    out["roe"] = roe * 100 if roe is not None else None   # yfinance gives a fraction
    out["roa"] = roa * 100 if roa is not None else None
    out["pbv"] = num("priceToBook")
    out["ev_ebitda"] = num("enterpriseToEbitda")
    if out["pbv"] is None:
        price = num("currentPrice") or num("regularMarketPrice")
        book = num("bookValue")
        if price and book:
            out["pbv"] = price / book
    return out


def _parse_fred_df(d):
    """FRED's CSV has 2 columns: observation date + series value ('.' = missing).
    The date column header is 'observation_date' (new) or 'DATE' (old)."""
    date_col, val_col = d.columns[0], d.columns[1]
    d = d.copy()
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    d[val_col] = pd.to_numeric(d[val_col], errors="coerce")
    return d.dropna().set_index(date_col)[val_col].sort_index()


@st.cache_data(ttl=6 * 3600)
def load_fred_series(series_id):
    """Public FRED CSV endpoint (no API key). Returns an empty Series on any failure."""
    try:
        return _parse_fred_df(pd.read_csv(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"))
    except Exception:
        return pd.Series(dtype=float)


# ---------- pure metric helpers ----------
def normalize_tnx(close):
    """Yahoo's ^TNX has been quoted both as yield*10 and as the yield itself."""
    return close / 10.0 if float(close.median()) > 20 else close


def change_over_days(series, days=90):
    s = series.dropna()
    if len(s) < 2:
        return None
    past = s[s.index <= s.index[-1] - pd.Timedelta(days=days)]
    return None if len(past) == 0 else float(s.iloc[-1] - past.iloc[-1])


def realized_vol_stats(close, window=20):
    r = np.log(close / close.shift(1)).dropna()
    if len(r) < window + 60:
        return None
    roll = (r.rolling(window).std() * np.sqrt(252)).dropna()
    cur, med = float(roll.iloc[-1]), float(roll.iloc[-252:].median())
    return {"vol": cur * 100, "median": med * 100, "ratio": cur / med if med > 0 else None}


def volume_stats(df_idx):
    if "Volume" not in df_idx.columns or len(df_idx) < 25:
        return None
    vol = df_idx["Volume"].replace(0, np.nan)
    v5, v20 = vol.iloc[-5:].mean(), vol.iloc[-20:].mean()
    if not (np.isfinite(v5) and np.isfinite(v20)) or v20 <= 0:
        return None
    c = df_idx["Close"]
    return {"ratio": float(v5 / v20), "ret5": float((c.iloc[-1] / c.iloc[-6] - 1) * 100)}


def breadth_stats(closes):
    if closes is None or closes.shape[1] < 8 or len(closes) < 25:
        return None
    pct = closes.pct_change()
    last = pct.iloc[-1]
    daily_net = (pct.iloc[-10:] > 0).sum(axis=1) - (pct.iloc[-10:] < 0).sum(axis=1)
    above = (closes.iloc[-1] > closes.rolling(20).mean().iloc[-1]).mean() * 100
    return {"n": int(closes.shape[1]), "adv": int((last > 0).sum()), "dec": int((last < 0).sum()),
            "net10": int(daily_net.sum()), "pct_above_ma20": float(above)}


def relative_strength_pct(stock_close, bench_close, days=63):
    j = pd.concat([stock_close, bench_close], axis=1, join="inner").dropna()
    if len(j) < days + 1:
        return None
    s = j.iloc[-1, 0] / j.iloc[-days - 1, 0] - 1
    b = j.iloc[-1, 1] / j.iloc[-days - 1, 1] - 1
    return float((s - b) * 100), float(s * 100), float(b * 100)


def trailing_dividend_yield_pct(dividend_series, last_price):
    """TTM dividend / price, computed from the real ex-date history — avoids the
    yfinance `dividendYield` field, whose units changed between versions."""
    if dividend_series is None or len(dividend_series) == 0 or not last_price:
        return None
    d = dividend_series.copy()
    d.index = _tz_naive(d.index)
    ttm = float(d[d.index >= pd.Timestamp.now().normalize() - pd.DateOffset(years=1)].sum())
    return ttm / float(last_price) * 100 if ttm > 0 else None


def justified_pbv(roe_pct, coe_pct, g_pct):
    """Gordon-growth justified P/B = (ROE - g) / (COE - g). Very sensitive to
    its inputs; returns None when COE - g is too small for the formula to be stable."""
    if roe_pct is None or coe_pct is None or (coe_pct - g_pct) <= 0.5:
        return None
    return max((roe_pct - g_pct) / (coe_pct - g_pct), 0.0)


def collect_macro_inputs(cfg, dividend_history, last_price):
    """Gather every factor's raw value. Missing -> None. Manual overrides win."""
    f = {"is_lq45": cfg["is_lq45"], "is_idx30": cfg["is_idx30"], "notes": []}

    ihsg = load_market_series(cfg["ihsg_symbol"])
    f["ihsg_vol"] = realized_vol_stats(ihsg["Close"]) if len(ihsg) else None
    f["ihsg_volume"] = volume_stats(ihsg) if len(ihsg) else None
    if not len(ihsg):
        f["notes"].append(f"No IHSG data for '{cfg['ihsg_symbol']}' (volume & volatility skipped).")

    stock = load_market_series(cfg["ticker"])

    def _rs(sym):
        b = load_market_series(sym) if sym else pd.DataFrame()
        return relative_strength_pct(stock["Close"], b["Close"]) if (len(stock) and len(b)) else None

    f["rs_lq45"] = _rs(cfg["lq45_symbol"])
    f["rs_idx30"] = _rs(cfg["idx30_symbol"])
    f["has_idx30_symbol"] = bool(cfg["idx30_symbol"])
    f["breadth"] = breadth_stats(load_basket_closes(tuple(cfg["basket"])))

    # --- fundamentals ---
    fund = load_fundamentals(cfg["ticker"])
    f["roe"] = cfg["roe_manual"] if cfg["roe_manual"] > 0 else fund["roe"]
    f["roa"] = cfg["roa_manual"] if cfg["roa_manual"] > 0 else fund["roa"]
    f["pbv"] = cfg["pbv_manual"] if cfg["pbv_manual"] > 0 else fund["pbv"]
    f["ev_ebitda"] = cfg["evebitda_manual"] if cfg["evebitda_manual"] > 0 else fund["ev_ebitda"]
    div = dividend_history if (dividend_history is not None and len(dividend_history) > 0) else load_dividend_history(cfg["ticker"])
    f["div_yield"] = trailing_dividend_yield_pct(div, last_price)

    # --- rates ---
    f["bi_rate"], f["bi_expect_bps"] = cfg["bi_rate"], cfg["bi_expect_bps"]

    ust = None
    tnx = load_market_series("^TNX")
    if len(tnx):
        ust = normalize_tnx(tnx["Close"])
    f["ust10"] = cfg["ust10_manual"] if cfg["ust10_manual"] > 0 else (float(ust.iloc[-1]) if ust is not None else None)
    f["ust10_chg3m"] = change_over_days(ust, 90) if (ust is not None and cfg["ust10_manual"] <= 0) else None

    dff = load_fred_series("DFF")
    f["fed"] = cfg["fed_manual"] if cfg["fed_manual"] > 0 else (float(dff.iloc[-1]) if len(dff) else None)
    f["fed_chg6m"] = change_over_days(dff, 180) if (len(dff) and cfg["fed_manual"] <= 0) else None

    idy = load_fred_series("IRLTLT01IDM156N")
    if cfg["id10y_manual"] > 0:
        f["id10y"], f["id10y_src"] = cfg["id10y_manual"], "manual"
        f["id10y_chg3m"] = (cfg["id10y_manual"] - cfg["id10y_prior_manual"]) if cfg["id10y_prior_manual"] > 0 else None
    elif len(idy):
        f["id10y"], f["id10y_src"] = float(idy.iloc[-1]), f"FRED/OECD monthly, as of {idy.index[-1].date()}"
        f["id10y_chg3m"] = change_over_days(idy, 90)
    else:
        f["id10y"], f["id10y_src"], f["id10y_chg3m"] = None, "unavailable", None
        f["notes"].append("Indonesia 10Y yield unavailable — enter it manually in the sidebar.")
    f["spread"] = (f["id10y"] - f["ust10"]) if (f["id10y"] is not None and f["ust10"] is not None) else None

    coe = (f["id10y"] + cfg["erp"]) if f["id10y"] is not None else None
    f["coe"], f["g"] = coe, cfg["g"]
    f["justified_pbv"] = justified_pbv(f["roe"], coe, cfg["g"])
    return f


def build_macro_scorecard(f):
    """Rule-based scorecard. Each factor -> +1 (supportive), 0 (neutral),
    -1 (adverse) or None (unavailable / info only). The composite is the mean of
    the three GROUP means so that the five highly-correlated rates factors
    cannot outvote everything else just by being numerous.
    Returns (rows, composite or None, n_scored, group_scores)."""
    rows = []
    MT, VQ, RP = "Market technicals", "Valuation & quality", "Rates & policy"

    def add(group, factor, latest, signal, rule, info=False):
        rows.append({"Group": group, "Factor": factor, "Latest": latest,
                     "Signal": signal, "Rule": rule, "Info": info})

    def x(v, nd=2, suf=""):
        return "n/a" if v is None else f"{v:.{nd}f}{suf}"

    add("Info", "LQ45 / IDX30 membership",
        f"LQ45: {'yes' if f.get('is_lq45') else 'no'} | IDX30: {'yes' if f.get('is_idx30') else 'no'} (manual — verify on idx.co.id)",
        None, "Context only. Membership is a constant, so scoring it would add a permanent bias; "
              "index-fund flow relevance is captured by relative strength below.", info=True)

    # ---- Market technicals ----
    for key, label, always in (("rs_lq45", "Relative strength vs LQ45 (63d)", True),
                               ("rs_idx30", "Relative strength vs IDX30 (63d)", f.get("has_idx30_symbol"))):
        rs = f.get(key)
        if rs is None:
            if always:
                add(MT, label, "n/a (index data unavailable)", None, "excess > +3pp = +1, < -3pp = -1")
            continue
        ex, s, b = rs
        add(MT, label, f"stock {s:+.1f}% vs index {b:+.1f}% (excess {ex:+.1f}pp)",
            1 if ex > 3 else (-1 if ex < -3 else 0), "excess > +3pp = +1, < -3pp = -1")

    br = f.get("breadth")
    if br is None:
        add(MT, "Market breadth (advance/decline)", "n/a (basket data unavailable)", None, ">60% above 20DMA = +1, <40% = -1")
    else:
        p = br["pct_above_ma20"]
        add(MT, "Market breadth (advance/decline)",
            f"today {br['adv']} adv / {br['dec']} dec; 10d net {br['net10']:+d}; {p:.0f}% of {br['n']} names above 20DMA",
            1 if p > 60 else (-1 if p < 40 else 0),
            ">60% above 20DMA = +1, <40% = -1 (proxy basket, not the full exchange)")

    vs = f.get("ihsg_volume")
    if vs is None:
        add(MT, "IHSG volume", "n/a", None, "5d/20d volume > 1.15x: direction of the 5d move decides +1/-1")
    else:
        sig = 0
        if vs["ratio"] > 1.15 and abs(vs["ret5"]) >= 0.3:
            sig = 1 if vs["ret5"] > 0 else -1
        add(MT, "IHSG volume", f"5d/20d volume {vs['ratio']:.2f}x; IHSG 5d {vs['ret5']:+.1f}%", sig,
            "5d/20d > 1.15x: volume expansion on an up-move = +1, on a down-move = -1, else 0")

    vo = f.get("ihsg_vol")
    if vo is None or vo["ratio"] is None:
        add(MT, "IHSG volatility", "n/a", None, "20d realized vol vs 1y median: >1.3x = -1, <0.8x = +1")
    else:
        add(MT, "IHSG volatility", f"20d realized {vo['vol']:.1f}% ann. (1y median {vo['median']:.1f}%, ratio {vo['ratio']:.2f}x)",
            -1 if vo["ratio"] > 1.3 else (1 if vo["ratio"] < 0.8 else 0), "ratio >1.3x = -1 (stress), <0.8x = +1 (calm)")

    # ---- Valuation & quality ----
    roe, roa = f.get("roe"), f.get("roa")
    add(VQ, "ROE", x(roe, 1, "%"), None if roe is None else (1 if roe >= 15 else (-1 if roe < 10 else 0)), ">=15% = +1, <10% = -1")
    add(VQ, "ROA", x(roa, 2, "%"), None if roa is None else (1 if roa >= 2 else (-1 if roa < 1 else 0)),
        ">=2% = +1, <1% = -1 (generic thresholds; bank ROA norms differ from other sectors)")

    pbv, jp = f.get("pbv"), f.get("justified_pbv")
    if pbv is None:
        add(VQ, "PBV vs justified PBV", "n/a", None, "actual/justified <0.8 = +1, >1.2 = -1")
    elif jp is None or jp <= 0:
        add(VQ, "PBV vs justified PBV", f"PBV {pbv:.2f}x; justified PBV n/a (needs ROE and Indonesia 10Y)", None, "actual/justified <0.8 = +1, >1.2 = -1")
    else:
        ratio = pbv / jp
        add(VQ, "PBV vs justified PBV",
            f"PBV {pbv:.2f}x vs justified {jp:.2f}x (ROE {x(f.get('roe'),1,'%')}, COE {x(f.get('coe'),1,'%')}, g {f.get('g'):.1f}%) -> {ratio:.2f}",
            1 if ratio < 0.8 else (-1 if ratio > 1.2 else 0),
            "actual/justified <0.8 = +1, >1.2 = -1. Justified P/B = (ROE-g)/(COE-g); highly assumption-sensitive")

    ev = f.get("ev_ebitda")
    if ev is None:
        add(VQ, "EV/EBITDA", "n/a (not reported / not meaningful for banks)", None, "<6x = +1, >12x = -1")
    else:
        add(VQ, "EV/EBITDA", f"{ev:.1f}x", 1 if ev < 6 else (-1 if ev > 12 else 0), "<6x = +1, >12x = -1 (generic thresholds)")

    dy, idy = f.get("div_yield"), f.get("id10y")
    if dy is None:
        add(VQ, "Dividend yield (TTM)", "n/a (no dividend in the last 12 months / data unavailable)", None, "yield minus Indonesia 10Y: >=0 = +1, < -3pp = -1")
    elif idy is None:
        add(VQ, "Dividend yield (TTM)", f"{dy:.1f}% (Indonesia 10Y unavailable for comparison)", None, "yield minus Indonesia 10Y: >=0 = +1, < -3pp = -1")
    else:
        d = dy - idy
        add(VQ, "Dividend yield (TTM)", f"{dy:.1f}% vs Indonesia 10Y {idy:.2f}% (gap {d:+.1f}pp)",
            1 if d >= 0 else (-1 if d < -3 else 0), "gap >= 0 = +1, < -3pp = -1")

    # ---- Rates & policy ----
    add(RP, "BI Rate (level)", f"{f.get('bi_rate'):.2f}% (manual input)", None,
        "Level shown for context; the scored signal is the expected direction below.", info=True)
    bps = f.get("bi_expect_bps")
    add(RP, "BI-Rate expectation", f"{bps:+.0f} bps over ~6 months (manual input)",
        None if bps is None else (1 if bps <= -25 else (-1 if bps >= 25 else 0)), "<= -25bps = +1, >= +25bps = -1")

    fed = f.get("fed")
    fchg = f.get("fed_chg6m")
    add(RP, "Fed Funds Rate", x(fed, 2, "%") + (f" (6M change {fchg:+.2f}pp)" if fchg is not None else ""),
        None if fchg is None else (-1 if fchg >= 0.25 else (1 if fchg <= -0.25 else 0)),
        "6M change >= +0.25pp = -1, <= -0.25pp = +1 (needs the automatic FRED series to know the change)")

    ust, uchg = f.get("ust10"), f.get("ust10_chg3m")
    add(RP, "US Treasury 10Y yield", x(ust, 2, "%") + (f" (3M change {uchg:+.2f}pp)" if uchg is not None else ""),
        None if uchg is None else (-1 if uchg >= 0.30 else (1 if uchg <= -0.30 else 0)),
        "3M change >= +0.30pp = -1, <= -0.30pp = +1")

    ichg = f.get("id10y_chg3m")
    add(RP, "Indonesia 10Y yield", f"{x(idy, 2, '%')} [{f.get('id10y_src')}]" + (f" (3M change {ichg:+.2f}pp)" if ichg is not None else ""),
        None if ichg is None else (-1 if ichg >= 0.30 else (1 if ichg <= -0.30 else 0)),
        "3M change >= +0.30pp = -1, <= -0.30pp = +1")

    sp = f.get("spread")
    add(RP, "Yield spread (ID10Y - UST10Y)", x(sp, 2, "pp"),
        None if sp is None else (-1 if sp < 2.0 else (1 if sp > 3.0 else 0)),
        "< 2.0pp = -1 (thin compensation for EM risk, outflow risk), > 3.0pp = +1")

    # ---- composite: mean of group means ----
    group_scores = {}
    for g in (MT, VQ, RP):
        sigs = [r["Signal"] for r in rows if r["Group"] == g and r["Signal"] is not None]
        if sigs:
            group_scores[g] = float(np.mean(sigs))
    n_scored = sum(1 for r in rows if r["Signal"] is not None)
    composite = float(np.mean(list(group_scores.values()))) if group_scores else None
    return rows, composite, n_scored, group_scores


def technical_evidence(pattern_result, price_diff, support_diff):
    """Technical-structure signals taken from what the app already detects.
    Each is in [-1, +1] (continuation patterns = +/-1, 'reversal watch' = +/-0.5)
    or None when there is no information. Returns (rows, score or None, n_scored)."""
    rows = []

    def add(factor, latest, signal, rule):
        rows.append({"Factor": factor, "Latest": latest, "Signal": signal, "Rule": rule})

    prim = ((pattern_result or {}).get("primary")) or {}
    if prim.get("pattern"):
        impl = prim.get("implication", "")
        if "ambiguous" in impl:
            sig = 0.0
        else:
            mag = 0.5 if "reversal watch" in impl else 1.0
            sig = mag if impl.startswith("bullish") else (-mag if impl.startswith("bearish") else 0.0)
        add("Primary chart pattern", f"{prim.get('pattern_short', prim['pattern'])} — {impl}", sig,
            "bullish continuation +1, bearish continuation -1, 'reversal watch' +/-0.5, ambiguous 0")
    else:
        add("Primary chart pattern", "none detected (or detector off)", None, "no pattern = no information")

    if pattern_result is not None:
        sec = pattern_result.get("secondary") or []
        bull = sum(1 for s in sec if s.get("direction") == "bullish")
        bear = sum(1 for s in sec if s.get("direction") == "bearish")
        add("Reversal patterns (peaks/troughs)", f"{bull} bullish / {bear} bearish",
            None if not sec else (1.0 if bull > bear else (-1.0 if bear > bull else 0.0)),
            "more bullish than bearish = +1, the reverse = -1, tie = 0")

    if price_diff is not None and support_diff is not None:
        sig = 1.0 if price_diff > 0 else (-1.0 if support_diff < 0 else 0.0)
        txt = ("above channel resistance (breakout)" if price_diff > 0 else
               "below channel support (breakdown)" if support_diff < 0 else "inside the channel")
        add("Price vs trend channel", txt, sig, "above resistance = +1, below support = -1, inside = 0 (depends on the selected date range)")

    sigs = [r["Signal"] for r in rows if r["Signal"] is not None]
    return rows, (float(np.mean(sigs)) if sigs else None), len(sigs)


def evidence_score(macro_composite, macro_n, macro_n_groups, tech_score, tech_n,
                   min_macro_signals=4, min_macro_groups=2, min_tech_signals=2):
    """Combine the two evidence sources with equal weight. A source only counts
    when it has enough independent signals, so one or two data points can never
    tilt the forecast. Returns (score in [-1, 1], number of sources used)."""
    parts = []
    if macro_composite is not None and macro_n >= min_macro_signals and macro_n_groups >= min_macro_groups:
        parts.append(macro_composite)
    if tech_score is not None and tech_n >= min_tech_signals:
        parts.append(tech_score)
    return (float(np.mean(parts)) if parts else 0.0), len(parts)


def scenario_probabilities(log_returns, last_price, horizon_days, evidence, strength,
                           band_sigma=0.5, n_sim=5000, seed=42, tilt_sigma_max=0.5):
    """Bullish / Sideways / Bearish probabilities for the price at the END of the horizon.

    1. Bootstrap `horizon_days` daily log-returns from the stock's own history, after
       DEMEANING them: the base case carries no directional view, so a past downtrend
       is not silently projected forward (see the regime-dependence issue we hit earlier).
    2. Classification on the horizon return R: Bullish if R > +band, Bearish if R < -band,
       Sideways otherwise, where band = band_sigma x (std of the simulated horizon return).
    3. The evidence score shifts the expected horizon return by
       tilt_sigma_max x strength x evidence x sigma_h  (at most +/-0.5 sigma at full strength).
    These are model-implied probabilities under those assumptions — not calibrated forecasts."""
    r = np.asarray(log_returns, dtype=float)
    r = r[np.isfinite(r)]
    horizon_days = int(horizon_days)
    if len(r) < 120 or horizon_days < 5:
        return None
    r_c = r - r.mean()
    rng = np.random.RandomState(seed)
    R = rng.choice(r_c, size=(n_sim, horizon_days), replace=True).sum(axis=1)
    sigma_h = float(R.std())
    shift = tilt_sigma_max * strength * evidence * sigma_h
    band = band_sigma * sigma_h

    def split(x):
        pb, pr = float((x > band).mean()), float((x < -band).mean())
        return pb, max(0.0, 1.0 - pb - pr), pr

    p_bull, p_side, p_bear = split(R + shift)
    price_end = last_price * np.exp(R + shift)
    p10, p50, p90 = np.percentile(price_end, [10, 50, 90])
    return {"p_bull": p_bull, "p_side": p_side, "p_bear": p_bear, "base": split(R),
            "sigma_h": sigma_h, "band": band, "shift": shift,
            "bull_price": last_price * float(np.exp(band)), "bear_price": last_price * float(np.exp(-band)),
            "p10": float(p10), "p50": float(p50), "p90": float(p90),
            "n_returns": int(len(r)), "horizon_days": horizon_days, "n_sim": n_sim, "price_end": price_end}


try:
    df = load_stock_data(ticker, start_date, end_date, interval)

    if df.empty:
        st.error(f"Stock data for {ticker} not found.")
        st.stop()

    # --- Corporate & Macro Event Calendar data ---
    dividend_history = pd.Series(dtype=float)
    if show_dividend_cal:
        if dividend_source == "Manual input":
            dividend_history = parse_dividend_manual_input(dividend_manual_input)
        else:
            dividend_history = load_dividend_history(ticker)
    dividend_months = infer_dividend_season(dividend_history) if show_dividend_cal else []
    rups_dates = parse_rups_input(rups_dates_input) if show_rups and rups_dates_input else []

    # --- S/R Zones: computed at the same interval as the main chart (per request:
    #     "based on the specified date range" = uses the same start/end date) ---
    sr_zones = []
    if show_sr_zones:
        sr_zones = detect_sr_zones(df, pivot_distance=int(pivot_distance),
                                    tolerance_pct=sr_tolerance_pct,
                                    min_touches=int(sr_min_touches),
                                    max_span_bars=int(sr_max_span_bars))

    # --- Chart Pattern Detector: ALWAYS uses DAILY data (weekly/monthly resolution
    #     is too coarse for these short-lived patterns), even if the main chart
    #     uses a different interval ---
    pattern_result = None
    if show_flag_pennant:
        if interval == "1d":
            df_daily_fp = df
        else:
            df_daily_fp = load_stock_data(ticker, start_date, end_date, "1d")
        if not df_daily_fp.empty:
            cycle_start_idx_fp = int(df_daily_fp['Low'].values.argmin())
            pattern_result = detect_chart_pattern(df_daily_fp, cycle_start_idx_fp)
        else:
            pattern_result = {'primary': {'pattern': None, 'reason': 'Daily data unavailable for pattern detection'},
                              'secondary': []}

    # Calculate Forecast Bars
    if interval == "1d":
        forecast_bars = int(forecast_weeks * 5)
        freq = "B"
    elif interval == "1wk":
        forecast_bars = int(forecast_weeks)
        freq = "W-MON"
    else:
        forecast_bars = max(1, int(round(forecast_weeks / 4.345)))
        freq = "MS"

    dates_historical = df.index
    last_date = dates_historical[-1]
    dates_future = pd.date_range(start=last_date, periods=forecast_bars + 1, freq=freq)[1:]

    # ============================================================
    # PATTERN DETECTION & RESISTANCE TRENDLINE
    # ============================================================
    high = df["High"].to_numpy(dtype=float)
    high_idx, _ = find_peaks(high, distance=int(pivot_distance))

    ath_idx = int(np.argmax(high))
    lh_idx = [int(i) for i in high_idx if i >= ath_idx]

    res_fallback = False
    if len(lh_idx) >= 2:
        res_fit = linregress(lh_idx, df["High"].iloc[lh_idx].values.astype(float))
        res_slope = float(res_fit.slope)
        res_intercept = float(res_fit.intercept)
        r2_val = float(res_fit.rvalue ** 2)
    else:
        res_fallback = True
        res_slope = -5.0
        res_intercept = float(df["High"].max())
        r2_val = None  # tidak ada fit asli -> jangan pura-pura punya R²

    trendline_y = res_slope * np.arange(len(df)) + res_intercept
    current_trendline_val = float(res_slope * (len(df) - 1) + res_intercept)

    # ============================================================
    # SUPPORT TRENDLINE (regression on swing LOWS, same period as resistance)
    # Methodology identical to resistance, but fitted on LOW pivot points
    # setelah ATH -- bukan level Fibonacci datar. Dua garis ini bareng
    # membentuk channel makro (sejajar) atau wedge/triangle (konvergen).
    # ============================================================
    low = df["Low"].to_numpy(dtype=float)
    low_idx, _ = find_peaks(-low, distance=int(pivot_distance))  # lembah = puncak pd -low
    ll_idx = [int(i) for i in low_idx if i >= ath_idx]

    sup_fallback = False
    if len(ll_idx) >= 2:
        sup_fit = linregress(ll_idx, df["Low"].iloc[ll_idx].values.astype(float))
        sup_slope = float(sup_fit.slope)
        sup_intercept = float(sup_fit.intercept)
        sup_r2_val = float(sup_fit.rvalue ** 2)
    else:
        sup_fallback = True
        sup_slope = res_slope
        sup_intercept = res_intercept - (float(df["High"].max()) - float(df["Low"].min())) * 0.3
        sup_r2_val = None

    support_trendline_y = sup_slope * np.arange(len(df)) + sup_intercept
    current_support_val = float(sup_slope * (len(df) - 1) + sup_intercept)

    # Klasifikasi channel: rasio kemiringan mendekati 1 -> sejajar (channel asli)
    if res_slope != 0 and not res_fallback and not sup_fallback:
        slope_ratio = sup_slope / res_slope
        is_parallel_channel = 0.7 <= slope_ratio <= 1.3
    else:
        slope_ratio = None
        is_parallel_channel = None
    channel_width_now = current_trendline_val - current_support_val

    # BREAKOUT / RETEST STATUS (terhadap resistance)
    last_price = float(df["Close"].iloc[-1])
    price_diff = float(last_price - current_trendline_val)
    support_diff = float(last_price - current_support_val)

    if price_diff > 0:
        pattern_status = "BREAKOUT from staircase!"
    elif support_diff < 0:
        pattern_status = "BREAKDOWN below channel support!"
    else:
        pattern_status = "RETEST / Inside Staircase"

    # ============================================================
    # AUTOMATED DYNAMIC FIBONACCI RETRACEMENT
    # ============================================================
    p_high = ath_level if (use_custom_fibo and ath_level > 0) else float(df["High"].max())
    p_low = crash_low_level if (use_custom_fibo and crash_low_level > 0) else float(df["Low"].min())
    diff = p_high - p_low

    fibo_levels = {
        "1.000 (Trend High)": p_high,
        "0.786": p_low + 0.786 * diff,
        "0.618 (Golden Ratio)": p_low + 0.618 * diff,
        "0.500": p_low + 0.500 * diff,
        "0.382": p_low + 0.382 * diff,
        "0.236": p_low + 0.236 * diff,
        "0.000 (Trend Low)": p_low
    }

    # ============================================================
    # WAVE GENERATOR UNTUK PROYEKSI FIBONACCI
    # ============================================================
    future_x_axis = pd.DatetimeIndex([last_date]).append(dates_future)
    steps = len(future_x_axis)
    t = np.linspace(0, 4 * np.pi, steps)

    returns = np.diff(np.log(df["Close"].values))
    volatility = np.std(returns) if len(returns) > 0 else 0.02
    amplitude = last_price * volatility * 1.8

    target_fibo_recovery = fibo_levels["0.618 (Golden Ratio)"]
    target_fibo_bearish = fibo_levels["0.000 (Trend Low)"]

    base_trend_b = np.linspace(last_price, target_fibo_recovery, steps)
    base_trend_a = np.linspace(last_price, target_fibo_bearish, steps)

    wave_component = amplitude * np.sin(t) + (amplitude * 0.4) * np.cos(2.5 * t)
    wave_component = wave_component - wave_component[0]

    y_b_median = base_trend_b + wave_component
    y_a_median = base_trend_a - np.abs(wave_component)

    band_expansion = np.linspace(0, amplitude * 1.5, steps)
    y_b_upper = y_b_median + band_expansion
    y_a_lower = y_a_median - band_expansion

    # --- Evidence -> Bullish / Sideways / Bearish probabilities ---
    macro_rows, macro_composite, macro_n, macro_groups = [], None, 0, {}
    macro_inputs = {"notes": []}
    if show_macro_factors:
        try:
            macro_inputs = collect_macro_inputs(macro_cfg, dividend_history, last_price)
            macro_rows, macro_composite, macro_n, macro_groups = build_macro_scorecard(macro_inputs)
        except Exception as e:  # never let an optional overlay take the whole app down
            macro_inputs = {"notes": [f"Macro scorecard failed and was skipped: {type(e).__name__}: {e}"]}
            macro_rows, macro_composite, macro_n, macro_groups = [], None, 0, {}

    tech_rows, tech_score, tech_n = technical_evidence(pattern_result, price_diff, support_diff)
    evidence, evidence_sources = evidence_score(macro_composite, macro_n, len(macro_groups), tech_score, tech_n)

    prob, prob_note = None, ""
    if show_probabilities:
        try:
            hist_ret = load_market_series(ticker, days_back=800)  # ~2 years of daily returns, independent of chart range
            if len(hist_ret) >= 130:
                close_for_ret = hist_ret["Close"]
            elif interval == "1d" and len(df) >= 130:
                close_for_ret = df["Close"]
            else:
                close_for_ret = None
                prob_note = "Not enough daily history (need >= 130 daily bars) or the data source failed."
            if close_for_ret is not None:
                lr = np.log(close_for_ret / close_for_ret.shift(1)).dropna().values
                prob = scenario_probabilities(lr, last_price, forecast_weeks * 5, evidence,
                                              macro_tilt_strength, prob_band_sigma)
                if prob is None:
                    prob_note = "Not enough usable daily returns (need >= 120) to estimate the probabilities."
        except Exception as e:
            prob, prob_note = None, f"Probability model failed and was skipped: {type(e).__name__}: {e}"

    # Illustrative sideways trajectory: oscillates around the current price at half the wave amplitude
    y_s_median = last_price + 0.5 * wave_component
    if prob is not None:
        pB, pS, pA = prob["p_bull"], prob["p_side"], prob["p_bear"]
        y_weighted = pB * y_b_median + pS * y_s_median + pA * y_a_median
        name_b = f"Scenario B: Bullish / Recovery (Fibo Wave Target) — {pB:.0%}"
        name_s = f"Scenario S: Sideways (range-bound) — {pS:.0%}"
        name_a = f"Scenario A: Bearish (Fibo Wave Target) — {pA:.0%}"
    else:
        pB = pS = pA = None
        y_weighted = None
        name_b = "Scenario B: Recovery (Fibo Wave Target)"
        name_s = None
        name_a = "Scenario A: Bearish (Fibo Wave Target)"

    # ============================================================
    # PLOTLY VISUALIZATION
    # ============================================================
    fig = go.Figure()

    # 1. Historical Candlestick Data
    fig.add_trace(go.Candlestick(
        x=dates_historical,
        open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
        name="Real OHLC",
        increasing_line_color="#10b981", decreasing_line_color="#ef4444"
    ))

    # 2. Lower-High Peaks (hanya kalau resistance aktif -- ini titik fitting-nya)
    if show_macro_resistance and len(lh_idx) > 0:
        fig.add_trace(go.Scatter(
            x=dates_historical[lh_idx],
            y=df["High"].iloc[lh_idx],
            mode="markers",
            marker=dict(symbol="triangle-down", size=9, color="yellow"),
            name="Lower-High Peaks"
        ))

    # 3. Macro Resistance Trendline (toggle-able)
    if show_macro_resistance:
        res_label = f"Macro Resistance Trendline (R²={r2_val:.2f})" if r2_val is not None \
            else "Macro Resistance Trendline (fallback — data lower-high < 2 titik)"
        fig.add_trace(go.Scatter(
            x=dates_historical,
            y=trendline_y,
            mode="lines",
            line=dict(color="#f59e0b", width=2.5),
            name=res_label
        ))

    # 3b. Macro Support Trendline (toggle-able; regresi pada swing low — BUKAN Fibonacci statis)
    if show_macro_support:
        sup_label = f"Macro Support Trendline (R²={sup_r2_val:.2f})" if sup_r2_val is not None \
            else "Macro Support Trendline (fallback — data swing-low < 2 titik)"
        fig.add_trace(go.Scatter(
            x=dates_historical,
            y=support_trendline_y,
            mode="lines",
            line=dict(color="#34d399", width=2.5),
            name=sup_label
        ))

    # 3c. Lower-Low markers (hanya kalau support aktif -- ini titik fitting-nya)
    if show_macro_support and len(ll_idx) > 0:
        fig.add_trace(go.Scatter(
            x=dates_historical[ll_idx],
            y=df["Low"].iloc[ll_idx],
            mode="markers",
            marker=dict(symbol="triangle-up", size=9, color="#34d399"),
            name="Lower-Low Pivots"
        ))

    # 3d. Shading area channel — hanya digambar kalau KEDUA garis aktif
    #     (area "antara resistance & support" tidak bermakna kalau salah satunya disembunyikan)
    if show_macro_resistance and show_macro_support:
        fig.add_trace(go.Scatter(
            x=dates_historical, y=trendline_y, mode="lines",
            line=dict(width=0), showlegend=False, hoverinfo="skip"
        ))
        fig.add_trace(go.Scatter(
            x=dates_historical, y=support_trendline_y, mode="lines",
            line=dict(width=0), fill="tonexty", fillcolor="rgba(156,163,175,0.06)",
            showlegend=False, hoverinfo="skip"
        ))

    # 3e. Historical S/R Zones (segmen horizontal per zona, gaya "polarity")
    if show_sr_zones and len(sr_zones) > 0:
        legend_shown = {'support': False, 'resistance': False}
        for z in sr_zones:
            col = "#34d399" if z['role'] == 'support' else "#f87171"
            show_leg = not legend_shown[z['role']]
            legend_shown[z['role']] = True
            fig.add_trace(go.Scatter(
                x=[z['start'], z['end']], y=[z['price'], z['price']],
                mode="lines", line=dict(color=col, width=2.5, dash="solid"),
                name=f"Historical {z['role'].capitalize()} Zone", legendgroup=z['role'],
                showlegend=show_leg,
                hovertemplate=f"{z['role'].capitalize()} @ Rp{z['price']:,.0f}<br>"
                               f"{z['start'].date()} – {z['end'].date()}<br>"
                               f"Touches: {z['touches']}<extra></extra>"
            ))

    # 3f. Chart Pattern — primary (flagpole + consolidation channel)
    primary_pat = pattern_result['primary'] if (show_flag_pennant and pattern_result is not None) else None
    if primary_pat is not None and primary_pat.get('pattern'):
        fp_color = "#22c9c3" if primary_pat['flagpole_direction'] == 'UP' else "#ef4444"
        fig.add_trace(go.Scatter(
            x=[primary_pat['flagpole_start'], primary_pat['flagpole_end']],
            y=[primary_pat['flagpole_start_price'], primary_pat['flagpole_end_price']],
            mode="lines+markers", line=dict(color=fp_color, width=3, dash="solid"),
            marker=dict(size=8, symbol="diamond"),
            name=f"Flagpole ({primary_pat['flagpole_direction']}, {primary_pat['flagpole_return_pct']:+.1f}%)"
        ))
        cdates = pd.to_datetime(primary_pat['consol_dates'])
        c_idx = np.arange(len(cdates))
        c_res = primary_pat['h_slope'] * c_idx + primary_pat['h_intercept']
        c_sup = primary_pat['l_slope'] * c_idx + primary_pat['l_intercept']
        pat_short = primary_pat['pattern_short']
        fig.add_trace(go.Scatter(x=cdates, y=c_res, mode="lines",
                      line=dict(color="#fbbf24", width=2, dash="dash"),
                      name=f"{pat_short} — Upper line"))
        fig.add_trace(go.Scatter(x=cdates, y=c_sup, mode="lines",
                      line=dict(color="#fbbf24", width=2, dash="dash"),
                      fill="tonexty", fillcolor="rgba(251,191,36,0.1)",
                      name=f"{pat_short} — Lower line"))
        fig.add_annotation(
            x=cdates[len(cdates)//2], y=max(c_res.max(), c_sup.max()),
            text=f"<b>{primary_pat['pattern']}</b>", showarrow=True, arrowhead=2,
            font=dict(color=fp_color, size=13), bgcolor="rgba(0,0,0,0.6)", ay=-40
        )

    # 3g. Chart Pattern — secondary (nested peak-counting reversal patterns:
    #     Double/Triple Top/Bottom, Head & Shoulders / Inverse H&S)
    secondary_pats = pattern_result['secondary'] if (show_flag_pennant and pattern_result is not None) else []
    for sp in secondary_pats:
        col = "#f87171" if sp['direction'] == 'bearish' else "#4ade80"
        fig.add_trace(go.Scatter(
            x=sp['dates'], y=sp['prices'], mode="markers+lines",
            line=dict(color=col, width=1.5, dash="dot"),
            marker=dict(size=10, symbol="x", color=col),
            name=f"{sp['pattern']} — pivots"
        ))
        fig.add_hline(y=sp['neckline'], line_dash="dashdot", line_color=col, opacity=0.6,
                      annotation_text=f"{sp['pattern']} Neckline: {sp['neckline']:,.0f}",
                      annotation_position="bottom left" if sp['direction']=='bullish' else "top left")

    # 3h. Corporate & Macro Event Calendar markers (real/official dates —
    #     high confidence, unlike the optional astrology overlay elsewhere)
    chart_x_min, chart_x_max = dates_historical.min(), future_x_axis.max()

    if show_dividend_cal and len(dividend_history) > 0:
        div_in_range = dividend_history[(dividend_history.index >= chart_x_min) & (dividend_history.index <= chart_x_max)]
        if len(div_in_range) > 0:
            fig.add_trace(go.Scatter(
                x=div_in_range.index, y=[df['Low'].min() * 0.97] * len(div_in_range),
                mode="markers", marker=dict(symbol="diamond", size=10, color="#4ade80"),
                name="Dividend Ex-Date (real)",
                hovertemplate="Ex-Date: %{x|%d %b %Y}<br>Dividend: Rp%{customdata:.0f}<extra></extra>",
                customdata=div_in_range.values
            ))
        if dividend_months:
            season_txt = ", ".join(pd.Timestamp(2000, m, 1).strftime("%B") for m in dividend_months)
            fig.add_annotation(x=dates_historical[-1], y=df['High'].max(),
                               text=f"Historical dividend season: {season_txt}", showarrow=False,
                               font=dict(color="#4ade80", size=10), xanchor="right", yanchor="top")

    if show_index_rebalance:
        for ann_date, eff_date, label in MSCI_REVIEW_2026:
            ann_ts, eff_ts = pd.Timestamp(ann_date), pd.Timestamp(eff_date)
            if chart_x_min <= eff_ts <= chart_x_max:
                fig.add_vline(x=eff_ts, line_dash="dot", line_color="#f472b6", opacity=0.6)
                fig.add_annotation(x=eff_ts, y=df['High'].max(), text=f"{label}<br>(effective)",
                                   showarrow=False, font=dict(color="#f472b6", size=9),
                                   textangle=-90, yanchor="top")
            if chart_x_min <= ann_ts <= chart_x_max:
                fig.add_vline(x=ann_ts, line_dash="dot", line_color="#f472b6", opacity=0.3)

    if show_bi_rdg:
        for rdg_date, label in BI_RDG_2026:
            rdg_ts = pd.Timestamp(rdg_date)
            if chart_x_min <= rdg_ts <= chart_x_max:
                fig.add_vline(x=rdg_ts, line_dash="dashdot", line_color="#60a5fa", opacity=0.5)
                fig.add_annotation(x=rdg_ts, y=df['Low'].min(), text=label, showarrow=False,
                                   font=dict(color="#60a5fa", size=8), textangle=-90, yanchor="bottom")

    if show_rups and rups_dates:
        for rd in rups_dates:
            if chart_x_min <= rd <= chart_x_max:
                fig.add_vline(x=rd, line_dash="solid", line_color="#fb923c", opacity=0.5, line_width=2)
                fig.add_annotation(x=rd, y=df['High'].max(), text="RUPS", showarrow=True, arrowhead=2,
                                   font=dict(color="#fb923c", size=10, family="Arial Black"))

    # 4. Scenario B (Recovery)
    fig.add_trace(go.Scatter(
        x=future_x_axis, y=y_b_upper,
        mode="lines", line=dict(color="rgba(34, 201, 195, 0)"),
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=future_x_axis, y=y_b_median,
        mode="lines", line=dict(color="#22c9c3", width=2.5),
        fill="tonexty", fillcolor="rgba(34, 201, 195, 0.15)",
        name=name_b
    ))

    # 5. Scenario A (Bearish)
    fig.add_trace(go.Scatter(
        x=future_x_axis, y=y_a_lower,
        mode="lines", line=dict(color="rgba(239, 68, 68, 0)"),
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=future_x_axis, y=y_a_median,
        mode="lines", line=dict(color="#ef4444", width=2.5),
        fill="tonexty", fillcolor="rgba(239, 68, 68, 0.15)",
        name=name_a
    ))

    # 5b. Sideways scenario + probability-weighted expected path
    if prob is not None:
        fig.add_trace(go.Scatter(
            x=future_x_axis, y=y_s_median, mode="lines",
            line=dict(color="#9ca3af", width=2, dash="dot"), name=name_s
        ))
        fig.add_trace(go.Scatter(
            x=future_x_axis, y=y_weighted, mode="lines",
            line=dict(color="#f8fafc", width=2.5, dash="dash"),
            name=f"Probability-weighted path (Bull {pB:.0%} / Side {pS:.0%} / Bear {pA:.0%})"
        ))

    # 6. Vertical Forecast Divider
    fig.add_vline(
        x=last_date,
        line_dash="dash", line_color="#9ca3af", line_width=1.5,
        annotation_text="HISTORICAL | FORECAST", annotation_position="top left"
    )

    # 7. Dynamic Fibonacci Lines
    if show_fibo:
        colors_fibo = ["#ef4444", "#f59e0b", "#10b981", "#3b82f6", "#8b5cf6", "#ec4899", "#6b7280"]
        for (lvl_name, val), col in zip(fibo_levels.items(), colors_fibo):
            fig.add_hline(
                y=val, line_dash="dot", line_color=col, opacity=0.6,
                annotation_text=f"Fibo {lvl_name}: {val:,.0f}", annotation_position="bottom right"
            )

    # BEP Level Line (Optional)
    if bep_level > 0:
        fig.add_hline(y=bep_level, line_dash="dashdot", line_color="#818cf8", annotation_text=f"BEP Position: {bep_level:,.0f}")

    # Layout Styling
    fig.update_layout(
        template="plotly_dark",
        title=f"{clean_ticker} — Macro Cycle: Bull Market ➔ Descending Staircase ➔ Breakout/Retest",
        yaxis_title="Price (IDR)",
        xaxis_title="Date Period",
        height=700,
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis_rangeslider_visible=False,
        xaxis=dict(
            type='date',
            tickformat=plotly_date_format_map.get(date_format_option, "%b %Y"),
            # hoverformat TERPISAH dari tickformat: sumbu tetap ringkas (mis. "Sep 2026"),
            # tapi tooltip saat hover selalu tampilkan TANGGAL LENGKAP, bukan ikut tickformat.
            hoverformat="%d %b %Y (%a)"
        )
    )

    st.plotly_chart(fig, use_container_width=True)

    # ============================================================
    # METRICS & ANALYSIS PANEL (STYLED & CLEAN)
    # ============================================================
    st.subheader("🔍 Automated Pattern Analysis")

    if price_diff > 0:
        short_status = "BREAKOUT 🚀"
    elif support_diff < 0:
        short_status = "BREAKDOWN 🔻"
    else:
        short_status = "RETEST 🔄"

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        with st.container(border=True):
            st.caption("Last Price")
            st.subheader(f"Rp {last_price:,.0f}")
    with col2:
        with st.container(border=True):
            st.caption("Resistance (now)")
            st.subheader(f"Rp {current_trendline_val:,.0f}")
    with col3:
        with st.container(border=True):
            st.caption("Support (now)")
            st.subheader(f"Rp {current_support_val:,.0f}")
    with col4:
        with st.container(border=True):
            st.caption("Channel Width")
            st.subheader(f"Rp {channel_width_now:,.0f}")
    with col5:
        with st.container(border=True):
            st.caption("Pattern Status")
            st.subheader(short_status)

    # Baris kedua: klasifikasi channel (sejajar vs konvergen)
    if is_parallel_channel is not None:
        shape_label = "📐 PARALLEL CHANNEL (sejajar)" if is_parallel_channel else "📐 CONVERGING WEDGE/TRIANGLE (konvergen)"
        st.info(f"{shape_label} — rasio kemiringan support/resistance = {slope_ratio:.2f} "
                f"(mendekati 1,0 = sejajar; menjauh dari 1,0 = konvergen)")
    if res_fallback or sup_fallback:
        st.warning("⚠️ One of the lines (resistance/support) is using a FALLBACK value because "
                   "fewer than 2 pivot points were found — try lowering 'Pivot Distance' in the "
                   "sidebar if the historical data range is long enough.")

    # Scenario Probabilities panel
    if show_probabilities:
        with st.expander("🎲 Scenario Probabilities — Bullish / Sideways / Bearish", expanded=True):
            if prob is None:
                st.warning(prob_note or "Probabilities could not be computed.")
            else:
                hd = prob["horizon_days"]
                base_b, base_s, base_a = prob["base"]

                def _dpp(x):
                    v = int(round(x * 100))
                    return f"{v:+d}pp vs no-view base" if v else "no change vs no-view base"
                pc1, pc2, pc3 = st.columns(3)
                pc1.metric("🟢 Bullish", f"{pB:.0%}", delta=_dpp(pB - base_b),
                           help=f"Price ends the {hd}-trading-day horizon above Rp {prob['bull_price']:,.0f}.")
                pc2.metric("🟡 Sideways", f"{pS:.0%}", delta=_dpp(pS - base_s),
                           delta_color="off",
                           help=f"Price ends between Rp {prob['bear_price']:,.0f} and Rp {prob['bull_price']:,.0f}.")
                pc3.metric("🔴 Bearish", f"{pA:.0%}", delta=_dpp(pA - base_a),
                           delta_color="inverse",
                           help=f"Price ends the {hd}-trading-day horizon below Rp {prob['bear_price']:,.0f}.")

                bar = go.Figure()
                for nm, val, col in (("Bullish", pB, "#22c9c3"), ("Sideways", pS, "#9ca3af"), ("Bearish", pA, "#ef4444")):
                    bar.add_trace(go.Bar(x=[val * 100], y=[""], orientation="h", name=nm, marker_color=col,
                                         text=f"{nm} {val:.0%}", textposition="inside", insidetextanchor="middle",
                                         hoverinfo="skip"))
                bar.update_layout(barmode="stack", height=90, template="plotly_dark", showlegend=False,
                                  margin=dict(l=0, r=0, t=0, b=0),
                                  xaxis=dict(range=[0, 100], showticklabels=False, showgrid=False),
                                  yaxis=dict(showticklabels=False))
                st.plotly_chart(bar, use_container_width=True)

                def_df = pd.DataFrame([
                    {"State": "🟢 Bullish", "Definition (price at end of horizon)":
                        f"above Rp {prob['bull_price']:,.0f}  (more than {(prob['bull_price'] / last_price - 1) * 100:+.1f}%)"},
                    {"State": "🟡 Sideways", "Definition (price at end of horizon)":
                        f"between Rp {prob['bear_price']:,.0f} and Rp {prob['bull_price']:,.0f}"},
                    {"State": "🔴 Bearish", "Definition (price at end of horizon)":
                        f"below Rp {prob['bear_price']:,.0f}  (more than {(prob['bear_price'] / last_price - 1) * 100:.1f}%)"},
                ])
                st.dataframe(def_df, use_container_width=True, hide_index=True)

                st.markdown(
                    f"**Simulated price after {hd} trading days:** P10 Rp {prob['p10']:,.0f} · "
                    f"median Rp {prob['p50']:,.0f} · P90 Rp {prob['p90']:,.0f}  "
                    f"(horizon σ ≈ {prob['sigma_h'] * 100:.1f}%, sideways band ±{prob['band'] * 100:.1f}%)")
                p_hit_b = float((prob["price_end"] >= target_fibo_recovery).mean())
                p_hit_a = float((prob["price_end"] <= target_fibo_bearish).mean())
                st.caption(
                    f"Cross-check with the plotted scenarios: P(price ends at or above Scenario B's Fibonacci target "
                    f"Rp {target_fibo_recovery:,.0f}) = {p_hit_b:.0%}; P(ends at or below Scenario A's target "
                    f"Rp {target_fibo_bearish:,.0f}) = {p_hit_a:.0%}. The plotted scenario paths are illustrative "
                    "trajectories toward those targets, not the simulated paths.")
                if target_fibo_recovery < prob["bull_price"]:
                    st.info(f"⚠️ Scenario B's Fibonacci target (Rp {target_fibo_recovery:,.0f}) is **below** the Bullish "
                            f"threshold (Rp {prob['bull_price']:,.0f}), so the plotted recovery path would not by itself "
                            "count as 'Bullish' here. This happens when the selected date range puts the 0.618 level close "
                            "to the current price — widen the Start Date or set custom High/Low levels to move it.")
                if target_fibo_bearish > prob["bear_price"]:
                    st.info(f"⚠️ Scenario A's Fibonacci target (Rp {target_fibo_bearish:,.0f}) is **above** the Bearish "
                            f"threshold (Rp {prob['bear_price']:,.0f}), so the plotted bearish path would not by itself "
                            "count as 'Bearish' here.")

                st.markdown("##### Evidence behind the tilt")
                ec1, ec2, ec3, ec4 = st.columns(4)
                macro_counted = macro_composite is not None and macro_n >= 4 and len(macro_groups) >= 2
                tech_counted = tech_score is not None and tech_n >= 2
                ec1.metric("Macro / fundamental",
                           f"{macro_composite:+.2f}" if macro_counted else "excluded",
                           help="Included only with >= 4 scored factors across >= 2 groups"
                                + ("" if macro_counted else f" (currently {macro_n} scored in {len(macro_groups)} group(s))."))
                ec2.metric("Technical structure",
                           f"{tech_score:+.2f}" if tech_counted else "excluded",
                           help="Included only with >= 2 scored technical signals"
                                + ("" if tech_counted else f" (currently {tech_n})."))
                ec3.metric("Combined evidence", f"{evidence:+.2f}", help=f"{evidence_sources} of 2 sources counted.")
                ec4.metric("Expected-return shift", f"{prob['shift'] * 100:+.1f}%",
                           help="Added to the simulated horizon return: 0.5 x tilt strength x evidence x horizon sigma.")

                def _sig_txt(v):
                    return "⚪ n/a" if v is None else (f"🟢 {v:+.1f}" if v > 0 else (f"🔴 {v:+.1f}" if v < 0 else "🟡 0"))
                st.dataframe(pd.DataFrame([{"Technical signal": r["Factor"], "Latest": r["Latest"],
                                            "Signal": _sig_txt(r["Signal"]), "Rule": r["Rule"]} for r in tech_rows]),
                             use_container_width=True, hide_index=True)

            st.caption(
                "**How these numbers are built.** The base rate bootstraps the stock's own daily returns over the "
                f"horizon after removing their average drift, so with no evidence the odds are simply a function "
                "of volatility and the sideways band (roughly 31/38/31 at the default band) — it carries no "
                "directional view. The evidence score (macro/fundamental + technical structure, equal weight, "
                "each only when it has enough independent signals) then shifts the expected return by at most "
                "+/-0.5 sigma at 100% tilt strength. These are **model-implied probabilities under those "
                "assumptions, not calibrated forecasts**: the signal thresholds are heuristics, returns are assumed "
                "to resemble their recent past, and event risk (earnings, policy shocks) is not modelled. "
                "The technical signals depend on the selected date range.")

    # S/R Zones panel
    if show_sr_zones:
        with st.expander(f"🔁 Historical S/R Zones — {len(sr_zones)} zone(s) detected", expanded=False):
            if len(sr_zones) > 0:
                zone_df = pd.DataFrame([{
                    'Role': z['role'].capitalize(), 'Price': f"Rp {z['price']:,.0f}",
                    'Start': z['start'].date(), 'End': z['end'].date(), 'Touches': z['touches']
                } for z in sr_zones])
                st.dataframe(zone_df, use_container_width=True, hide_index=True)
                st.caption("Polarity principle: the same price level can appear in multiple rows with "
                          "a different Role in different periods — that's the signature of old "
                          "support/resistance switching roles (see the reference 'Support Line Becomes "
                          "Resistance' image).")
            else:
                st.write("No zones detected with the current parameters — try increasing "
                        "'Zone Price Tolerance' or lowering 'Min Touches per Zone'.")

    # Chart Pattern Detector panel (primary geometric pattern + secondary reversal patterns)
    if show_flag_pennant:
        with st.expander("📐 Chart Pattern Detector — Last Fibonacci Cycle", expanded=True):
            if pattern_result is None:
                st.write("No result.")
            else:
                primary_pat = pattern_result['primary']
                secondary_pats = pattern_result['secondary']

                st.markdown("#### 🔷 Primary Pattern (flagpole + consolidation channel)")
                if not primary_pat.get('pattern'):
                    st.warning(f"No primary pattern detected: {primary_pat.get('reason', 'unknown reason')}")
                else:
                    fc1, fc2 = st.columns(2)
                    with fc1:
                        st.markdown(f"**Flagpole ({primary_pat['flagpole_direction']})**")
                        st.write(f"• Period: {primary_pat['flagpole_start'].date()} → {primary_pat['flagpole_end'].date()}")
                        st.write(f"• Return: {primary_pat['flagpole_return_pct']:+.1f}%")
                        st.write(f"• Price: Rp {primary_pat['flagpole_start_price']:,.0f} → Rp {primary_pat['flagpole_end_price']:,.0f}")
                    with fc2:
                        st.markdown(f"**Consolidation ({primary_pat['convergence'].capitalize()})**")
                        st.write(f"• Period: {primary_pat['consol_start'].date()} → {primary_pat['consol_end'].date()} "
                                f"({primary_pat['n_consol_bars']} days)")
                        st.write(f"• Upper line slope: {primary_pat['h_slope']:+.2f}/day (R²={primary_pat['h_r2']:.2f})")
                        st.write(f"• Lower line slope: {primary_pat['l_slope']:+.2f}/day (R²={primary_pat['l_r2']:.2f})")
                    st.success(f"**Pattern Detected: {primary_pat['pattern']}**")
                    st.caption("Classic implication for this shape is noted in parentheses above. This is not "
                              "a guarantee — any of these patterns can fail if the breakout goes the "
                              "opposite way (a 'failed' flag/pennant/triangle/wedge).")

                st.markdown("#### 🔶 Secondary Pattern(s) — nested reversal signals")
                if len(secondary_pats) == 0:
                    st.write("No Double/Triple Top/Bottom or Head & Shoulders pattern detected "
                            "within this cycle.")
                else:
                    for sp in secondary_pats:
                        icon = "🔴" if sp['direction'] == 'bearish' else "🟢"
                        st.info(f"{icon} **{sp['pattern']}** — level ≈ Rp {sp['level']:,.0f}, "
                               f"neckline ≈ Rp {sp['neckline']:,.0f} "
                               f"({sp['dates'][0].date()} → {sp['dates'][-1].date()})")
                    st.caption("Secondary patterns can be NESTED inside the primary pattern's "
                              "consolidation leg (e.g. a Double Top forming within a larger Bull Flag) — "
                              "both readings can be valid at the same time on different scales.")

    # Corporate & Macro Event Calendar panel
    if show_dividend_cal or show_index_rebalance or show_bi_rdg or show_rups:
        with st.expander("📅 Corporate & Macro Event Calendar", expanded=True):
            st.caption("Unlike the optional astrology overlay, these events have well-documented "
                      "causal mechanisms on price: ex-dividend dates cause a mechanical price "
                      "adjustment, index rebalancing moves real passive-fund flows, and policy "
                      "rate decisions directly affect bank/rate-sensitive stocks.")

            if show_dividend_cal:
                source_label = "manual input" if dividend_source == "Manual input" else "yfinance (automatic)"
                st.markdown(f"#### 💰 Dividend History (source: {source_label})")
                if len(dividend_history) == 0:
                    if dividend_source == "Manual input":
                        st.write("No dividend data entered yet — paste lines in the sidebar "
                                "(format: YYYY-MM-DD,amount).")
                    else:
                        st.write(f"No dividend history found for {ticker} via yfinance (may be a "
                                "non-dividend-paying stock, a pure index like ^JKSE, a yfinance/network "
                                "error, or the ticker not yet covered). Try switching the sidebar's "
                                "'Dividend data source' to **Manual input** and paste known ex-dates "
                                "from IDX (www.idx.co.id) or your broker app.")
                else:
                    recent_div = dividend_history.tail(8)
                    div_df = pd.DataFrame({'Ex-Date': recent_div.index.date, 'Amount (Rp)': recent_div.values})
                    st.dataframe(div_df, use_container_width=True, hide_index=True)
                    if dividend_months:
                        season_txt = ", ".join(pd.Timestamp(2000, m, 1).strftime("%B") for m in dividend_months)
                        st.info(f"📌 Based on the last 3 years, {clean_ticker} has historically paid "
                               f"dividends around: **{season_txt}**. This is an observed pattern from "
                               "real data, not a confirmed future date — always verify against the "
                               "company's official announcement.")

            if show_index_rebalance:
                st.markdown("#### 🌍 MSCI Index Review 2026 (officially published schedule)")
                msci_df = pd.DataFrame(MSCI_REVIEW_2026, columns=['Announcement', 'Effective', 'Review'])
                st.dataframe(msci_df, use_container_width=True, hide_index=True)
                st.caption("Source: MSCI.com official press releases. If BBRI/the selected stock is "
                          "part of MSCI Indonesia or MSCI EM, these are the dates passive index funds "
                          "rebalance their holdings — can cause real volume/price impact around the "
                          "effective date.")

            if show_bi_rdg:
                st.markdown("#### 🏦 Bank Indonesia RDG 2026 (officially published schedule)")
                bi_df = pd.DataFrame(BI_RDG_2026, columns=['Decision Date', 'Meeting'])
                st.dataframe(bi_df, use_container_width=True, hide_index=True)
                st.caption("Source: Bank Indonesia official press releases (as announced Dec 2025). "
                          "BI Rate decisions are announced on the 2nd day of each RDG and directly "
                          "affect bank-sector stocks like BBRI/BBCA/BMRI.")

            if show_rups:
                st.markdown("#### 🗳️ RUPS (Shareholders Meeting) — manual input")
                if not rups_dates:
                    st.write("No RUPS dates entered yet. Paste known dates in the sidebar — check "
                            "**www.idx.co.id** (keterbukaan informasi) or the company's investor "
                            "relations page for official schedules.")
                else:
                    st.write(f"{len(rups_dates)} RUPS date(s) marked on the chart: " +
                             ", ".join(d.strftime('%d %b %Y') for d in sorted(rups_dates)))
                    st.caption("RUPS can announce dividends, buybacks, rights issues, or management "
                              "changes — any of which can move the stock. No free public API covers "
                              "this, so dates must be entered manually.")

    # Macro & Fundamental Scorecard panel
    if show_macro_factors:
        with st.expander("🧮 Macro & Fundamental Scorecard — feeds the probability tilt", expanded=True):
            if macro_n == 0:
                st.warning("No factor could be computed — every data source failed or is offline. "
                           "Check your connection, or fill the manual inputs in the sidebar. "
                           "The macro scorecard is left out of the probability tilt.")
            else:
                mc1, mc2, mc3 = st.columns(3)
                mc1.metric("Composite score", f"{macro_composite:+.2f}",
                           help="Mean of the group means, range -1 (adverse) to +1 (supportive).")
                mc2.metric("Factors scored", f"{macro_n} of {sum(1 for r in macro_rows if not r['Info'])}")
                mc3.metric("Counted in the probability tilt?",
                           "Yes" if (macro_n >= 4 and len(macro_groups) >= 2) else "No",
                           help="Needs at least 4 scored factors across at least 2 groups; otherwise the "
                                "scorecard is shown for information but excluded from the tilt.")

                gcols = st.columns(max(len(macro_groups), 1))
                for gc, (gname, gscore) in zip(gcols, macro_groups.items()):
                    gc.metric(gname, f"{gscore:+.2f}")

                sig_label = {1: "🟢 +1", 0: "🟡 0", -1: "🔴 -1", None: "⚪ n/a"}
                score_df = pd.DataFrame([{
                    "Group": r["Group"], "Factor": r["Factor"], "Latest": r["Latest"],
                    "Signal": "ℹ️ info" if r["Info"] else sig_label[r["Signal"]], "Rule": r["Rule"]
                } for r in macro_rows])
                st.dataframe(score_df, use_container_width=True, hide_index=True)

            for note in macro_inputs.get("notes", []):
                st.warning(note)

            st.caption(
                "**How to read this.** Each factor is a rule-of-thumb signal (+1 supportive / 0 neutral / "
                "-1 adverse). The thresholds are heuristics, **not calibrated against this stock's own "
                "history**, so treat the composite as a structured checklist, not a statistical forecast. "
                "The composite is one of two evidence sources (the other is technical structure) that tilt "
                "the Bullish/Sideways/Bearish probabilities; the scenario price targets never change. "
                "It averages three groups equally "
                "so the five correlated rates factors cannot dominate by sheer count. "
                "Data caveats: yfinance fundamentals are often missing/stale for IDX tickers; EV/EBITDA is "
                "generally not meaningful for banks; the Indonesia 10Y series on FRED is monthly and lags "
                "(override it manually when you have a fresher quote); BI Rate and BI-Rate expectation have "
                "no free API and are manual inputs."
            )

    st.write("")  # Spacer

    # Card Ringkasan Struktur Pasar
    st.subheader(f"📊 Market Structure Summary for {clean_ticker}")
    
    summary_col1, summary_col2 = st.columns(2)

    with summary_col1:
        with st.container(border=True):
            st.markdown("#### 📉 Pattern & Signal Status")
            st.write(f"• **Primary Pattern:** Descending Staircase")
            st.write(f"• **Trend Reference:** Peaks identified since High at **Rp {p_high:,.0f}**")
            st.write(f"• **Current Trading Price:** **Rp {last_price:,.0f}**")
            st.write(f"• **Signal Output:** Price is currently in **{pattern_status}** relative to macro trendline (**Rp {current_trendline_val:,.0f}**).")
            st.write(f"• **Support Trendline:** **Rp {current_support_val:,.0f}** "
                     f"({'parallel to resistance — channel' if is_parallel_channel else 'converging — wedge/triangle' if is_parallel_channel is not None else 'fallback'})")

    with summary_col2:
        with st.container(border=True):
            st.markdown("#### 🎯 Key Wave Forecast Targets")
            st.write(f"• 🟢 **Bullish Wave Target (Fibo 0.618):** Rp {fibo_levels['0.618 (Golden Ratio)']:,.0f}")
            st.write(f"• 🔴 **Bearish Wave Target (Fibo 0.000):** Rp {fibo_levels['0.000 (Trend Low)']:,.0f}")
            st.write(f"• 🟡 **Resistance Level (Fibo 0.382):** Rp {fibo_levels['0.382']:,.0f}")
            st.write(f"• 🔵 **High Resistance (Fibo 1.000):** Rp {p_high:,.0f}")

except Exception as e:
    st.error(f"Failed to process analysis: {e}")