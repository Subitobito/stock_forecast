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

# Default saat pertama kali dibuka / restart: BBRI.JK, histori sejak 1 Jan 2026
# s/d hari ini, interval harian — mereplikasi tampilan "siap pakai" yang biasa
# dipakai (lihat bbri-macro-cycle-bull-market-descending.png), supaya tidak
# perlu setup ulang toggle satu-satu tiap buka aplikasi.
default_end = pd.Timestamp.now().normalize()
default_start = default_end - pd.DateOffset(months=6)

ticker = st.sidebar.text_input("Stock Ticker", value="^JKSE")

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
        name="Scenario B: Recovery (Fibo Wave Target)"
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
        name="Scenario A: Bearish (Fibo Wave Target)"
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