# Changelog: Macro Pattern & Fibonacci Analyzer (+ IDX-API sync companion)

**How to read this log**

- Newest first. Dates are **UTC**, taken from file timestamps (WIB = UTC + 7 h).
- **Verified** = I actually ran it. **Not verified** = could not be tested in my sandbox (it has no access to idx.co.id, no Windows, no real browser).
- Everything before v1.4.0 was never version-numbered; that section is reconstructed from this conversation and is less precise.
- IHSG (Indeks Harga Saham Gabungan) is the Jakarta Composite Index. Since 1.4.6 the app calls it **JCI**; older entries below keep the name IHSG.

## At a glance

| Version | Date (UTC) | Theme | Needs |
|---|---|---|---|
| **1.5.0** | 2026-10-15 | Deep Forecast: Random Forest, XGBoost and SARIMAX, validated walk-forward | `pip install -r requirements.txt` (scikit-learn, xgboost, statsmodels) |
| **1.4.7** | 2026-10-12 | macOS-style dark look (CSS) | none |
| **1.4.6** | 2026-10-06 | Company logo and name in the header; all text in English | the IDX-API `sample/img` folder (part of the download) |
| **1.4.5** | 2026-10-06 | IDX-API source switches itself on at the first launch | Cron.ts with the `idx-cron-compat` marker |
| **1.4.4** | 2026-10-06 | Background auto-update of the database | Deno + the modified Cron.ts |
| **1.4.3** | 2026-10-06 | End Date inclusive; IDX-API = additional information only; refresh | none |
| **1.4.2** | 2026-10-06 | Longer IHSG history from IDX-API | none |
| **1.4.1** | 2026-10-05 | IDX-API SQLite integration repaired | none |
| **1.4.0** | n/a (your upload) | Baseline with the first IDX-API adapter | none |
| < 1.4.0 | n/a | Unversioned history | n/a |

---

## 1.5.0: 2026-10-15

**Added: Deep Forecast (ML)**, in the sidebar (🤖) and as a panel under the Forecast Engine; the code is in `deep_forecast.py` next to the app
- Random Forest and XGBoost (scikit-learn pipelines: median imputer + scaler) predict the forward return over the same horizon as the engine, and classify Bullish / Neutral / Bearish with the same neutral band. SARIMAX (statsmodels) forecasts the log price with an 80% band.
- 20 technical features, all computed from data up to the prediction day (returns, volatility, RSI, MACD, moving-average distances, Bollinger %B, ATR %, volume ratio, 52-week distances).
- Validation is walk-forward (`TimeSeriesSplit`, `gap` = horizon) against a naive baseline. The panel shows skill vs baseline, direction hit-rate and class accuracy vs always guessing the majority class; a warning appears when no model beats the baseline.
- Ranges come from out-of-fold residuals; class probabilities are shrunk toward the historical base rate by the classifier's out-of-sample advantage.
- Chart overlay: SARIMAX path with its band, and each model's target with its 80% range at the horizon end.
- Verified: a pure random walk shows negative skill and triggers the warning; a series with a known drift is recovered by SARIMAX; the full app runs on ^JKSE and BBRI.JK without errors.

**Added: Forecast Engine + Deep Forecast combined** (Streamlit and the HTML UI; one rule set, `deep_forecast.combine` in Python and `combineED` in the page)
- Skill-weighted blend of the scenario probabilities: weight = cap (default 40%) x the classifiers' out-of-sample advantage over the base rate, and 0 when no ML model beats the naive baseline, so the engine is left untouched for stocks where ML shows no edge. The blend feeds the probability-weighted path and the scenario labels (engine % -> blended %).
- Conviction score 0-100 from a weighted vote (engine 0.40, each ML model 0.20, SARIMAX 0.20; a voter without out-of-sample skill counts a quarter), shown on the chart and in the Deep Forecast panel with the individual votes, plus a combined expected price and range.
- Verified: the Python and JavaScript implementations give identical results on 200 random cases (max relative difference 3e-16).

**Changed: the Fibonacci forecast line uses a fixed window** (the last year of daily data: 250 bars) for the pattern, the Fibonacci levels and the cycle trendlines, so it no longer changes with the chart period (3M/6M/1Y/2Y; in Streamlit, the start date). The patterns and Fibonacci levels drawn on the chart still follow the chart period.

**Added: A-B-C correction after the pattern target** (A -23.6% of the move, B recovers 50% of A, C ends at the 38.2% retracement; spread over 0.7 x the flagpole duration after the target, flat afterwards).

**Changed: the dashed pattern line moves in Fibonacci waves** (`fib_wave`): 5 waves to the target, wave 1 = 40% of the move, wave 2 retraces 50%, wave 3 extends 1.618x wave 1, wave 4 retraces 38.2%, wave 5 ends at the target, over the flagpole duration. The peaks and troughs show up as events.

**Changed: two separate forecasts** (Streamlit and the HTML UI). 'Forecast line' = a DASHED line, the pattern projection (smooth move to the Fibonacci / measured-move target of the detected pattern; the engine's tilt when no pattern is detected) with the dates of the breakout / breakdown. 'Include the ML models' = realistic CANDLES whose trend is the ML models only (Random Forest + XGBoost + SARIMAX, skill-weighted). They are independent switches; the all-methods consensus stays as a reference in the tooltip and the panel (`pattern_line`, `consensus_line(ml_only=True)` in `deep_forecast.py`).

**Changed: the forecast is drawn as realistic candles** (Streamlit and the HTML UI). The closes oscillate around the consensus line (mean-reverting random walk driven by the stock's own daily returns), open = previous close, wicks drawn from the real wicks of the last 250 candles (`forecast_candles` in `deep_forecast.py`). Seeded with the last bar's date, so it is stable when the horizon changes and identical in both UIs. The consensus line stays as a thin dotted 'trend' line (the expected value; the events are measured on it).

**Changed: ONE forecast line replaces the scenario lines** (Streamlit and the HTML UI)
- The Bullish / Neutral / Bearish lines, the probability-weighted line, the SARIMAX band, the ML markers, the 'Smooth scenario lines' switch, the 'Scenario line style' radio and the 'Monte Carlo paths' switch are gone. The chart shows one line: the weighted mean, day by day, of the forecast engine (0.30), the Fibonacci pattern projection (0.20), Random Forest (0.15), XGBoost (0.15) and SARIMAX (0.20). Components without proven skill count a quarter; missing components are left out and the weights renormalised.
- Fibonacci pattern component: the flagpole height projected from the breakout side of the pattern (reversal patterns: their own measured move), snapped to a Fibonacci level within 4%, reached smoothly over the flagpole's duration.
- Projected timing: the dates at which the line crosses the cycle / macro trendlines (breakout, breakdown), the pattern lines and the Fibonacci levels, and its turning points. Shown on the chart and in a table (Deep Forecast tab / the 'Forecast line' panel).
- The scenario PROBABILITIES stay (inspector, Forecast Engine panel, blend and conviction); the Bull / Neutral / Bear percentages are printed under the end of the line.
- Stable when 'weeks ahead' changes: the ML components come from models trained for horizons 5-260 days (independent of the chosen horizon), their skill is measured once at a 40-day reference, and SARIMAX uses a path of at least 130 days with a fixed 60-day hold-out. The Fibonacci component is not clipped to the horizon. The HTML page uses the latest finished ML result of the ticker, so the line does not jump while a new horizon trains.
- Fixed: with a 26-week horizon `TimeSeriesSplit` did not fit 5 folds (gap = horizon), so the ML models failed and the ML curve disappeared; the number of folds now shrinks until it fits.
- Verified: Python and JavaScript produce identical lines, components and events on 150 random cases (max relative difference 3.5e-16).

**Changed: scenario lines are data-driven and no longer jump when 'weeks ahead' changes** (Streamlit and the HTML UI)
- Before: the Bullish / Neutral / Bearish lines were a straight line to the target plus a decorative sine wave whose period was rescaled by the horizon, and the simulation seed depended on the horizon, so the whole line changed shape with every 'weeks ahead' change.
- Now: each line is the day-by-day median of simulated price paths (bootstrapped returns, FIXED seed over the longest horizon), with a P25 / P75 band. The Neutral line and the probability-weighted line are stable when the horizon changes. The Bullish / Bearish lines are conditional on ending beyond the neutral band at the chosen horizon, so they still reach further for longer horizons.
- Line style (default: realistic): the median path is smooth and almost straight, because the random up and down moves of thousands of paths average out. 'Realistic' draws ONE simulated path per scenario that ends at that scenario's median target (the 30 nearest by end value, then the median roughness), so the line shows plausible ups and downs. The timing of those moves is random: nothing in the engine predicts when the price rises or falls. Switch: sidebar 'Scenario line style' (Streamlit) / 'Realistic scenario lines' (HTML).
- ML in the lines: `ml_curve` trains a Random Forest and an XGBoost regressor for each horizon from 5 to 260 days and interpolates their predictions into an expected cumulative-return curve. With the blend on, the path drift = (1 - w) x the engine tilt + w x this curve, where w is the skill-gated blend weight (0 when ML shows no edge). The curve is also drawn as 'ML expected path' in the Deep Forecast overlay.

**Notes**
- With ~3 years of daily data there are only about 7 independent 80-day windows, so the scores are noisy and regime-dependent. Treat the output as a research aid, not a calibrated forecast.
- SARIMAX is fitted on 100 x log price; fitted on the raw log price its drift term did not converge on real data.

---

## 1.4.6: 2026-10-06

**Changed: everything the app shows is in English** (a revised build of 1.4.6, same number, at your request)
- Checked three ways: a word list, a language detector, and a word-frequency test (words common in Indonesian but absent from English), first on every string and comment in the source, then on the text the running app actually renders (widget labels and help texts, tables, chart legends and annotations; 932 distinct strings over six scenarios). Before: 15 × RDG, 8 × RUPS, 5 × IHSG, 1 × BEP, "keterbukaan informasi", and the channel card in mixed Indonesian/English. After: none left.

| Was | Now |
|---|---|
| RDG (Rapat Dewan Gubernur): sidebar toggle, chart labels, calendar table, BI-Rate label | **BI Board of Governors' meetings**; chart and table labels read "BI Meeting Apr (Quarterly)" etc. |
| RUPS: sidebar toggle and input, chart annotation, warnings, section | **AGM/EGM (shareholders' meeting)** |
| IHSG: sidebar label, regime rows, cross-check table, messages | **JCI** (sidebar: "JCI symbol (Jakarta Composite Index)") |
| BEP Position (Optional) | **Break-even price (BEP, optional)**; chart line "Break-even price: ..." |
| "keterbukaan informasi" | **IDX disclosures** |
| Channel card: "PARALLEL CHANNEL (sejajar) ... rasio kemiringan ... mendekati 1,0 ..." | "PARALLEL CHANNEL" / "CONVERGING WEDGE/TRIANGLE", "support/resistance slope ratio = 0.97 (close to 1.0 = parallel; further from 1.0 = converging)" |
| Fallback trendline labels "... < 2 titik" | "... < 2 points" |
| "09:00–15:50 WIB" in the intraday-bar note | "09:00–15:50 Jakarta time" |
| 21 code comments and one docstring in Indonesian | English |

**Fixed**: the Data Health caption still said stale data is ignored for the IHSG and "by the chart price guard" (both removed in 1.4.3) and told you to edit a hard-coded date range in Cron.ts (the modified Cron.ts computes it, and the app has an update button). It now says what really happens.

**Kept on purpose**: **Rp** (the rupiah symbol; the chart axis says IDR), **Tbk** and **Persero** (legal forms inside company names), the index names **LQ45** and **IDX30**, **MSCI**, and internal variable names such as `show_rups` and `BI_RDG_2026`, which no user sees.

**Not covered**: text that Streamlit itself draws in the browser's language (for example "Choose options" in a dropdown), and company names or fundamentals supplied by yfinance or IDX.

**Revision of the logo header, same day: frame removed, logo 20% larger**
- The first build put the logo on a white rounded square. The reason I gave (dark-text logos would vanish on the dark theme) was **wrong**: 933 of the 955 logo files are round badges with their own background, so the square was a second, unnecessary frame around a round one. Measured on the dark theme, only 16 files (1.7%) have mostly low-contrast artwork (BMRI is the only one among your 35, and it stays legible: white text and a yellow mark on a navy disc).
- Size: the visible logo went from 48 px to **58 px (+20%)**. Because the frame is gone, the whole element is a little smaller than the old 64 px square. If you meant +20% of that square (77 px), change the single constant `IDX_LOGO_PX`.
- No toggle was added for a backing: none of your 35 stocks needs one.

**Added: company logo and name in the header**
- The logo is the SVG file `<IDX-API folder>/sample/img/<CODE>.svg`. `STOCKLIST.md` only **references** these images (it also lists code, name and listing date), so the logo is found by file name, next to the database the app already knows. It works with `STOCKLIST.md` placed in `data/` while the images stay in `sample/img/`, as on your machine.
- The logo is shown **alone: no frame, no background, no border**, at **58 px** (the 48 px logo that used to sit inside the white frame, +20%).
- Under the title: company name and listing date. Name order: full name from `company_profile` (when that table is filled) → name from `STOCKLIST.md`.
- Stocks without a logo (NANO and SMKM are marked `_N/A_`) or non-IDX tickers get the plain title with the 📈 emoji, never an error and never an empty frame.
- Sidebar **🖼️ Stock logo**: show/hide, optional logo folder, and a line saying which file was found.
- New adapter method `company_name()`; `company_profile` added to the table map.

**Data facts found while building it** (from your copy of the repo)
- `STOCKLIST.md`: 956 stocks; 955 logo files. 954 rows reference a logo, NANO and SMKM say `_N/A_`, and `MFIN.svg` exists without a row. All 35 stocks on your list have a logo.
- Company names in `STOCKLIST.md` are **cut at 30 characters** (e.g. "Bank Rakyat Indonesia (Persero"). A cut name gets an ellipsis. Detection uses the raw length (19 names are cut right after a space, which `strip()` would hide) and treats a name ending in "Tbk" as complete (NANO's full name is exactly 30 characters). 127 names are flagged. This is a heuristic.
- The logos are real brand artwork, 3 KB to 67 KB each for your 35 stocks; none contains a script or an external reference. **933 of the 955 files are round badges that carry their own background** (white or coloured disc), so they need no frame on either theme.

**Safety**
- The ticker is validated (`^[A-Z0-9]{2,6}$`) before it becomes part of a path, so text such as `../../etc/passwd` can never reach the file system.
- Only plain SVG up to 600 KB is shown (a file containing `<script`, or an oversized one, is refused), embedded as an `<img>` data URI.

**Known limitation**
- The browser-tab icon cannot follow the ticker: Streamlit requires `set_page_config` to run before the ticker is known.

**Verified**: 18 checks on the helpers (including your folder layout, malformed and hostile codes, SVG limits, escaping); 8 end-to-end scenarios in the full app (BBRI byte-for-byte logo, truncated vs full name, TLKM frameless, NANO plain title, logo switched off, folder typed in the sidebar, path-like ticker text, no IDX-API folder at all); the 9-scenario regression matrix and the date/refresh tests, all re-run after the English pass; the rendered English texts were displayed and checked (channel card, AGM/EGM warning, BI meeting table, chart annotations).

**Not verified**: how the frameless logo looks in a real browser (size, vertical alignment, your theme); the fallback branch for Streamlit older than 1.36 (`vertical_alignment`).

---

## 1.4.5: 2026-10-06

**Fixed**
- The "Enable IDX-API SQLite source" checkbox started **unticked in every new browser session**, so the auto-update of 1.4.4 never ran on the first launch. It now switches itself on when the **live** database (`data/database.sqlite`) is found next to the script and opens.
- The Git-LFS sample snapshot (`sample/sample-database.sqlite`) never turns the source on by itself: it is old and a sync never updates it. The check is on the file name `database.sqlite`.

**Added**
- Sidebar note "Enabled automatically: a live database was found at ...".

**Behaviour to know**
- With the source on, foreign flow, dividends and fundamental ratios from the database enter the analysis. Stale data is still ignored (limit set in the sidebar). Trend, price and IHSG still come only from yfinance. Untick the box to ignore the database completely.

**Verified**: five first-launch scenarios (live database a few days behind → sync starts by itself; a second session while it runs → no second sync; database already current → "not needed"; sample-only → stays off; original Cron.ts → refused with an explanation) and the 9-scenario regression matrix, all with 0 exceptions.

---

## 1.4.4: 2026-10-06

**Added: background auto-update (`IDXAutoSync`)**
- Starts `deno run -A Cron.ts --days=N --no-per-company` **detached** when the database is behind the last trading session (today after 17:00 WIB, otherwise the previous weekday; exchange holidays are unknown).
- `N` = how far behind the database is + 10 days, between 14 and 420; an empty database gets 365.
- Never blocks the app: the analysis renders at once with the data that exists; a status fragment polls every 5 s and reruns the app when the sync finishes.
- Sidebar: auto-update checkbox, days (0 = automatic), per-company toggle, minimum hours between automatic attempts (default 6), **⬇️ Update database now**, progress bar, log tail; Data Health shows the last sync attempt.
- Progress is read from the log (`N requests in total` and one `[Sync] Starting` line per request). State is kept in `data/autosync_state.json` and `data/autosync.log`, so a restarted Streamlit reads the same status.
- States: idle / running / finished_ok / finished_warn / failed / stalled. A dead run is never shown as "running" forever.

**Safety refusals** (it will not run in these cases and says why)
- **Original Cron.ts** (it ignores `--days` and would fetch about 960 companies over a hard-coded range). The modified Cron.ts carries the marker line `idx-cron-compat: 2`.
- **Database path is not `<IDX-API folder>/data/database.sqlite`**, the only file a sync writes.
- **Deno not found.**
- A second start while one is running is refused.

**Changed**
- "Reload IDX-API database" no longer clears the resource cache that holds the running process handle.
- Companion: `Cron.ts` gained the marker line (see the Cron.ts section).

**Verified (real Deno 2.9.6 in the sandbox)**: 18 checks on the class, including a real background run (running → finished_warn, exit code 2, progress 40 of 40, non-blocking start in under 5 s, throttle, restart-safe state); stalled/failed classification; six app scenarios (stale DB auto-starts, current DB does not, wrong path refused, original Cron.ts refused, manual button twice, IDX-API disabled); 9-scenario regression.

**Not verified**: Windows process flags (`CREATE_NO_WINDOW`, new process group); the `st.fragment(run_every=5)` timer and the automatic full rerun in a real browser; a successful download (the sandbox cannot reach IDX, so every run ended "no newer prices", reported honestly).

---

## 1.4.3: 2026-10-06

**Fixed**
- **End Date was effectively exclusive.** yfinance's `end` is exclusive, so End Date = 6 Oct stopped at the 5 Oct bar (shown with test data: last candle 5 Oct close 3,090 instead of 6 Oct 3,115). The forecast's anchor price came from that bar. End Date is now inclusive.
- An empty yfinance download no longer crashes with a `KeyError`; it shows "Stock data ... not found".
- The index is named `Date` at the source (`load_stock_data`), so the pattern detectors work for every interval.

**Changed**
- **IDX-API is additional information only** (foreign flow, dividends, fundamental ratios). It no longer replaces the chart price (1.4.1 guard) or the IHSG (1.4.2). Prices, trends, indicators, Fibonacci and IHSG always come from yfinance. Shown by comparing 123 candles with IDX-API off and on: identical.
- Cache lifetime of price loaders (stock, market series, breadth basket) 1 h → **5 min**. Fundamentals and dividends stay 1 h, FRED 6 h.

**Added**
- Data Health **cross-check table**, yfinance vs IDX-API: last bar, close on the latest common day and the difference (flag above 0.5%), and who is newer. Information only.
- **🔄 Refresh market data** button (clears the data cache).
- Note when the last daily bar is dated today while the IDX session is open (09:00–15:50 WIB, Mon–Fri): the bar is still forming, so RSI/MACD and the forecast anchor move with it (`bar_is_forming`).
- Caption under the chart: price source, last bar date and close.

**Known limitation**
- End Date drives the **chart and the anchor price only**. The engine (IHSG, USD/IDR, indicators, bootstrap) still uses data up to today. Identical when End Date = today; a past End Date does not roll the engine back.

**Verified**: inclusive End Date; identical candles with IDX-API on/off; cross-check table; 5-minute cache plus refresh (an updated bar appears only after the click); `bar_is_forming` across open session / after close / before open / older bar / weekend; empty download; 9-scenario matrix.

---

## 1.4.2: 2026-10-06

**Added**
- IHSG series merged from `index_summary` (one request per day, has High/Low/Volume) and `daily_index` (one request per month, close only): close from `index_summary` where it exists, otherwise from `daily_index`; Volume only where `index_summary` has it (never invented).
- IDX IHSG was used only with **at least 260 rows** (trend needs 64 bars, SMA200 needs 200, the volatility ratio about 330); otherwise yfinance's longer series was kept with a note.

**Status**: superseded in effect by 1.4.3 (IHSG for trend now always yfinance). The merged series is still what the cross-check uses.

**Verified**: merge test (160 rows, every close equal to the ground truth, no duplicate dates, volume only for the recent 45 days); the 7-scenario matrix.

---

## 1.4.1: 2026-10-05

Repair of the IDX-API SQLite integration. The original problem was "DB path not found"; investigating it exposed further defects (each reproduced against a schema-faithful database before being fixed).

**Fixed: reproduced defects of 1.4.0**

| Defect | Symptom |
|---|---|
| `pd.to_datetime()` on epoch **milliseconds** | every row dated **1970** |
| Substring column matching | `id` read as the ex-dividend date (dividends = 0 rows); `eps` read as `eps_growth` (EPS Rp210 became "+210% growth") |
| x100 heuristic on ROE/ROA | ROA 1.2% became 120% (IDX already stores percent) |
| Per-stock foreign flow never computed | always `None` (it exists only in `stock_summary`, in **shares**) |
| One cached SQLite connection shared across threads | "SQLite objects created in a thread..." → empty results after the first rerun |
| IHSG series returned only `Close` | no High/Low/Volume, so the IHSG volume statistic could not work. The fallback table `daily_index` has no code column and would have mixed all indices (derived from the code, not reproduced) |
| PER looked up under key `"per"`, yfinance stores `"pe"` | PER always `n/a` (a v1.4.0 bug, not an IDX-API one) |
| TTM dividend yield summed announced-but-future ex-dates | e.g. 19.35% instead of 15.44% |

**Added**
- **Path diagnosis** with a precise reason: empty, missing (lists the `.sqlite` files in that folder), a folder, a **Git-LFS pointer** (`sample-database.sqlite` is 133 bytes unless `git lfs pull` was run; real size about 68 MB), not SQLite, empty file, cannot open. Quotes from Windows "Copy as path" are stripped.
- **Auto-discovery** of a database next to the script (script folder, working directory, `~/Documents`, shallow walk). URI-safe read-only open (`Path.as_uri`) with a plain-open fallback; paths with spaces and `#` work.
- Schema-aware adapter: exact table/column names; dates = epoch ms + 7 h (WIB) floored to the day; one connection per query; per-stock foreign flow = `foreign_net` shares × daily VWAP; market flow from `foreign_trading` (rupiah) with a fallback to the sum of all stocks (needs 300 or more stocks per day); IHSG from `index_summary` code `COMPOSITE`; YoY EPS growth on the same cumulative basis; zero Open/High/Low repaired.
- **Data Health** panel (table inventory with latest date and fresh/STALE status), **Reload IDX-API database** button, **ratio-priority** switch (default yfinance first: IDX ratios belong to the last report period, annualisation is undocumented), **stale-data limit** (default 10 days; IDX flows and IHSG older than the limit are ignored).
- Stale-price guard for the chart (**removed again in 1.4.3**).

**Verified**
- 25 adapter checks against a database built from the repo's Drizzle schemas and the sample-README row formats; later confirmed that the DDL of the six tables the app reads is **identical** to the real output of `deno task db:sync`.
- Whole app headless in 12+ scenarios, 0 exceptions.

**Not verified at release**: the real 68 MB sample database (LFS host blocked in the sandbox) and real IDX data. A later screenshot (v1.4.2) showed your real database opening: "IDX-API database OK · 39 tables · 91.9 MB".

---

## 1.4.0: baseline (supplied by you)

Contained the layered forecast engine and everything listed under "Before 1.4.0", plus the **first IDX-API adapter**: reads the SQLite file written by `NeaByteLab/IDX-API` (alias-based table/column discovery), sidebar source section, a Data Health expander, IDX values preferred over yfinance for foreign flow, ratios and dividends, and IDX prices replacing the chart price whenever 20 or more rows came back.

Defects found later are listed under 1.4.1. The placeholder in the path field was itself a real-looking path in grey, which made an empty field look filled.

---

## Before 1.4.0 (unversioned, reconstructed)

Order is reliable; dates and intermediate version numbers were not recorded. The very first part of the conversation was compacted before this log was written, so it is the least detailed.

**Analysis scripts and skill**
- `saham-forecast` skill with `portfolio_calc.py` (P&L, BEP, Fibonacci, GBM bands). Container resets made its persistence uncertain; I could not re-verify it exists.
- Real-data analysis of BBRI: ATH 6,450, crash low 2,540, bootstrap Monte Carlo under different return regimes (the result swings from 18% to 81% depending on the period, which is why returns are later de-meaned), macro "descending staircase" trendline (R² about 0.78 to 0.83), descending channel, weekly breakout and retest.
- `bbri_forecast_auto.py`: yfinance download with cache fallback, swing detection, trendline and channel fit, Fibonacci targets, bootstrap paths. Two bugs found and fixed while testing (empty download overwriting the cache; y-axis scaling).

**Streamlit app (your `app.py`, then iterated)**
1. Your baseline: resistance trendline from lower highs, Fibonacci levels, sine-wave scenario paths. Flagged: hard-coded fallback "R² = 0.83".
2. Added a regression **support trendline** and channel classification (parallel vs converging by slope ratio), shading, lower-low markers, 5 metric columns, 3-state status; fallback R² now shown as unavailable.
3. **Historical S/R zones** (polarity) and a **flag/pennant detector**, later generalised into a pattern scanner for the 20 patterns of your reference chart (continuation and reversal, anchored to the last Fibonacci cycle). English UI text.
4. Toggles for macro resistance/support; full date on hover (`hoverformat`); defaults changed (ticker, interval, rolling date range).
5. **Event calendar**: dividend history (yfinance), MSCI 2026 reviews, BI RDG 2026, RUPS (manual). A timezone bug in the dividend comparison was fixed and a manual dividend source added.
6. **Forecast engine**, in steps: a macro/fundamental scorecard (15 factors) tilting a two-way weight → Bullish/Sideways/Bearish probabilities (bootstrap of de-meaned returns plus an evidence tilt) → the **layered engine** (Market regime → Technical → Fundamental, with RSI/MACD/MA/Bollinger/ATR), "Sideways" renamed **Neutral**, scenario targets from the engine's conditional medians (Fibonacci targets optional).
7. Defaults saved to match your preferred screen (BBRI.JK, rolling 6 months, S/R zones, pattern detector, dividends, MSCI, BI RDG on).

**Charts only, not in the app**: synthesis forecast chart; an experimental moon-phase / Mercury-retrograde overlay, labelled unvalidated (no evidence that such cycles predict prices).

---

## Companion: `Cron.ts` / sync scripts (NeaByteLab/IDX-API)

| Step | Date (UTC) | What |
|---|---|---|
| Original | repo commit 910b8db, 2026-03-28 | Hard-coded `2026-01-01 → 2026-03-10`; every calendar day incl. weekends; hard-coded report years (`2025 audit`, `2026 TW1`); `syncCompanyProfile` downloads the **details of all ~960 companies one by one** (500 ms pause each); dozens of per-company syncs |
| `SyncForApp.ts` (separate lean script) | created 2026-10-05, `--tickers` pruning added 2026-10-06 | Only the six tables the app reads; automatic dates; weekdays only; 15 months of IHSG history (one request per month); Before/After table and **exit code 2** when no newer prices arrived (the repo's sync functions swallow network errors and only log "No data found", so "0 threw an error" proves nothing); `--tickers` pruning with a typo guard. Replaced by the modified Cron.ts. |
| Modified `Cron.ts` | 2026-10-06 | Same logic inside `Cron.ts`: `TICKERS` allow-list, `DAYS_BACK`, `RUN` switches (app tables / other market tables / per-company / company list), per-company syncs only for `TICKERS`, deferred planning when no tickers are set, dynamic report years, pruning of other stocks (disk space only: IDX serves those datasets market-wide per request), CLI overrides (`--days --from --to --tickers --delay --no-prune --no-per-company --all-market --profiles --dry-run`) |
| 35-stock list | 2026-10-06 | Your list (ADRO … UNVR) installed; **coverage report** lists requested codes that have no price rows |
| Marker | 2026-10-06 | First line `// idx-cron-compat: 2`, which app 1.4.4+ checks before auto-running the file |

**Request budget with 35 stocks** (from `--dry-run`)

| Run | With per-company data | `--no-per-company` |
|---|---|---|
| Daily (45 days) | 382 requests, about 9 min | 102 requests, about 3 min |
| One year (`--days=365`) | 852, about 20 min | 572, about 14 min |
| First fill (`--days=420`) | 932, about 22 min | 652, about 16 min |

Per-company data (financial reports, `trading_ss`, announcements) is **not read by the app**. Times assume about 1 s per request on top of the delay.

**Verified (Deno 2.9.6)**: `deno check` with the repo's strict compiler options; `deno task db:sync` on a fresh clone (39 tables); deleting `database.sqlite` and re-running `db:sync` recreates a clean schema; dry-runs; real executions with the network blocked; pruning in 5 cases checked against database contents (normal, typo guard, `.JK` suffix and lower case, invalid characters rejected, `ALL`); coverage report with 3 of 35 stocks present. **Not verified**: any real download from IDX.

---

## Compatibility

| | Requires |
|---|---|
| App 1.4.0–1.4.3 | nothing from Cron.ts |
| App 1.4.4+ auto-update | Deno on PATH, Cron.ts with `idx-cron-compat`, database at `<IDX-API folder>/data/database.sqlite` |
| Modified Cron.ts | works stand-alone with any app version |
| Database schema | NeaByteLab/IDX-API at commit 910b8db (39 tables after `db:sync`) |

## Known open items

1. **Engine is not "as-of End Date"** (see 1.4.3). A past End Date does not roll the engine back.
2. **IDX ratios**: whether ROE/ROA/PER for Mar/Jun/Sep periods are annualised is undocumented; hence yfinance first by default.
3. **`foreign_trading` plausibility**: a sample row in the repo shows foreign buy Rp5.07 T against sell Rp1.48 T in one day, which looks odd. Compare a day or two with IDX's published figures before trusting the market-level flow.
4. **Real downloads, Windows behaviour and the browser polling timer** are unverified (see 1.4.4).
5. Signal thresholds in the scorecard are heuristics, not calibrated to the stock's own history; probabilities are model-implied, not calibrated forecasts.

## Files

| File | Status |
|---|---|
| `stock_forecast_app_1_4_6.py` | **current** |
| `Cron.ts` (modified, marker `idx-cron-compat: 2`) | **current** |
| `CHANGELOG.md` | this file |
| `SyncForApp.ts`, app 1.4.1 to 1.4.5 | superseded and removed from the download folder; they exist only where you saved them earlier |
| 1.4.0 | your own upload |
