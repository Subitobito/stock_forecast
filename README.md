# IDX Trend Studio

A local analysis and forecasting tool for Indonesian (IDX) stocks. It has two interfaces:

- **Web UI (HTML)**: served by the small Python server `server.py` at `http://localhost:8000`. No Streamlit needed.
- **Streamlit** (`../stock_forecast_app_1.x.py`): the full earlier app, which `server.py` can launch alongside.

Prices come from Yahoo Finance (yfinance). Extra information (dividends, ratios, foreign flow, company names) is read from the IDX-API database (SQLite).

> All numbers are model probabilities, not calibrated predictions and not investment advice.

## Folder layout

```
stock_forecast/
├── requirements.txt            Python libraries
├── deep_forecast.py            shared logic: Random Forest, XGBoost, SARIMAX, forecast line, candles
├── stock_forecast_app_1.5.0.py Streamlit (latest; older versions are kept)
└── idx_trend_studio/
    ├── server.py               web server + auto-update + Streamlit launcher
    ├── idx_source.py           IDX-API database reader + sync (Cron.ts via Deno)
    ├── static/index.html       the whole Web UI
    ├── cache/                  saved Deep Forecast results (created automatically, safe to delete)
    ├── idx_trend_studio.bat    start at login on Windows
    └── mac/                    start at login on macOS (see below)
```

## Installation

Requires Python 3.10+.

```bash
cd stock_forecast
python -m venv .venv
# Windows: .venv\Scripts\activate      Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Optional, for the IDX sync: [Deno](https://deno.com) and the IDX-API folder with `Cron.ts`. Without them the IDX information is disabled and everything else still works.
On a Mac, XGBoost needs OpenMP: `brew install libomp`.

## Running

```bash
cd idx_trend_studio
python server.py
```

Open **http://localhost:8000** (the browser opens automatically unless you pass `--no-browser`).

| Option | What it does |
|---|---|
| `--port 8010` | another port (default **8000**) |
| `--no-browser` | do not open a browser tab |
| `--app off` / `latest` / `1.5.0` | also launch Streamlit; without it, `STREAMLIT_APP` in `server.py` is used (currently `latest`, so Streamlit starts too) |
| `--list` | list the available `stock_forecast_app_*.py` versions |
| `--app-port 8501` | first port to try for Streamlit |
| `--interval 5` | auto-update period in minutes (default 10) |
| `--no-sync` | never start the IDX sync (Cron.ts) |
| `--no-updater` | turn the background auto-update off completely |
| `--db PATH` | location of `IDX-API/data/database.sqlite` (default: auto-discovered) |
| `--host 0.0.0.0` | reach it from other devices on the network (default: local only) |

Stopping the server (Ctrl+C, or closing the console) also stops the IDX sync and the Streamlit process it started.

### Auto-update (while the server runs)

Every 10 minutes, one step after the other:
1. **yfinance**: download the watchlist prices.
2. **IDX sync**: when the database is behind the last trading session, run `Cron.ts` in the background (at most once every 30 minutes).
3. **IDX info**: read dividends, ratios, foreign flow and company names.

The page watches `/api/status` and only downloads what changed. The "Update now" button runs one cycle immediately.

## Forecasts on the chart

Two separate forecasts, each with its own switch in the Forecast panel:

**Forecast line** (dashed): the projection of the detected pattern (Bullish Flag and so on).
- It moves to the Fibonacci / measured-move target in 5 Fibonacci waves (wave 1 = 40% of the move, pullbacks of 50% and 38.2%, wave 3 = 1.618 × wave 1), then an A-B-C correction to the 38.2% retracement, then flat.
- It uses the **last 250 daily bars**, so it does not change when you change the chart period.
- Without a detected pattern it follows the forecast engine's tilt.
- Event markers show when the line crosses the cycle trendlines (Breakout / Breakdown), the pattern lines or a Fibonacci level, and where it peaks and troughs.

**Include the ML models** (candles): forecast candles from Random Forest + XGBoost + SARIMAX, weighted by their walk-forward skill.
- The candles follow the ML trend with realistic ups and downs (volatility and wicks drawn from the last 250 real candles), seeded with the last bar's date so they stay stable when the horizon changes.
- The candle shape is one plausible path, **not** a day-by-day forecast. Read direction and timing from the trend and the event markers.
- Models are trained on the server (15–30 s per ticker). Results are saved to `cache/` for 12 hours and reused after a restart as long as no new daily bar has arrived. "Retrain now" forces a new run.

**Blend ML into scenario probabilities**: feeds the ML into the Bull / Neutral / Bear probabilities and the conviction score, limited by the *ML blend cap*.

The all-methods consensus (engine, pattern, ML) is shown as a reference in the tooltip and the Deep Forecast tab; it is not drawn on the chart.

### Key settings

| Setting | Effect |
|---|---|
| Forecast weeks ahead | length of the projection; it does not change the shape of the line, only where it is cut |
| Tilt strength (%) | how strongly the evidence score shifts the forecast engine's direction |
| Neutral band (×σ) | width of the Neutral zone in the probabilities |
| ML blend cap (%) | maximum influence of the ML on the probabilities (0 = none) |
| Deep Forecast tab | number of trees, walk-forward folds, SARIMAX p/d/q order, weekly seasonality |

## Start at login

### Windows

`idx_trend_studio.bat` runs `python server.py --app off` (Web UI only). Put a shortcut, or a `.vbs` script that calls it, in the Startup folder:
`%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup`.
To launch Streamlit as well, change it to `--app latest`.

### macOS (launchd)

```bash
cd stock_forecast/idx_trend_studio/mac
chmod +x *.sh
./install_autostart.sh                  # Web UI only
./install_autostart.sh --app latest     # plus Streamlit
./install_autostart.sh --browser        # open the browser at every login
./install_autostart.sh --db /path/to/IDX-API/data/database.sqlite
```

- The installer creates `.venv` if it is missing, writes `~/Library/LaunchAgents/com.idxtrendstudio.plist`, loads it, and starts the server right away. The server starts again at every login.
- UI: **http://localhost:8000**. Log: `idx_trend_studio/server.log`.
- The server is restarted if it crashes, but not when you stop it yourself.
- `./uninstall_autostart.sh` removes it. `./diagnose.sh` collects diagnostic information when the auto start does not work (send its whole output).

Useful commands:

```bash
launchctl print gui/$(id -u)/com.idxtrendstudio | head -n 20     # status
launchctl kickstart -k gui/$(id -u)/com.idxtrendstudio           # restart
tail -f idx_trend_studio/server.log                              # log
```

Differences from Windows: child processes (the IDX sync, Streamlit) only stop automatically when the server is stopped properly; `kill -9` can leave an orphaned Deno process. The 10-minute update pauses while the Mac sleeps.

## Local API

| Endpoint | Purpose |
|---|---|
| `GET /` | the Web UI |
| `GET /api/bars?t=BBCA,JKSE&years=3[&refresh=1]` | OHLCV data (`JKSE` → `^JKSE`, anything else → `<CODE>.JK`) |
| `GET /api/idx?t=BBCA,BBRI` | IDX-API information |
| `GET /api/deep?t=BBCA&weeks=16` | Deep Forecast; `{"state":"running"}` until it finishes, then `{"state":"done","result":...}` |
| `GET /api/status` | pipeline state, data versions, next run |
| `POST /api/refresh` | run one cycle now |

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| The page does not open | check the port: the default is **8000** (not 8765). Look at `server.log` |
| "ML models: training…" takes long | normal for the first 15–30 s; a ticker that was trained before loads straight from the cache |
| Deep Forecast unavailable | `pip install scikit-learn xgboost statsmodels`; on a Mac `brew install libomp` |
| IDX information is empty | database not found: use `--db`; the sync needs Deno |
| Streamlit tab did not open at startup | the server waits up to 3 minutes for Streamlit to be ready; see `streamlit_app.log` |
| Odd UI settings after an update | they are stored in the browser; clear the site data for `localhost:8000` |

Release notes are in `../Changelog.md`.
