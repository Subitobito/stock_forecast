"""IDX Trend Studio: a small local server (standard library + yfinance + pandas, no Streamlit).

    python server.py                       # http://127.0.0.1:8000, opens the browser, auto-update ON (every 10 min)
    python server.py --interval 5          # change the cycle period (minutes)
    python server.py --no-sync             # never start the IDX-API sync (Cron.ts via Deno)
    python server.py --list                # list the stock_forecast_app_*.py versions that can be launched
    python server.py --app 1.4.7           # also launch that Streamlit version (or: latest | ask | off); default: STREAMLIT_APP below
    python server.py --db "D:\\IDX-API\\data\\database.sqlite" --no-browser --port 8010

AUTO-UPDATE: a background thread runs a SEQUENTIAL cycle every 10 minutes, for as long as this Python process runs:
    1. yfinance     download the watchlist prices
    2. IDX sync     when the IDX-API database is behind the last trading session, run Cron.ts in the background and
                    wait for it (at most one attempt per 30 min; needs Deno + the modified Cron.ts).
                    The Deno process is bound to this server: Ctrl+C / closing the console / killing Python stops it too.
    3. IDX info     read dividends, ratios, foreign flow and company names from the SQLite database
The page polls /api/status and re-downloads only what changed.

Endpoints
    GET  /                      the UI (static/index.html)
    GET  /api/bars?t=BBCA,JKSE&years=3[&refresh=1]
    GET  /api/idx?t=BBCA,BBRI   IDX-API information (dividends, ratios, foreign flow, names)
    GET  /api/deep?t=BBCA&weeks=16   Deep Forecast (Random Forest, XGBoost, SARIMAX; deep_forecast.py), trained in the background:
                                {"state": "running"} until it finishes, then {"state": "done", "result": {...}}
    GET  /api/status            pipeline state: steps, versions, next run
    POST /api/refresh           run a cycle now
Ticker mapping: JKSE -> ^JKSE, anything else -> <CODE>.JK.
"""
import argparse
import hashlib
import json
import re
import signal
import sys
import threading
import time
import webbrowser
from datetime import datetime, timedelta, timezone
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pandas as pd
import yfinance as yf

import idx_source

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))     # deep_forecast.py lives next to the Streamlit app, one folder up
try:
    import deep_forecast
    DEEP_ERR = ""
except Exception as _e:                                               # scikit-learn / xgboost / statsmodels missing
    deep_forecast, DEEP_ERR = None, f"{type(_e).__name__}: {_e}"

STATIC = Path(__file__).with_name("static")

# ----------------------------------------------------------------------------------------------------------------------
# WHICH stock_forecast_app_*.py VERSION TO LAUNCH WITH THE SERVER  <-- edit this line, or override it with --app
#     "latest"   the highest version found next to this folder (e.g. stock_forecast_app_1.5.0.py)
#     "1.4.7"    an exact version; a prefix such as "1.4" picks the highest 1.4.x
#     "ask"      show a numbered menu in the terminal every time the server starts
#     "off"      do not launch Streamlit (only the HTML UI)
# The Streamlit process lives and dies with this server, just like the IDX sync.
STREAMLIT_APP = "latest"
STREAMLIT_PORT = 8501
# ----------------------------------------------------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent.parent
APP_FILE_RE = re.compile(r"^stock_forecast_app_(\d+(?:\.\d+)*)\.py$")
APP_INFO = {}                      # filled in main(): {"version", "file", "url"}; reported by /api/status
APP_PROCS = []                     # Streamlit subprocesses started by this server
CACHE_TTL = 300
YEARS = 3
WIB = timezone(timedelta(hours=7))
WATCH = ["JKSE", "ADRO", "AKRA", "AMMN", "AMRT", "ANTM", "ASII", "BBCA", "BBNI", "BBRI", "BMRI", "BREN", "BRIS", "BRPT", "CPIN",
         "CTRA", "EXCL", "GOTO", "ICBP", "INDF", "INCO", "INKP", "INTP", "ITMG", "JPFA", "JSMR", "KLBF", "MDKA", "MEDC", "PGAS",
         "PTBA", "SMGR", "TLKM", "TPIA", "UNTR", "UNVR"]
_cache = {}                    # (ticker, years) -> (timestamp, list of bars)
_lock = threading.Lock()
_dl_lock = threading.Lock()    # one yfinance download at a time (page requests and the updater share the cache)
TICKER_RE = re.compile(r"^[A-Z0-9]{2,8}$")


# ------------------------------------------------------------------ yfinance
def yahoo_symbol(code):
    return "^JKSE" if code == "JKSE" else f"{code}.JK"


def frame_to_bars(df):
    """OHLCV DataFrame -> list of dicts the UI expects (dates as YYYY-MM-DD, NaN rows dropped)."""
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Open", "High", "Low", "Close"])
    return [{"t": pd.Timestamp(ts).strftime("%Y-%m-%d"), "o": float(r.Open), "h": float(r.High), "l": float(r.Low),
             "c": float(r.Close), "v": int(r.Volume) if pd.notna(r.Volume) else 0} for ts, r in df.iterrows()]


def download(codes, years):
    """One batched yfinance call for all codes. Returns {code: bars}; codes with no data are omitted."""
    symbols = [yahoo_symbol(c) for c in codes]
    df = yf.download(symbols, period=f"{years}y", interval="1d", auto_adjust=False, progress=False, group_by="ticker", threads=True)
    result = {}
    for code, sym in zip(codes, symbols):
        try:
            sub = df[sym] if isinstance(df.columns, pd.MultiIndex) else df
            bars = frame_to_bars(sub)
            if bars:
                result[code] = bars
        except KeyError:
            pass
    return result


def get_bars(codes, years, refresh):
    def stale_codes():
        now = time.time()
        with _lock:
            return [c for c in codes if refresh or (c, years) not in _cache or now - _cache[(c, years)][0] > CACHE_TTL]

    if stale_codes():
        with _dl_lock:
            # re-check: a concurrent request or the updater may have just filled the cache (a forced refresh always downloads)
            stale = stale_codes() if not refresh else list(codes)
            for i in range(0, len(stale), 50):
                chunk = stale[i:i + 50]
                fresh = download(chunk, years)
                now = time.time()
                with _lock:
                    for c, bars in fresh.items():
                        _cache[(c, years)] = (now, bars)
    with _lock:
        bars = {c: _cache[(c, years)][1] for c in codes if (c, years) in _cache}
    return bars, [c for c in codes if c not in bars]


# ------------------------------------------------------------------ sequential auto-update
def now_wib():
    return datetime.now(WIB)


def market_open(t=None):
    t = t or now_wib()
    return t.weekday() < 5 and (9, 0) <= (t.hour, t.minute) < (16, 10)


class Updater:
    STEPS = (("prices", "1 · yfinance"), ("sync", "2 · IDX sync"), ("idx", "3 · IDX info"))

    def __init__(self, db_path, interval_min=10, do_sync=True):
        self.db = db_path
        self.interval_min = interval_min
        self.do_sync = do_sync
        self.lock = threading.Lock()
        self.wake = threading.Event()
        self.tracked = list(WATCH)
        self.steps = {k: {"id": k, "label": lbl, "state": "idle", "detail": "waiting", "at": ""} for k, lbl in self.STEPS}
        self.versions = {"prices": 0, "idx": 0}
        self.sigs = {}
        self.idx = {"info": {}, "market": {}, "db": {}}
        self.running, self.cycle, self.next_run, self.trigger = False, 0, None, "startup"

    # --- state helpers
    def _set(self, step, state, detail):
        with self.lock:
            self.steps[step].update(state=state, detail=detail, at=now_wib().strftime("%H:%M:%S"))

    def _bump(self, kind, payload):
        sig = hashlib.md5(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()
        with self.lock:
            if self.sigs.get(kind) != sig:
                self.sigs[kind] = sig
                self.versions[kind] += 1
                return True
        return False

    def track(self, codes):
        with self.lock:
            for c in codes:
                if c not in self.tracked:
                    self.tracked.append(c)

    def status(self):
        with self.lock:
            return {"now": now_wib().strftime("%Y-%m-%d %H:%M:%S"), "market_open": market_open(), "running": self.running, "cycle": self.cycle,
                    "trigger": self.trigger, "steps": [dict(self.steps[k]) for k, _ in self.STEPS], "versions": dict(self.versions),
                    "next_run": datetime.fromtimestamp(self.next_run, WIB).strftime("%H:%M") if self.next_run else None,
                    "interval_min": self.interval_min, "db": self.idx.get("db", {}), "sync_enabled": self.do_sync}

    # --- steps
    def step_prices(self):
        self._set("prices", "run", "downloading from Yahoo Finance…")
        try:
            with self.lock:
                codes = list(self.tracked)
            bars, missing = get_bars(codes, YEARS, refresh=True)
            if not bars:
                raise RuntimeError("Yahoo returned no data (offline?)")
            last = max(b[-1]["t"] for b in bars.values())
            self._bump("prices", {c: (b[-1]["t"], b[-1]["c"], b[-1]["v"], len(b)) for c, b in bars.items()})
            self._set("prices", "ok", f"{len(bars)}/{len(codes)} tickers · last bar {last}" + (f" · no data: {', '.join(missing[:4])}" if missing else ""))
        except Exception as e:
            self._set("prices", "err", f"{type(e).__name__}: {e}"[:140])

    def step_sync(self):
        if not self.do_sync:
            return self._set("sync", "skip", "disabled (--no-sync)")
        if not self.db:
            return self._set("sync", "skip", "no IDX-API database found")
        sync = idx_source.AutoSync(self.db)
        ok, msg = sync.readiness()
        if not ok:
            return self._set("sync", "skip", msg[:140])
        if sync.status()["state"] != "running":
            start, reason, days = sync.should_auto_update()
            if not start:
                return self._set("sync", "ok", reason[:140])
            started, msg = sync.start(days, trigger="auto")
            if not started:
                return self._set("sync", "err", msg[:140])
            self._set("sync", "run", f"{reason} · running…")
        deadline = time.time() + 45 * 60
        while time.time() < deadline:
            s = sync.status()
            if s["state"] != "running":
                break
            self._set("sync", "run", f"Cron.ts {s.get('done', 0)}/{s.get('total') or '?'} requests · {s.get('last_line', '')[:60]}")
            time.sleep(5)
        s = sync.status()
        if s["state"] == "finished_ok":
            self._set("sync", "ok", f"prices now run to {s.get('prices_to')}")
        elif s["state"] == "finished_warn":
            self._set("sync", "warn", "finished but no newer prices (offline, blocked, or holiday)")
        elif s["state"] == "running":
            self._set("sync", "warn", "still running after 45 min; continuing in the background")
        else:
            self._set("sync", "err", f"{s['state']}: {s.get('last_line', '')[:100]}")

    def step_idx(self):
        if not self.db:
            return self._set("idx", "skip", "no IDX-API database found (use --db)")
        self._set("idx", "run", "reading the IDX-API database…")
        try:
            with self.lock:
                codes = [c for c in self.tracked if c != "JKSE"]
            snap = idx_source.snapshot(self.db, codes)
            with self.lock:
                self.idx = snap
            self._bump("idx", snap)
            n = sum(1 for v in snap["info"].values() if v.get("flow_bn") is not None or v.get("ratio") or v.get("dividends"))
            self._set("idx", "ok", f"{n} tickers with data · DB prices to {snap['db'].get('latest')}")
        except Exception as e:
            self._set("idx", "err", f"{type(e).__name__}: {e}"[:140])

    def idx_for(self, codes):
        """IDX info for `codes`; tickers not read yet are read now and tracked for the next cycles."""
        if not self.db:
            return {"info": {}, "market": {}, "db": {}}
        with self.lock:
            have = set(self.idx.get("info", {}))
        missing = [c for c in codes if c not in have and c != "JKSE"]
        if missing:
            snap = idx_source.snapshot(self.db, missing)
            self.track(missing)
            with self.lock:
                self.idx["info"].update(snap["info"])
                if not self.idx.get("market"):
                    self.idx["market"] = snap["market"]
                if not self.idx.get("db"):
                    self.idx["db"] = snap["db"]
        with self.lock:
            return {"info": {c: self.idx["info"][c] for c in codes if c in self.idx["info"]}, "market": self.idx.get("market", {}), "db": self.idx.get("db", {})}

    # --- loop
    def run_cycle(self, trigger):
        with self.lock:
            if self.running:
                return
            self.running, self.trigger, self.cycle = True, trigger, self.cycle + 1
        try:
            self.step_prices()      # 1 -> 2 -> 3, strictly in order
            self.step_sync()
            self.step_idx()
        finally:
            with self.lock:
                self.running = False

    def interval_seconds(self):
        return max(60, int(self.interval_min * 60))

    def loop(self):
        trigger = "startup"
        while True:
            self.run_cycle(trigger)
            wait = self.interval_seconds()
            self.next_run = time.time() + wait
            woke = self.wake.wait(wait)
            self.wake.clear()
            trigger = "manual" if woke else "timer"


class DeepService:
    """Trains the Deep Forecast models in a background thread (one at a time: it is CPU heavy) and caches the result.
    The first request for a ticker/settings starts the job and returns {"state": "running"}; the page polls until "done".
    A result is reused for 60 minutes, and retrained when a new daily bar arrives or on refresh=1."""

    TTL = 3600
    DISK_TTL = 12 * 3600              # a saved result is reused for 12 hours (and only while no new daily bar has arrived)
    CACHE_DIR = Path(__file__).resolve().parent / "cache"

    def __init__(self):
        self.jobs = {}
        self.lock = threading.Lock()
        self.sem = threading.Semaphore(1)

    def _disk_path(self, code, weeks, opts):
        """One JSON file per ticker/horizon/settings. The deep_forecast.py modification time is part of the name, so a code change never reuses old results."""
        try:
            ver = int(Path(deep_forecast.__file__).stat().st_mtime)
        except Exception:
            ver = 0
        tag = hashlib.md5(json.dumps([weeks, sorted(opts.items()), ver]).encode()).hexdigest()[:10]
        return self.CACHE_DIR / f"deep_{code}_{tag}.json"

    def _disk_load(self, path, sig):
        try:
            d = json.loads(path.read_text(encoding="utf-8"))
            if d.get("sig") == list(sig) and time.time() - d["saved"] < self.DISK_TTL and d.get("result"):
                return d
        except Exception:
            pass
        return None

    def _disk_save(self, path, sig, result):
        try:
            self.CACHE_DIR.mkdir(exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps({"sig": list(sig), "saved": time.time(), "result": result}), encoding="utf-8")
            tmp.replace(path)                                    # atomic: a half-written file is never read
            for old in sorted(self.CACHE_DIR.glob("deep_*.json"), key=lambda f: f.stat().st_mtime)[:-150]:
                old.unlink(missing_ok=True)                      # keep the folder small
        except Exception as e:
            print(f"  (deep cache not saved: {e})")

    def request(self, code, weeks, opts, refresh=False):
        if deep_forecast is None:
            return {"state": "error", "error": f"Deep Forecast unavailable ({DEEP_ERR}). Install: pip install scikit-learn xgboost statsmodels"}
        bars, _ = get_bars([code], YEARS, False)
        b = bars.get(code)
        if not b:
            return {"state": "error", "error": f"no price data for {code} (Yahoo returned nothing)"}
        sig = (b[-1]["t"], len(b))
        key = (code, weeks, tuple(sorted(opts.items())))
        with self.lock:
            job = self.jobs.get(key)
            if job and job["state"] == "running":
                return {"state": "running", "elapsed": round(time.time() - job["t0"])}
            fresh = job and time.time() - job["t"] < (self.TTL if job["state"] == "done" else 60)
            if job and fresh and not refresh and job["sig"] == sig:
                return {k: v for k, v in job.items() if k in ("state", "result", "error")} | {"age_s": round(time.time() - job["t"])}
            path = self._disk_path(code, weeks, opts)
            saved = None if refresh else self._disk_load(path, sig)
            if saved:                                            # trained earlier (maybe before a restart): no retraining
                job = {"state": "done", "result": saved["result"], "t": saved["saved"], "sig": sig}
                self.jobs[key] = job
                return {"state": "done", "result": job["result"], "age_s": round(time.time() - saved["saved"]), "from_disk": True}
            job = {"state": "running", "t0": time.time(), "t": time.time(), "sig": sig}
            self.jobs[key] = job
            for old in sorted(self.jobs, key=lambda k: self.jobs[k]["t"])[:-30]:      # keep the cache small
                if self.jobs[old]["state"] != "running":
                    del self.jobs[old]
        threading.Thread(target=self._run, args=(job, b, weeks, opts, path, sig), daemon=True, name=f"deep-{code}").start()
        return {"state": "running", "elapsed": 0}

    def _run(self, job, bars, weeks, o, path, sig):
        with self.sem:
            try:
                df = pd.DataFrame(bars)
                df.index = pd.to_datetime(df.pop("t"))
                df = df.rename(columns={"o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"})[["Open", "High", "Low", "Close", "Volume"]]
                res = deep_forecast.run(df, int(weeks) * 5, band_sigma=o["band"], n_estimators=o["trees"], n_splits=o["folds"],
                                        use_rf=bool(o["rf"]), use_xgb=bool(o["xgb"]), use_sarimax=bool(o["sx"]),
                                        sarimax_order=(o["p"], o["d"], o["q"]), sarimax_seasonal=(1, 0, 0, 5) if o["sea"] else (0, 0, 0, 0))
                job.update(state="done", result=idx_source.clean(res), t=time.time())
                self._disk_save(path, sig, job["result"])
            except Exception as e:
                job.update(state="error", error=f"{type(e).__name__}: {e}"[:300], t=time.time())


DEEP = DeepService()
UPDATER = None


# ------------------------------------------------------------------ stock_forecast_app_* versions
def find_app_versions():
    """[(version string, version tuple, Path)] of every stock_forecast_app_<version>.py in the project folder, oldest first."""
    found = []
    for f in PROJECT_DIR.glob("stock_forecast_app_*.py"):
        m = APP_FILE_RE.match(f.name)
        if m:
            found.append((m.group(1), tuple(int(x) for x in m.group(1).split(".")), f))
    return sorted(found, key=lambda t: t[1])


def choose_app(spec):
    """Resolve `spec` (latest | ask | off | exact version | version prefix) to (version, Path) or None for off. Exits with a clear message otherwise."""
    versions = find_app_versions()
    spec = (spec or "off").strip().lower()
    if spec in ("off", "none", "no", ""):
        return None
    if not versions:
        raise SystemExit(f"No stock_forecast_app_*.py found in {PROJECT_DIR}")
    if spec == "latest":
        return versions[-1][0], versions[-1][2]
    if spec == "ask":
        print("\nStock forecast versions:")
        for i, (v, _, f) in enumerate(versions, 1):
            print(f"  {i:2d}. v{v:<8s} {f.name}" + ("   (latest)" if i == len(versions) else ""))
        print("   0. none (HTML UI only)")
        while True:
            ans = input(f"Choose [1-{len(versions)}, Enter = latest, 0 = none]: ").strip()
            if ans == "":
                return versions[-1][0], versions[-1][2]
            if ans == "0":
                return None
            if ans.isdigit() and 1 <= int(ans) <= len(versions):
                return versions[int(ans) - 1][0], versions[int(ans) - 1][2]
            print("  not a valid choice")
    spec = spec.lstrip("v")
    exact = [t for t in versions if t[0] == spec]
    if exact:
        return exact[0][0], exact[0][2]
    prefix = [t for t in versions if t[0].startswith(spec + ".")]
    if prefix:
        return prefix[-1][0], prefix[-1][2]
    raise SystemExit(f"Version '{spec}' not found. Available: {', '.join(v for v, _, _ in versions)}")


def free_port(start):
    import socket
    for port in range(start, start + 30):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sk:
            if sk.connect_ex(("127.0.0.1", port)) != 0:          # nothing listens there
                return port
    return start


def start_streamlit(version, path, port):
    """Run `streamlit run <path>` from the project folder (so .streamlit/config.toml and deep_forecast.py are found).
    It is bound to this process: stopped on exit, and killed by Windows if this process is killed."""
    import atexit
    import subprocess
    port = free_port(port)
    cmd = [sys.executable, "-m", "streamlit", "run", str(path), "--server.port", str(port), "--server.headless", "true",
           "--browser.gatherUsageStats", "false"]
    log = open(Path(__file__).with_name("streamlit_app.log"), "w", encoding="utf-8")
    kw = {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0)} if sys.platform == "win32" else {}
    proc = subprocess.Popen(cmd, cwd=str(PROJECT_DIR), stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, **kw)
    idx_source._bind_to_this_process(proc)
    APP_PROCS.append(proc)
    atexit.register(stop_streamlit)
    APP_INFO.update(version=version, file=path.name, url=f"http://localhost:{port}/")
    return proc


def open_when_ready(url, timeout=180):
    """Open the Streamlit page only once Streamlit answers. At system start-up it can need much longer than a few seconds
    (the machine is busy), and a tab opened too early just shows 'connection refused'."""
    import urllib.request
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(url.rstrip("/") + "/_stcore/health", timeout=3) as r:
                if r.status == 200:
                    webbrowser.open(url)
                    return
        except Exception:
            pass
        time.sleep(1.5)
    print(f"Streamlit did not answer within {timeout}s; see streamlit_app.log. Open {url} manually.")


def stop_streamlit():
    import subprocess
    for proc in APP_PROCS:
        try:
            if proc.poll() is None:
                if sys.platform == "win32":
                    subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15)
                else:
                    proc.terminate()
        except Exception:
            pass
    APP_PROCS.clear()


# ------------------------------------------------------------------ HTTP
class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(STATIC), **kw)

    def _json(self, status, payload):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    @staticmethod
    def _codes(q, limit=60):
        codes = [c.strip().upper().lstrip("^") for c in q.get("t", [""])[0].split(",") if c.strip()]
        codes = list(dict.fromkeys(codes))[:limit]
        return codes if codes and all(TICKER_RE.match(c) for c in codes) else None

    def do_POST(self):
        if urlparse(self.path).path == "/api/refresh":
            if UPDATER:
                UPDATER.wake.set()
            return self._json(200, {"ok": True})
        self._json(404, {"error": "not found"})

    def do_GET(self):
        url = urlparse(self.path)
        if not url.path.startswith("/api/"):
            if url.path == "/":
                self.path = "/index.html"
            return super().do_GET()
        q = parse_qs(url.query)
        try:
            if url.path == "/api/status":
                payload = UPDATER.status() if UPDATER else {"error": "updater disabled", "steps": [], "versions": {}}
                return self._json(200, {**payload, "app": APP_INFO or None})
            if url.path == "/api/refresh":
                if UPDATER:
                    UPDATER.wake.set()
                return self._json(200, {"ok": True})
            codes = self._codes(q)
            if not codes:
                return self._json(400, {"error": "t must be a comma-separated list of ticker codes, e.g. BBCA,BBRI,JKSE"})
            if url.path == "/api/bars":
                years = max(1, min(10, int(q.get("years", [str(YEARS)])[0])))
                if UPDATER and years == YEARS:
                    UPDATER.track(codes)
                bars, missing = get_bars(codes, years, q.get("refresh", ["0"])[0] == "1")
                return self._json(200, {"bars": bars, "missing": missing, "fetched": time.strftime("%Y-%m-%d %H:%M:%S")})
            if url.path == "/api/deep":
                def num(name, default, lo, hi, cast=int):
                    return max(lo, min(hi, cast(q.get(name, [default])[0])))
                flag = lambda name, default="1": q.get(name, [default])[0] == "1"
                opts = {"band": round(num("band", 0.5, 0.2, 1.0, float), 2), "trees": num("trees", 200, 50, 600), "folds": num("folds", 5, 3, 8),
                        "rf": flag("rf"), "xgb": flag("xgb"), "sx": flag("sx"), "p": num("p", 1, 0, 4), "d": num("d", 1, 0, 2),
                        "q": num("q", 1, 0, 4), "sea": flag("sea", "0")}
                return self._json(200, DEEP.request(codes[0], num("weeks", 16, 1, 26), opts, q.get("refresh", ["0"])[0] == "1"))
            if url.path == "/api/idx":
                return self._json(200, UPDATER.idx_for(codes) if UPDATER else {"info": {}, "market": {}, "db": {}})
            self._json(404, {"error": "unknown endpoint"})
        except Exception as e:                      # yfinance/network/SQLite errors -> a readable message for the UI
            self._json(502, {"error": f"{type(e).__name__}: {e}"})

    def log_message(self, fmt, *args):
        if "/api/" in (args[0] if args else "") and "/api/status" not in args[0]:
            super().log_message(fmt, *args)


def main():
    global UPDATER
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--db", help="path to IDX-API data/database.sqlite (default: auto-discover)")
    ap.add_argument("--interval", type=float, default=10, help="minutes between update cycles (default 10)")
    ap.add_argument("--no-sync", action="store_true", help="never start the IDX-API sync (Cron.ts)")
    ap.add_argument("--no-updater", action="store_true", help="disable the background auto-update entirely")
    ap.add_argument("--app", default=None, metavar="VERSION",
                    help=f"stock_forecast_app version to launch with Streamlit: latest | ask | off | 1.5.0 | 1.4 (default: {STREAMLIT_APP!r}, set STREAMLIT_APP in server.py)")
    ap.add_argument("--app-port", type=int, default=STREAMLIT_PORT, help="first port to try for Streamlit (default 8501)")
    ap.add_argument("--list", action="store_true", help="list the available stock_forecast_app_*.py versions and exit")
    a = ap.parse_args()
    if a.list:
        vs = find_app_versions()
        print(f"stock_forecast_app versions in {PROJECT_DIR}:")
        for i, (v, _, f) in enumerate(vs, 1):
            print(f"  v{v:<8s} {f.name}" + ("   <- latest" if i == len(vs) else ""))
        print(f"\nDefault (STREAMLIT_APP in server.py): {STREAMLIT_APP!r}.  Override: python server.py --app 1.4.7")
        return
    chosen = choose_app(a.app if a.app is not None else STREAMLIT_APP)
    db = idx_source.find_db(a.db)
    print("IDX-API database:", db or "not found (IDX information disabled; use --db)")
    if not a.no_updater:
        UPDATER = Updater(db, a.interval, not a.no_sync)
        threading.Thread(target=UPDATER.loop, daemon=True, name="updater").start()
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    url = f"http://{a.host}:{a.port}/"
    print(f"IDX Trend Studio running at {url}  (Ctrl+C to stop)")
    if chosen:
        start_streamlit(chosen[0], chosen[1], a.app_port)
        print(f"Streamlit stock forecast v{chosen[0]} ({chosen[1].name}) starting at {APP_INFO['url']}   (log: streamlit_app.log)")
        if not a.no_browser:
            threading.Thread(target=open_when_ready, args=(APP_INFO["url"],), daemon=True, name="open-streamlit").start()
    else:
        print("Streamlit not launched (STREAMLIT_APP = off).")
    if not a.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    def _terminate(signum, frame):      # SIGTERM / Ctrl+Break: leave serve_forever() so the finally block runs
        raise SystemExit(0)

    for name in ("SIGTERM", "SIGBREAK"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), _terminate)
    try:
        srv.serve_forever()
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        print("\nStopping... (also stopping the IDX sync if it is running)")
        idx_source.stop_all()
        stop_streamlit()
        srv.server_close()
        print("Stopped.")


if __name__ == "__main__":
    main()
