import streamlit as st
import os
import sqlite3
from pathlib import Path
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.signal import find_peaks
from scipy.stats import linregress

try:   # v1.5.0: needs scikit-learn, xgboost, statsmodels (pip install -r requirements.txt); the rest of the app works without them
    import deep_forecast as dfc
    DEEP_IMPORT_ERR = ""
except Exception as _deep_err:
    dfc, DEEP_IMPORT_ERR = None, f"{type(_deep_err).__name__}: {_deep_err}"

# Web page configuration
st.set_page_config(page_title="Macro Pattern & Fibonacci Analyzer v1.5.0", layout="wide")
APP_VERSION = "1.5.0"
# v1.5.0: Deep Forecast (ML): Random Forest + XGBoost (forward-return regressors and Bullish/Neutral/Bearish classifiers, scikit-learn pipelines
#         with imputer + scaler) and a SARIMAX path on the log price (statsmodels). Everything lives in deep_forecast.py next to this file.
#         Validation is walk-forward (TimeSeriesSplit, gap = horizon) against a naive baseline; ranges come from out-of-fold residuals.
#         New requirements: scikit-learn, xgboost, statsmodels (see requirements.txt).
# v1.4.7: macOS-style look (SF font stack, rounded controls, translucent sidebar, system-blue accent) always Dark (fixed in code and in .streamlit/config.toml; no Light/Dark picker).
# v1.4.6: company logo (SVG files in <IDX-API>/sample/img, found next to your database; shown without a frame, 58 px) and company name in the header; STOCKLIST.md only
#         lists the logos, so its 30-character-cut names are completed from company_profile when that table is filled.
# v1.4.5: the IDX-API source switches itself on at start-up when the live database is found, so the auto-update below runs on the very
#         first launch (before, the checkbox started unticked in every new browser session and nothing happened until you ticked it).
# v1.4.4: auto-update of the IDX-API database: runs Cron.ts in the background on startup when the database is behind the last trading
#         session (needs Deno + the modified Cron.ts; refuses the original Cron.ts and a database path a sync would not update).
# v1.4.3: (1) End Date is inclusive (yfinance's `end` is exclusive, so End Date = 6 Oct used to stop at 5 Oct);
#         (2) IDX-API is ADDITIONAL information only (foreign flow, dividends, ratios): prices, trends and the JCI always come from yfinance;
#         (3) yfinance-vs-IDX-API cross-check in the data-health panel; (4) 5-minute cache + "Refresh market data" button;
#         (5) note when the last bar is still forming intraday.
# v1.4.2: JCI = index_summary (volume) merged with daily_index (cheap monthly history); IDX JCI is used only with >= 260 rows,
#         otherwise yfinance's longer series is kept so the trend/volatility statistics are not lost.
# v1.4.1 changelog (IDX-API SQLite integration fixes)
#  - Path handling: quotes stripped (Windows "Copy as path"), precise diagnosis of an empty / missing / folder / Git-LFS-pointer / non-SQLite
#    path, auto-discovery next to this script, URI-safe read-only open (Path.as_uri) with a plain-open fallback.
#  - Dates: IDX-API stores epoch MILLISECONDS; the old pd.to_datetime() read nanoseconds and dated every row 1970 (+7h WIB shift added).
#  - Column matching: exact schema names only. The old substring matching picked 'id' as the ex-dividend date and 'eps' as EPS growth.
#  - Foreign flow: per-stock flow now comes from stock_summary (foreign_net is in SHARES, converted at the daily VWAP);
#    market flow from foreign_trading (rupiah) with a fallback to the sum of all stocks.
#  - JCI: index_summary code COMPOSITE (OHLCV incl. volume), not a mix of all indices.
#  - Ratios: IDX stores ROE/ROA in percent already (the old x100 heuristic turned ROA 1.2% into 120%); latest period chosen by date.
#  - Thread safety: a connection per query (a cached sqlite connection raised "created in a thread..." on every rerun).
#  - Freshness guard: stale IDX-API prices/JCI/flows are ignored instead of silently anchoring the forecast to an old price.
#  - TTM dividend yield ignores announced-but-not-yet-ex dividends; PER from yfinance was never used (key "per" vs "pe").

# ============================================================
# IDX-API · DATABASE LOCATION & HEALTH HELPERS (v1.4.6)
# Pure functions with no Streamlit calls, so they can run before the sidebar renders.
# ============================================================
import re
from contextlib import closing

IDX_SQLITE_MAGIC = b"SQLite format 3\x00"
IDX_LFS_MAGIC = b"version https://git-lfs"
# Where the database can live, relative to a search root. The LIVE database written by
# `deno task db:sync` + Cron.ts is data/database.sqlite (the data/ folder is git-ignored, so a
# fresh `git clone` does not contain it). sample/sample-database.sqlite is a Git-LFS snapshot.
IDX_REL_CANDIDATES = (
    "data/database.sqlite", "IDX-API/data/database.sqlite", "Documents/IDX-API/data/database.sqlite",
    "database.sqlite",
    "sample/sample-database.sqlite", "IDX-API/sample/sample-database.sqlite",
    "Documents/IDX-API/sample/sample-database.sqlite",
)


def idx_clean_path(p):
    """Windows 'Copy as path' wraps the path in double quotes — strip quotes and whitespace."""
    return str(p or "").strip().strip('"').strip("'").strip()


def idx_connect(path):
    """Open the SQLite file read-only. Path.as_uri() percent-encodes spaces, '#', '%' and drive
    letters correctly (a hand-built 'file:C:\\...' URI is fragile on Windows); fall back to a
    plain open if the URI form is rejected. A NEW connection is made per call because a sqlite3
    connection must not be shared between threads and Streamlit runs every rerun in a new thread."""
    p = Path(idx_clean_path(path)).expanduser().resolve()
    try:
        return sqlite3.connect(f"{p.as_uri()}?mode=ro", uri=True, timeout=5)
    except sqlite3.Error:
        return sqlite3.connect(str(p), timeout=5)


def _idx_file_kind(path):
    """Sniff the first bytes: ('sqlite'|'lfs_pointer'|'other'|'unreadable', detail)."""
    try:
        with open(path, "rb") as fh:
            head = fh.read(256)
    except OSError as e:
        return "unreadable", str(e)
    if head.startswith(IDX_SQLITE_MAGIC):
        return "sqlite", ""
    if head.startswith(IDX_LFS_MAGIC):
        m = re.search(rb"size (\d+)", head)
        return "lfs_pointer", (f"{int(m.group(1)) / 1e6:.0f} MB" if m else "")
    return "other", ""


def idx_discover_db():
    """Look for IDX-API databases near this script, the working directory and ~/Documents.
    Returns [{path, kind, size_mb, live}], valid SQLite files first and the live DB before the sample."""
    roots = []
    try:
        roots.append(Path(__file__).resolve().parent)
    except NameError:
        pass
    roots += [Path.cwd(), Path.home() / "Documents", Path.home()]
    seen, found = set(), []

    def add(c):
        try:
            c = c.resolve()
            if c in seen or not c.is_file():
                return
            seen.add(c)
            kind, _ = _idx_file_kind(c)
            found.append({"path": str(c), "kind": kind, "size_mb": c.stat().st_size / 1e6, "live": c.name == "database.sqlite"})
        except OSError:
            pass

    for base in roots:
        for rel in IDX_REL_CANDIDATES:
            add(base / rel)
    if roots:  # shallow walk of the script folder, wherever IDX-API was cloned inside the project
        top = roots[0]
        for dirpath, dirs, files in os.walk(top):
            dirs[:] = [d for d in dirs if d not in {".git", "node_modules", ".deno", "__pycache__", ".venv", "venv"}]
            if len(Path(dirpath).relative_to(top).parts) > 4:
                dirs[:] = []
                continue
            for fn in files:
                if fn.lower().endswith((".sqlite", ".sqlite3", ".db")):
                    add(Path(dirpath) / fn)
    found.sort(key=lambda d: (d["kind"] != "sqlite", not d["live"]))
    return found


@st.cache_data(ttl=60, show_spinner=False)
def idx_discover_db_cached():
    return idx_discover_db()


def idx_diagnose_db(path):
    """Say precisely WHY a path cannot be used. Returns {ok, code, level, message, hint, path, size_mb, n_tables}."""
    raw = idx_clean_path(path)
    out = {"ok": False, "code": "empty", "level": "warning", "path": raw, "message": "", "hint": "", "size_mb": None, "n_tables": 0}
    live_hint = ("The live database is <IDX-API folder>/data/database.sqlite. It does not exist after a plain `git clone` "
                 "(data/ is git-ignored): install Deno, then run `deno task db:sync` and `deno run -A Cron.ts` inside the IDX-API folder.")
    if not raw:
        out["message"] = "No database path entered yet."
        out["hint"] = "Paste the full path of the .sqlite FILE. " + live_hint
        return out
    p = Path(raw).expanduser()
    if not p.exists():
        parent = p.parent
        siblings = sorted(x.name for x in parent.glob("*.sqlite"))[:5] if parent.is_dir() else []
        out.update(code="missing", message=f"File not found: {p}")
        if siblings:
            out["hint"] = f"That folder exists and contains: {', '.join(siblings)}. Check the file name. " + live_hint
        elif parent.is_dir():
            out["hint"] = "That folder exists but has no .sqlite file. " + live_hint
        else:
            out["hint"] = "That folder does not exist — check the drive/folder spelling. " + live_hint
        return out
    if p.is_dir():
        inside = []
        for rel in ("data/database.sqlite", "database.sqlite", "sample/sample-database.sqlite"):
            if (p / rel).is_file():
                inside.append(str(p / rel))
        inside += [str(x) for x in sorted(p.rglob("*.sqlite"))[:3] if str(x) not in inside]
        out.update(code="directory", message=f"This is a folder, not a database file: {p}")
        out["hint"] = ("Use the file itself, e.g. " + inside[0]) if inside else ("No .sqlite file inside it. " + live_hint)
        return out
    size = p.stat().st_size
    out["size_mb"] = size / 1e6
    if size == 0:
        out.update(code="no_tables", message="The file is empty (0 bytes): a database that was created but never filled.")
        out["hint"] = "Run `deno task db:sync` (creates the tables) and then Cron.ts (fills them), or point to a database that has data."
        return out
    kind, detail = _idx_file_kind(p)
    if kind == "lfs_pointer":
        out.update(code="lfs_pointer", level="error",
                   message=f"This file is only a Git LFS pointer ({size} bytes), not the real database"
                           + (f" (the real file is about {detail})." if detail else "."))
        out["hint"] = ("sample-database.sqlite is stored with Git LFS and a clone made without git-lfs keeps just this pointer. "
                       "Run `git lfs install` then `git lfs pull` inside the IDX-API folder, or download the file from the GitHub page. "
                       "Note the sample is an old snapshot — for current data build your own: " + live_hint)
        return out
    if kind == "unreadable":
        out.update(code="unreadable", level="error", message=f"Cannot read the file: {detail}")
        return out
    if kind != "sqlite":
        out.update(code="not_sqlite", level="error", message=f"Not a SQLite database ({size:,} bytes; the header is wrong).")
        out["hint"] = "Point to the .sqlite file written by IDX-API (data/database.sqlite)."
        return out
    try:
        with closing(idx_connect(raw)) as c:
            n = c.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'").fetchone()[0]
    except Exception as e:
        out.update(code="open_failed", level="error", message=f"SQLite could not open the file: {type(e).__name__}: {e}")
        out["hint"] = "If Deno/Cron.ts is writing to it right now, wait for it to finish; otherwise copy the file elsewhere and point to the copy."
        return out
    out["n_tables"] = n
    if n == 0:
        out.update(code="no_tables", message="The file is a valid SQLite database but has no tables.")
        out["hint"] = "It was created empty. Run `deno task db:sync` (creates the tables) and then Cron.ts (fills them)."
        return out
    out.update(ok=True, code="ok", level="ok", message=f"IDX-API database OK · {n} tables · {size / 1e6:.1f} MB")
    return out




# ============================================================
# IDX-API · AUTO-UPDATE (v1.4.6)
# Starts `deno run -A Cron.ts` in the BACKGROUND when the app opens and the database is behind. Pure Python (no Streamlit
# calls) so it can be tested; the UI lives in the sidebar. The app never waits for the sync: it analyses with the data that
# exists now and reruns itself when the sync has finished.
# ============================================================
import json
import shutil
import subprocess

IDX_CRON_MARKER = "idx-cron-compat:"


def idx_expected_last_session(now_wib=None, close_hour=17):
    """The last trading date the database should already hold: today once the session has closed (IDX publishes
    after ~16:00 WIB, we wait until 17:00), otherwise the previous weekday. Exchange holidays are not known, so on a
    holiday this date is never reached (the minimum-hours throttle then prevents repeated attempts)."""
    now = now_wib if now_wib is not None else pd.Timestamp.now(tz="Asia/Jakarta")
    d = pd.Timestamp(now.year, now.month, now.day)
    if not (now.weekday() < 5 and now.hour >= close_hour):
        d -= pd.Timedelta(days=1)
    while d.weekday() >= 5:
        d -= pd.Timedelta(days=1)
    return d


def idx_db_latest_stock_date(db_path):
    """Newest date in stock_summary (epoch ms + 7h = WIB calendar day), or None for no file / no table / no rows."""
    try:
        with closing(idx_connect(db_path)) as c:
            mx = c.execute('SELECT MAX("date") FROM "stock_summary"').fetchone()[0]
        return None if mx is None else (pd.Timestamp(int(mx), unit="ms") + pd.Timedelta(hours=7)).normalize()
    except Exception:
        return None


class IDXAutoSync:
    """Runs the IDX-API sync (Cron.ts) as a background process and reports its progress from the log file."""

    def __init__(self, db_path, registry=None):
        self.db = Path(idx_clean_path(db_path)).expanduser()
        self.data_dir = self.db.parent
        self.folder = self.data_dir.parent
        self.cron = self.folder / "Cron.ts"
        self.state_file = self.data_dir / "autosync_state.json"
        self.log_file = self.data_dir / "autosync.log"
        self.registry = registry if registry is not None else {}   # keeps the Popen handle alive across Streamlit reruns

    # ---------- prerequisites ----------
    def deno(self):
        found = shutil.which("deno")
        if found:
            return found
        for c in (Path.home() / ".deno" / "bin" / "deno.exe", Path.home() / ".deno" / "bin" / "deno"):
            if c.is_file():
                return str(c)
        return None

    def readiness(self):
        """(ok, message). The auto-update is allowed only when every prerequisite holds."""
        if self.db.name != "database.sqlite" or self.data_dir.name != "data":
            return False, (f"Auto-update writes to <IDX-API folder>/data/database.sqlite, but the app reads {self.db}. "
                           "Point the path to that file (sample/sample-database.sqlite is never updated by a sync).")
        if not self.cron.is_file():
            return False, f"Cron.ts was not found in {self.folder}."
        try:
            head = self.cron.read_text(encoding="utf-8", errors="ignore")[:4000]
        except OSError as e:
            return False, f"Cannot read Cron.ts: {e}"
        if IDX_CRON_MARKER not in head:
            return False, ("This Cron.ts is the original one: it ignores --days/--tickers and would download all ~960 companies over a hard-coded "
                           "date range. Replace it with the modified Cron.ts before enabling the auto-update.")
        if not self.deno():
            return False, "Deno was not found. Install it from deno.com (or add it to PATH), then restart the app."
        return True, "ready"

    # ---------- state ----------
    def _read_state(self):
        try:
            return json.loads(self.state_file.read_text(encoding="utf-8"))
        except Exception:
            return None

    def _write_state(self, **kw):
        try:
            self.state_file.write_text(json.dumps(kw, ensure_ascii=False), encoding="utf-8")
        except OSError:
            pass

    def log_tail(self, n=25):
        try:
            return "\n".join(self.log_file.read_text(encoding="utf-8", errors="replace").splitlines()[-n:])
        except OSError:
            return ""

    def status(self):
        """{state: idle|running|finished_ok|finished_warn|failed|stalled, done, total, last_line, ...}.
        Progress is read from the log (the plan line says how many requests; every request logs '[Sync] Starting')."""
        st_ = self._read_state()
        if not st_:
            return {"state": "idle"}
        try:
            log = self.log_file.read_text(encoding="utf-8", errors="replace")
            age_min = (pd.Timestamp.now().timestamp() - self.log_file.stat().st_mtime) / 60
        except OSError:
            log, age_min = "", 1e9
        m = re.search(r"^\s*(\d+)\s+requests in total", log, re.M)
        total = int(m.group(1)) if m else None
        done = len(re.findall(r"\[Sync\] Starting", log))
        lines = [ln.strip() for ln in log.splitlines() if ln.strip()]
        proc = self.registry.get(str(self.folder))
        alive = proc is not None and proc.poll() is None
        ok_m = re.search(r"OK: stock prices now run to (\d{4}-\d{2}-\d{2})", log)
        finished = re.search(r"requests attempted, (\d+) threw", log) is not None
        if finished:
            state = "finished_ok" if ok_m else "finished_warn"
        elif alive:
            state = "running"
        elif proc is not None:
            state = "failed"          # the process ended without printing its summary
        elif age_min < 10:
            state = "running"         # started by another Streamlit process and still writing the log
        else:
            state = "stalled"
        return {"state": state, "done": done, "total": total, "last_line": lines[-1][:200] if lines else "",
                "prices_to": ok_m.group(1) if ok_m else None, "started_at": st_.get("started_at"), "days": st_.get("days"),
                "trigger": st_.get("trigger"), "exit_code": proc.poll() if proc is not None else None}

    # ---------- decisions ----------
    def should_auto_update(self, min_hours=6, now_wib=None):
        """(start?, reason, days). Starts only when the database is behind, nothing is running and the last attempt is old enough."""
        ok, msg = self.readiness()
        if not ok:
            return False, msg, 0
        if self.status()["state"] == "running":
            return False, "a sync is already running", 0
        now = now_wib if now_wib is not None else pd.Timestamp.now(tz="Asia/Jakarta")
        latest, expected = idx_db_latest_stock_date(self.db), idx_expected_last_session(now)
        if latest is not None and latest >= expected:
            return False, f"the database is current (latest prices {latest:%Y-%m-%d})", 0
        prev = self._read_state()
        if prev and prev.get("started_at"):
            try:
                hrs = (now - pd.Timestamp(prev["started_at"])).total_seconds() / 3600
                if hrs < min_hours:
                    return False, f"the last attempt was {hrs:.1f} h ago (minimum {min_hours} h between automatic attempts)", 0
            except Exception:
                pass
        if latest is None:
            return True, "the database has no prices yet", 365
        gap = (expected - latest).days
        return True, f"the latest prices are {latest:%Y-%m-%d}, the last session was {expected:%Y-%m-%d}", int(min(max(gap + 10, 14), 420))

    # ---------- action ----------
    def start(self, days, per_company=False, trigger="auto"):
        """Launch Cron.ts detached from this process. Returns (ok, message). Never blocks."""
        ok, msg = self.readiness()
        if not ok:
            return False, msg
        if self.status()["state"] == "running":
            return False, "a sync is already running"
        cmd = [self.deno(), "run", "-A", "Cron.ts", f"--days={int(days)}"] + ([] if per_company else ["--no-per-company"])
        self._write_state(started_at=pd.Timestamp.now(tz="Asia/Jakarta").isoformat(), days=int(days), trigger=trigger, cmd=cmd)
        kw = dict(cwd=str(self.folder), stdin=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        if os.name == "nt":   # no console window, and Ctrl+C in the Streamlit console must not kill the sync
            kw["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        else:
            kw["start_new_session"] = True
        try:
            with open(self.log_file, "w", encoding="utf-8") as fh:
                proc = subprocess.Popen(cmd, stdout=fh, **kw)
        except OSError as e:
            return False, f"could not start Deno: {e}"
        self.registry[str(self.folder)] = proc
        return True, f"started in the background ({' '.join(cmd[1:])})"



@st.cache_resource
def _idx_sync_registry():
    """Keeps the Popen handles alive across Streamlit reruns (module-level objects are rebuilt on every rerun)."""
    return {}


def _idx_sync_status_body(db_path):
    sync = IDXAutoSync(db_path, _idx_sync_registry())
    s_ = sync.status()
    state = s_["state"]
    if state == "idle":
        st.caption("Auto-update: no run yet on this computer.")
    elif state == "running":
        tot, done = s_.get("total") or 0, s_.get("done", 0)
        st.info(f"⏳ Updating the database in the background: {done} of {tot or '?'} requests "
                f"({'automatic' if s_.get('trigger') == 'auto' else 'manual'} run, {s_.get('days')} days).")
        st.progress(min(done / tot, 1.0) if tot else 0.0)
        st.caption(s_.get("last_line", ""))
    elif state == "finished_ok":
        st.success(f"✅ Last update OK: stock prices now run to {s_.get('prices_to')}.")
    elif state == "finished_warn":
        st.warning("The last update finished but stored NO newer stock prices (no internet, idx.co.id refused the requests, or only "
                   "holidays in the window). See the log below.")
    else:
        st.error("The last update did not finish" + (f" (exit code {s_['exit_code']})" if s_.get("exit_code") is not None else "")
                 + ". See the log below.")
    with st.expander("Update log (last lines)", expanded=state in ("failed", "stalled")):
        st.code(sync.log_tail(25) or "(empty)", language=None)


if hasattr(st, "fragment"):
    @st.fragment(run_every=5)
    def _idx_sync_status_live(db_path):
        """Polls every 5 s while a sync runs, then reruns the whole app once so the analysis picks up the new data."""
        _idx_sync_status_body(db_path)
        if IDXAutoSync(db_path, _idx_sync_registry()).status()["state"] != "running":
            st.rerun()

    @st.fragment
    def _idx_sync_status_static(db_path):
        _idx_sync_status_body(db_path)
else:   # very old Streamlit: no fragments, so a manual refresh button
    def _idx_sync_status_live(db_path):
        _idx_sync_status_body(db_path)
        st.button("Refresh status", key="idx_sync_refresh_status")

    _idx_sync_status_static = _idx_sync_status_live



# ============================================================
# STOCK LOGOS & COMPANY NAME (v1.4.6)
# The logos are the SVG files <IDX-API>/sample/img/<CODE>.svg. STOCKLIST.md only REFERENCES them (and lists code, name and listing
# date; its names are cut at 30 characters), so the logo is looked up by file name, not through the table.
# ============================================================
import base64
import html as _html

IDX_CODE_RE = re.compile(r"^[A-Z0-9]{2,6}$")


def idx_asset_roots(db_path=""):
    """Folders that may be an IDX-API checkout, most specific first: the one that owns the configured database, those next to databases
    found automatically, then the usual places next to this script."""
    roots = []

    def add(p):
        try:
            p = Path(p).expanduser().resolve()
        except OSError:
            return
        if p.is_dir() and p not in roots:
            roots.append(p)

    if db_path:
        add(Path(idx_clean_path(db_path)).expanduser().parent.parent)
    for d in idx_discover_db_cached():
        add(Path(d["path"]).parent.parent)
    try:
        here = Path(__file__).resolve().parent
    except NameError:
        here = Path.cwd()
    for base in (here, Path.cwd()):
        for rel in ("Documents/IDX-API", "IDX-API", "."):
            add(base / rel)
    return roots


def idx_find_logo(code, roots, override=""):
    """Path of <CODE>.svg, or None. The code is validated first so arbitrary text can never become part of a path."""
    code = str(code).upper()
    if not IDX_CODE_RE.match(code):
        return None
    dirs = [Path(idx_clean_path(override)).expanduser()] if override else []
    for r in roots:
        dirs += [r / "sample" / "img", r / "img", r / "data" / "img"]
    for d in dirs:
        f = d / f"{code}.svg"
        if f.is_file():
            return f
    return None


def idx_find_stocklist(roots):
    for r in roots:
        for rel in ("data/STOCKLIST.md", "sample/STOCKLIST.md", "STOCKLIST.md"):
            if (r / rel).is_file():
                return r / rel
    return None


@st.cache_data(ttl=3600, show_spinner=False)
def idx_parse_stocklist(path, mtime):
    """STOCKLIST.md -> {code: {name, listed, name_truncated}}. The file cuts long company names at 30 characters. The RAW length is used
    (a name cut right after a space would measure 29 after strip), and a name ending in "Tbk" is complete even when it is exactly 30
    characters long (e.g. "Nanotech Indonesia Global Tbk.")."""
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {}
    out = {}
    for ln in text.splitlines():
        if not ln.startswith("|"):
            continue
        f = ln.split(" | ")
        if len(f) < 4:
            continue
        m = re.match(r"^\*\*([A-Z0-9]{2,6})\*\*$", f[1].strip())
        d = re.match(r"^(\d{4}-\d{2}-\d{2})\s*\|?\s*$", f[3])
        if not (m and d):
            continue
        raw, name = f[2], f[2].strip()
        out[m.group(1)] = {"name": name, "listed": d.group(1), "name_truncated": len(raw) >= 30 and not re.search(r"\bTbk\.?$", name)}
    return out


@st.cache_data(ttl=3600, show_spinner=False)
def idx_logo_data_uri(path, mtime):
    """Inline data URI for an <img> tag. Only plain vector artwork is accepted: an SVG holding a script, or an oversized file, is refused
    (inside <img> a script would not run anyway; this is belt and braces)."""
    try:
        raw = Path(path).read_bytes()
    except OSError:
        return None
    if len(raw) > 600_000 or b"<script" in raw.lower():
        return None
    return "data:image/svg+xml;base64," + base64.b64encode(raw).decode("ascii")


# 48 px was the visible logo inside the old 64 px frame; +20% = 57.6, rounded to 58.
IDX_LOGO_PX = 80


def idx_logo_html(data_uri, code, size=IDX_LOGO_PX):
    """The logo alone: no frame, no background, no border. The IDX-API artwork is almost always a round badge that carries its own
    background (933 of the 955 files), so it reads fine on both Streamlit themes. Returns "" when there is no logo."""
    if not data_uri:
        return ""
    return (f'<div style="display:flex;align-items:center;justify-content:center">'
            f'<img src="{data_uri}" width="{size}" height="{size}" style="display:block" alt="{_html.escape(str(code))} logo"></div>')

# ============================================================
# SIDEBAR CUSTOMIZATION
# ============================================================
# ============================================================
# macOS-STYLE THEME (Light / Dark)
# Streamlit cannot switch its own theme at runtime, so the look is applied with injected CSS variables. The st.dataframe grid is drawn on a
# canvas that CSS cannot restyle; when the chosen mode differs from Streamlit's own theme it is colour-inverted to match.
# ============================================================
MAC_THEMES = {
    "light": dict(bg="#f5f5f7", card="#ffffff", text="#1d1d1f", sub="#6e6e73", sep="rgba(60,60,67,0.18)",
                  ctl="rgba(118,118,128,0.12)", ctls="#e9e9ec", side="rgba(246,246,246,0.82)", accent="#007aff",
                  shadow="0 1px 3px rgba(0,0,0,0.08), 0 4px 14px rgba(0,0,0,0.05)", plotly="plotly_white", grid="rgba(60,60,67,0.10)"),
    "dark": dict(bg="#1c1c1e", card="#2c2c2e", text="#f5f5f7", sub="#98989d", sep="rgba(235,235,245,0.16)",
                 ctl="rgba(118,118,128,0.26)", ctls="#3a3a3c", side="rgba(44,44,46,0.80)", accent="#0a84ff",
                 shadow="0 1px 3px rgba(0,0,0,0.40), 0 4px 14px rgba(0,0,0,0.25)", plotly="plotly_dark", grid="rgba(235,235,245,0.10)"),
}
MAC_FONT = '-apple-system, BlinkMacSystemFont, "SF Pro Text", "SF Pro Display", "Segoe UI Variable", "Segoe UI", "Helvetica Neue", Helvetica, Arial, sans-serif'

try:
    _native_theme = st.context.theme.type            # "light" / "dark" as seen by Streamlit
except Exception:
    _native_theme = None
MAC_MODE = "dark"   # fixed; .streamlit/config.toml sets base = "dark", which also removes the Light/Dark picker from the Settings menu
MAC = MAC_THEMES[MAC_MODE]


def inject_mac_css(t, mismatch):
    css = f"""
    :root {{ --m-bg:{t['bg']}; --m-card:{t['card']}; --m-text:{t['text']}; --m-sub:{t['sub']}; --m-sep:{t['sep']};
             --m-ctl:{t['ctl']}; --m-ctls:{t['ctls']}; --m-side:{t['side']}; --m-accent:{t['accent']}; --m-shadow:{t['shadow']}; }}
    html, body, .stApp, button, input, textarea, select {{ font-family:{MAC_FONT} !important; -webkit-font-smoothing:antialiased; }}
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {{ background:var(--m-bg) !important; color:var(--m-text) !important; }}
    [data-testid="stHeader"] {{ background:transparent !important; }}
    [data-testid="stHeader"] * {{ color:var(--m-text) !important; }}
    .block-container {{ padding-top:3.2rem; }}

    /* text */
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
    .stApp p, .stApp li, .stApp label, .stApp summary, [data-testid="stWidgetLabel"] p, [data-testid="stMarkdownContainer"] {{ color:var(--m-text) !important; }}
    .stApp h1 {{ font-weight:700; letter-spacing:-0.02em; }}
    .stApp h2, .stApp h3 {{ font-weight:650; letter-spacing:-0.01em; }}
    [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] * {{ color:var(--m-sub) !important; }}
    .stApp hr {{ border-color:var(--m-sep) !important; }}
    .stApp a {{ color:var(--m-accent) !important; }}
    .stApp code {{ background:var(--m-ctl) !important; color:var(--m-text) !important; border-radius:6px; }}

    /* sidebar: translucent "vibrancy" panel */
    [data-testid="stSidebar"] {{ background:var(--m-side) !important; backdrop-filter:blur(24px) saturate(180%);
                                -webkit-backdrop-filter:blur(24px) saturate(180%); border-right:1px solid var(--m-sep); }}
    [data-testid="stSidebar"] > div:first-child {{ background:transparent !important; }}
    [data-testid="stSidebarHeader"] {{ background:transparent !important; }}

    /* inputs, selects, text areas */
    [data-baseweb="base-input"] {{ background:transparent !important; }}
    [data-baseweb="input"], [data-baseweb="textarea"], [data-baseweb="select"] > div {{
        background:var(--m-ctls) !important; border-color:transparent !important; border-radius:8px !important; color:var(--m-text) !important; }}
    [data-baseweb="input"]:focus-within, [data-baseweb="textarea"]:focus-within, [data-baseweb="select"]:focus-within > div {{
        box-shadow:0 0 0 3px color-mix(in srgb, var(--m-accent) 35%, transparent) !important; border-color:var(--m-accent) !important; }}
    input, textarea, [data-baseweb="select"] * {{ color:var(--m-text) !important; -webkit-text-fill-color:var(--m-text) !important; background-color:transparent; }}
    [data-baseweb="select"] svg, [data-baseweb="input"] svg {{ fill:var(--m-sub) !important; }}
    [data-testid="stNumberInput"] button {{ background:transparent !important; color:var(--m-sub) !important; border:none !important; }}
    [data-baseweb="popover"], [data-baseweb="popover"] > div, [data-baseweb="menu"], [role="listbox"] {{
        background:var(--m-card) !important; color:var(--m-text) !important; border-radius:10px !important; }}
    [data-baseweb="popover"] > div {{ box-shadow:var(--m-shadow) !important; border:1px solid var(--m-sep); }}
    [role="option"] {{ color:var(--m-text) !important; }}
    [role="option"]:hover, [role="option"][aria-selected="true"] {{ background:var(--m-ctl) !important; }}

    /* buttons: rounded, Mac "push button" feel */
    [data-testid^="stBaseButton"] {{ border-radius:8px !important; border:1px solid var(--m-sep) !important; background:var(--m-card) !important;
        color:var(--m-text) !important; box-shadow:0 1px 1px rgba(0,0,0,0.06); transition:filter .15s, transform .05s; }}
    [data-testid^="stBaseButton"] * {{ color:var(--m-text) !important; }}
    [data-testid^="stBaseButton"]:hover {{ filter:brightness(0.97); border-color:var(--m-sep) !important; }}
    [data-testid^="stBaseButton"]:active {{ transform:scale(0.98); }}
    [data-testid="stBaseButton-primary"], [data-testid="stBaseButton-primary"] * {{ background:var(--m-accent) !important; color:#fff !important; border-color:transparent !important; }}

    /* checkbox / radio / slider / toggle: system blue accent */
    [data-baseweb="checkbox"] > span:first-child {{ border-radius:5px !important; }}
    [data-baseweb="checkbox"] input:checked + div, [data-baseweb="radio"] input:checked + div {{ background-color:var(--m-accent) !important; border-color:var(--m-accent) !important; }}
    [data-testid="stCheckbox"] label span[data-baseweb="checkbox"] > span:first-child[class] {{ border-color:var(--m-sub); }}
    [data-baseweb="slider"] [role="slider"] {{ background:#fff !important; border:0.5px solid var(--m-sep) !important; box-shadow:0 1px 4px rgba(0,0,0,0.35) !important; }}
    [data-baseweb="slider"] [role="slider"] + div, [data-baseweb="slider"] [data-testid="stSliderThumbValue"] {{ color:var(--m-text) !important; }}
    [data-baseweb="tab-highlight"] {{ background:var(--m-accent) !important; }}
    [data-baseweb="tab-list"] button {{ color:var(--m-sub) !important; }}
    [data-baseweb="tab-list"] button[aria-selected="true"] {{ color:var(--m-text) !important; }}

    /* cards: metrics, expanders, alerts, charts */
    [data-testid="stMetric"] {{ background:var(--m-card); border:1px solid var(--m-sep); border-radius:12px; padding:12px 14px; box-shadow:var(--m-shadow); }}
    [data-testid="stMetricLabel"] *, [data-testid="stMetricLabel"] {{ color:var(--m-sub) !important; }}
    [data-testid="stMetricValue"], [data-testid="stMetricValue"] * {{ color:var(--m-text) !important; }}
    [data-testid="stExpander"] details {{ background:var(--m-card) !important; border:1px solid var(--m-sep) !important; border-radius:12px !important; box-shadow:var(--m-shadow); }}
    [data-testid="stExpander"] summary:hover {{ background:var(--m-ctl) !important; border-radius:12px; }}
    [data-testid="stAlert"] {{ border-radius:12px !important; border:1px solid var(--m-sep) !important; }}
    [data-testid="stAlert"] * {{ color:var(--m-text) !important; }}
    [data-testid="stPlotlyChart"] {{ background:var(--m-card); border:1px solid var(--m-sep); border-radius:14px; padding:6px; box-shadow:var(--m-shadow); overflow:hidden; }}
    /* Plotly toolbar: inset from the card's rounded corner (down and left) so it is not clipped */
    [data-testid="stPlotlyChart"] .modebar-container {{ top:16px !important; right:18px !important; }}
    [data-testid="stPlotlyChart"] .modebar {{ top:0 !important; right:0 !important; background:transparent !important; }}
    [data-testid="stTable"] table, [data-testid="stMarkdownContainer"] table {{ border-radius:10px; overflow:hidden; border-color:var(--m-sep) !important; }}
    [data-testid="stTable"] th, [data-testid="stTable"] td, [data-testid="stMarkdownContainer"] th, [data-testid="stMarkdownContainer"] td {{
        color:var(--m-text) !important; border-color:var(--m-sep) !important; background:transparent !important; }}
    [data-testid="stDataFrame"] {{ border-radius:12px; overflow:hidden; border:1px solid var(--m-sep); }}
    [data-testid="stTooltipContent"] {{ border-radius:8px; }}

    /* scrollbars */
    ::-webkit-scrollbar {{ width:10px; height:10px; }}
    ::-webkit-scrollbar-thumb {{ background:var(--m-sub); border-radius:10px; border:3px solid transparent; background-clip:content-box; opacity:.5; }}
    ::-webkit-scrollbar-track {{ background:transparent; }}
    """
    if mismatch:   # canvas-drawn data grid cannot be restyled with CSS; invert it to the chosen mode
        css += '[data-testid="stDataFrame"] canvas, [data-testid="stDataFrame"] [data-testid="stDataFrameResizable"] { filter:invert(1) hue-rotate(180deg); }'
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


inject_mac_css(MAC, mismatch=False)


def mac_plotly(fig):
    """Apply the active macOS theme to a Plotly figure (transparent so the card behind it shows; SF font; hairline grid)."""
    fig.update_layout(template=MAC["plotly"], paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family=MAC_FONT, color=MAC["text"]))
    fig.update_xaxes(gridcolor=MAC["grid"], zerolinecolor=MAC["grid"], linecolor=MAC["sep"])
    fig.update_yaxes(gridcolor=MAC["grid"], zerolinecolor=MAC["grid"], linecolor=MAC["sep"])
    return fig


st.sidebar.header("⚙️ Parameters & Customization")

# Defaults on first open / restart: BBRI.JK, the last 6 months of history
# (rolling, not a fixed date) up to today, daily interval — replicating the
# ready-to-use view, so the toggles do not have to be set up again one by one
# every time the app is opened.
default_end = pd.Timestamp.now().normalize()
default_start = default_end - pd.DateOffset(months=6)

ticker = st.sidebar.text_input("Stock Ticker", value="^JKSE")
if st.sidebar.button("🔄 Refresh market data", help="Market data is cached for 5 minutes. This clears the cache so the next run downloads the latest bars."):
    st.cache_data.clear()

# Take the clean name without the .JK suffix (e.g. BMRI.JK -> BMRI; ^JKSE -> JKSE)
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
bep_level = float(st.sidebar.number_input("Break-even price (BEP, optional)", value=0))

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
            help="Check IDX disclosures (www.idx.co.id) or your broker app's "
                 "dividend/corporate action history for exact figures."
        )
show_index_rebalance = st.sidebar.checkbox("Show MSCI / Index Rebalance Dates", value=True)
show_bi_rdg = st.sidebar.checkbox("Show BI Board of Governors' meetings (policy rate decision dates)", value=True)
show_rups = st.sidebar.checkbox("Show AGM/EGM (shareholders' meeting) dates — manual input", value=False)
rups_dates_input = ""
if show_rups:
    rups_dates_input = st.sidebar.text_area(
        "AGM/EGM dates (one per line, YYYY-MM-DD)",
        placeholder="2026-04-15\n2026-11-20",
        help="No free public API provides shareholders' meeting (AGM/EGM) schedules. Paste dates from IDX "
             "disclosures (www.idx.co.id) or the company's investor "
             "relations page."
    )
st.sidebar.caption("MSCI and BI meeting dates below are the **officially published 2026 schedule** "
                   "(verified against MSCI.com and Bank Indonesia press releases as of this app's "
                   "build date) — update the tables in code for other years once new schedules "
                   "are released.")

# ============================================================
# FORECAST ENGINE: Market regime -> Technical -> Fundamental -> probabilities -> target/range
# ============================================================
# Long-standing LQ45 & IDX30 members — used ONLY to pre-tick the membership
# checkboxes below. Verify against the official IDX list (revised every Feb/Aug).
KNOWN_CORE_MEMBERS = {"BBCA", "BBRI", "BMRI", "BBNI", "TLKM", "ASII"}

# Proxy basket for the advance/decline calculation. IDX publishes no free breadth
# API, so breadth is computed from this basket via yfinance. Edit it to match the
# current LQ45/IDX30 constituents for a closer proxy.
DEFAULT_BREADTH_BASKET = (
    "BBCA.JK,BBRI.JK,BMRI.JK,BBNI.JK,TLKM.JK,ASII.JK,UNVR.JK,ICBP.JK,INDF.JK,KLBF.JK,"
    "ADRO.JK,ANTM.JK,MDKA.JK,PTBA.JK,CPIN.JK,SMGR.JK,INKP.JK,AMRT.JK,MAPI.JK,EXCL.JK,"
    "TOWR.JK,JSMR.JK,PGAS.JK,ITMG.JK,BRPT.JK,GOTO.JK,MIKA.JK,ACES.JK,INCO.JK,TBIG.JK"
)

st.sidebar.subheader("🧭 Forecast Engine")
show_engine = st.sidebar.checkbox("Enable layered engine (Regime → Technical → Fundamental)", value=True)
tilt_strength = st.sidebar.slider(
    "Evidence tilt strength (%)", 0, 100, 50,
    help="How far the combined evidence (Regime + Technical + Fundamental layers) may shift the "
         "Bullish/Neutral/Bearish probabilities away from the no-view base rate. 0% = no tilt. "
         "At 100%, fully supportive evidence shifts the expected horizon return by +0.5 sigma "
         "(roughly 31/38/31 -> 50/35/15; at the default 50% strength, 40/37/23). "
         "The scenario price targets never change with the tilt - only their odds."
) / 100

engine_cfg = {
    "ticker": ticker, "ihsg_symbol": "^JKSE", "lq45_symbol": "^JKLQ45", "idx30_symbol": "",
    "usdidr_symbol": "IDR=X", "dxy_symbol": "DX-Y.NYB", "spx_symbol": "^GSPC", "vix_symbol": "^VIX", "em_symbol": "EEM",
    "is_lq45": clean_ticker in KNOWN_CORE_MEMBERS, "is_idx30": clean_ticker in KNOWN_CORE_MEMBERS,
    "basket": tuple(t.strip() for t in DEFAULT_BREADTH_BASKET.split(",") if t.strip()),
    "bi_rate": 5.75, "bi_expect_bps": 0,
    "fed_manual": 0.0, "ust10_manual": 0.0, "id10y_manual": 0.0, "id10y_prior_manual": 0.0,
    "foreign_stock_bn": 0.0, "foreign_mkt_bn": 0.0,
    "roe_manual": 0.0, "roa_manual": 0.0, "pbv_manual": 0.0, "evebitda_manual": 0.0,
    "per_manual": 0.0, "eps_growth_manual": 0.0,
    "idx_api_enabled": False, "idx_api_db": "", "idx_api_days": 20,
    "erp": 5.5, "g": 5.0,
    "layer_weights": {"Regime": 0.40, "Technical": 0.35, "Fundamental": 0.25},
}
if show_engine:
    with st.sidebar.expander("Layer weights", expanded=False):
        st.caption("Relative weight of each layer in the combined evidence. Weights are normalised over the "
                   "layers that have enough data, so they need not add up to 100.")
        w_reg = st.slider("1 · Market regime (%)", 0, 100, 40)
        w_tec = st.slider("2 · Technical (%)", 0, 100, 35)
        w_fun = st.slider("3 · Fundamental (%)", 0, 100, 25)
        engine_cfg["layer_weights"] = {"Regime": w_reg / 100, "Technical": w_tec / 100, "Fundamental": w_fun / 100}

    with st.sidebar.expander("🇮🇩 IDX-API data source (v1.4.6)", expanded=True):
        st.caption("Reads the SQLite file written by NeaByteLab/IDX-API (a Deno sync tool, not a hosted REST service). "
                   "**Additional information only:** foreign flow, dividends and fundamental ratios. Prices, trends, indicators and the JCI "
                   "always come from yfinance. The live database is `<IDX-API folder>/data/database.sqlite`.")
        _idx_found = idx_discover_db_cached()
        _idx_default = next((d["path"] for d in _idx_found if d["kind"] == "sqlite"), "")
        # Switched on by default ONLY when the LIVE database (data/database.sqlite, the file a sync writes) was found and opens; the Git-LFS
        # sample snapshot is old and a sync never updates it, so it must not turn the source on by itself.
        _idx_default_on = bool(_idx_default) and Path(_idx_default).name == "database.sqlite" and idx_diagnose_db(_idx_default)["ok"]
        engine_cfg["idx_api_enabled"] = st.checkbox(
            "Enable IDX-API SQLite source", value=_idx_default_on, key="idx_api_enabled_v141",
            help="Switched on automatically at start-up when the live database (data/database.sqlite) is found next to this script, so the "
                 "auto-update below can run on the very first launch. Untick it to ignore the database completely.")
        if _idx_default_on and engine_cfg["idx_api_enabled"] and not st.session_state.get("idx_default_on_noted"):
            st.session_state["idx_default_on_noted"] = True
            st.caption(f"Enabled automatically: a live database was found at {_idx_default}")
        _idx_raw = st.text_input(
            "IDX-API SQLite DB path (the .sqlite FILE, not the folder)", value=_idx_default, key="idx_api_db_v141",
            placeholder="e.g. C:/Users/<you>/.../IDX-API/data/database.sqlite",
            help="Quotes are stripped, so Windows 'Copy as path' works. If a database is found next to this script it is filled in automatically.")
        engine_cfg["idx_api_db"] = idx_clean_path(_idx_raw)
        engine_cfg["idx_api_days"] = st.number_input("Foreign-flow lookback (trading days)", min_value=5, max_value=60, value=20, step=5)
        engine_cfg["idx_api_max_age"] = st.number_input(
            "Treat IDX-API data older than this as stale (calendar days)", min_value=3, max_value=60, value=10, step=1,
            help="A database that was synced a while ago must not silently feed today's forecast. Foreign-flow data older than this is ignored "
                 "(its date is always shown next to the number).")
        engine_cfg["idx_api_ratio_priority"] = st.radio(
            "Fundamental ratios: which source wins?", ["yfinance first (TTM)", "IDX-API first (report period)"], index=0,
            help="IDX-API ratios belong to the last REPORT PERIOD (for Mar/Jun/Sep a cumulative year-to-date figure; annualisation is not documented). "
                 "yfinance gives trailing-12-month figures. The other source fills any gap.")
        if st.button("Reload IDX-API database", key="idx_api_reload_v141", help="Clears the cached table catalogue — use after running a new sync."):
            st.session_state["idx_reload_requested"] = True   # get_idx_adapter is defined further down; cleared there (the sync registry is kept)
            idx_discover_db_cached.clear()
        if engine_cfg["idx_api_db"] or engine_cfg["idx_api_enabled"]:
            _d = idx_diagnose_db(engine_cfg["idx_api_db"])
            {"ok": st.success, "warning": st.warning, "error": st.error}.get(_d["level"], st.warning)(_d["message"])
            if _d["hint"]:
                st.caption(_d["hint"])
        if _idx_found and not (engine_cfg["idx_api_db"] and idx_diagnose_db(engine_cfg["idx_api_db"])["ok"]):
            st.caption("Found on this computer: " + " · ".join(
                f"{'✅' if d['kind'] == 'sqlite' else '⚠️ LFS pointer' if d['kind'] == 'lfs_pointer' else '❔'} {d['path']}" for d in _idx_found[:4]))

        st.markdown("**Auto-update**")
        engine_cfg["idx_autosync"] = st.checkbox(
            "Update the database automatically when the app opens", value=True, key="idx_autosync_v144",
            help="Runs `deno run -A Cron.ts` in the BACKGROUND when the database is behind the last trading session. The app never waits for it: "
                 "it analyses with the data that exists now and reruns itself when the sync has finished. Needs Deno and the modified Cron.ts.")
        _as_days = st.number_input("Days to refresh (0 = automatic from how far behind the database is)", min_value=0, max_value=420, value=0, step=5, key="idx_autosync_days_v144")
        _as_pc = st.checkbox("Also download per-company data (slow, the app does not use it)", value=False, key="idx_autosync_pc_v144")
        _as_hours = st.number_input("Minimum hours between automatic attempts", min_value=1, max_value=72, value=6, step=1, key="idx_autosync_hours_v144",
                                    help="Stops the app from hitting IDX every time you open it, e.g. on a market holiday when the database cannot get newer.")
        if engine_cfg["idx_api_enabled"] and engine_cfg["idx_api_db"]:
            _sync = IDXAutoSync(engine_cfg["idx_api_db"], _idx_sync_registry())
            _ready, _ready_msg = _sync.readiness()
            if not _ready:
                st.warning("Auto-update unavailable: " + _ready_msg)
            else:
                _go, _why, _auto_days = _sync.should_auto_update(_as_hours)
                if st.button("⬇️ Update database now", key="idx_autosync_now_v144", help="Starts the sync immediately, in the background."):
                    _ok, _msg = _sync.start(_as_days or _auto_days or 14, _as_pc, "manual")
                    (st.success if _ok else st.warning)(("Update " + _msg) if _ok else _msg.capitalize())
                elif engine_cfg["idx_autosync"] and not st.session_state.get("idx_autosync_checked"):
                    st.session_state["idx_autosync_checked"] = True   # decide once per browser session
                    if _go:
                        _ok, _msg = _sync.start(_as_days or _auto_days, _as_pc, "auto")
                        (st.success if _ok else st.warning)(("Auto-update " + _msg + f" because {_why}.") if _ok else ("Auto-update not started: " + _msg))
                    else:
                        st.caption(f"Auto-update not needed: {_why}.")
                (_idx_sync_status_live if _sync.status()["state"] == "running" else _idx_sync_status_static)(engine_cfg["idx_api_db"])

    with st.sidebar.expander("Data sources & manual inputs", expanded=False):
        st.caption("Values left at 0 use the automatic source. Anything you type overrides it. IDX-API, when enabled, is preferred for IDX-native data.")
        st.markdown("**Market symbols (Yahoo Finance)**")
        engine_cfg["ihsg_symbol"] = st.text_input("JCI symbol (Jakarta Composite Index)", value="^JKSE")
        engine_cfg["usdidr_symbol"] = st.text_input("USD/IDR symbol", value="IDR=X")
        engine_cfg["dxy_symbol"] = st.text_input("Dollar index (DXY) symbol", value="DX-Y.NYB")
        engine_cfg["spx_symbol"] = st.text_input("Global equity benchmark (S&P 500)", value="^GSPC")
        engine_cfg["vix_symbol"] = st.text_input("Global risk gauge (VIX)", value="^VIX")
        engine_cfg["em_symbol"] = st.text_input("Emerging-market equities proxy", value="EEM")
        engine_cfg["lq45_symbol"] = st.text_input(
            "LQ45 index symbol", value="^JKLQ45",
            help="Yahoo Finance coverage of Indonesian sub-indices is patchy. Clear it if it returns no data.")
        engine_cfg["idx30_symbol"] = st.text_input(
            "IDX30 index symbol (optional)", value="",
            help="Left blank on purpose - no IDX30 symbol is verified on Yahoo. Fill one in if you find a working symbol.")
        engine_cfg["is_lq45"] = st.checkbox("Stock is an LQ45 constituent", value=engine_cfg["is_lq45"])
        engine_cfg["is_idx30"] = st.checkbox("Stock is an IDX30 constituent", value=engine_cfg["is_idx30"])
        basket_text = st.text_area("Breadth proxy basket (comma-separated Yahoo tickers)", value=DEFAULT_BREADTH_BASKET, height=120)
        engine_cfg["basket"] = tuple(t.strip() for t in basket_text.split(",") if t.strip())

        st.markdown("**Rates & policy**")
        engine_cfg["bi_rate"] = st.number_input("BI Rate (%) — last known: 5.75% held at the Sep 2026 BI meeting; update after each BI meeting", value=5.75, step=0.25)
        engine_cfg["bi_expect_bps"] = st.number_input(
            "BI-Rate expectation: expected change over next ~6 months (bps; negative = cuts)", value=0, step=25,
            help="No free API publishes the consensus. Use the latest Bloomberg/Reuters poll, broker research, or BI forward guidance.")
        engine_cfg["fed_manual"] = st.number_input("Fed Funds Rate (%) — 0 = auto (FRED: DFF)", value=0.0, step=0.25)
        engine_cfg["ust10_manual"] = st.number_input("US 10Y Treasury yield (%) — 0 = auto (^TNX)", value=0.0, step=0.05)
        engine_cfg["id10y_manual"] = st.number_input(
            "Indonesia 10Y yield (%) — 0 = auto (FRED OECD monthly series, lagged)", value=0.0, step=0.05)
        engine_cfg["id10y_prior_manual"] = st.number_input(
            "Indonesia 10Y yield ~3 months ago (%) — optional, used for the trend", value=0.0, step=0.05)

        st.markdown("**Foreign flow fallback / manual override**")
        st.caption("With IDX-API enabled, these values are only used when IDX-API cannot supply the requested flow.")
        engine_cfg["foreign_stock_bn"] = st.number_input(
            "Stock: net foreign buy(+)/sell(-), last 20 trading days (Rp bn)", value=0.0, step=50.0)
        engine_cfg["foreign_mkt_bn"] = st.number_input(
            "Market (JCI): net foreign buy(+)/sell(-), last 20 trading days (Rp bn)", value=0.0, step=500.0)

        st.markdown("**Fundamentals** (0 = auto from yfinance)")
        engine_cfg["eps_growth_manual"] = st.number_input("EPS growth (%) — any non-zero value overrides", value=0.0, step=1.0)
        engine_cfg["roe_manual"] = st.number_input("ROE (%)", value=0.0, step=0.5)
        engine_cfg["roa_manual"] = st.number_input("ROA (%)", value=0.0, step=0.1)
        engine_cfg["pbv_manual"] = st.number_input("PBV (x)", value=0.0, step=0.1)
        engine_cfg["per_manual"] = st.number_input("PER (x)", value=0.0, step=0.5)
        engine_cfg["evebitda_manual"] = st.number_input("EV/EBITDA (x) — not meaningful for banks", value=0.0, step=0.5)
        engine_cfg["erp"] = st.number_input("Equity risk premium (%) — for the justified-PBV check", value=5.5, step=0.5)
        engine_cfg["g"] = st.number_input("Long-term growth g (%) — for the justified-PBV check", value=5.0, step=0.5)

st.sidebar.subheader("🎲 Scenario Probabilities")
show_probabilities = st.sidebar.checkbox("Show Bullish / Neutral / Bearish probabilities", value=True)
prob_band_sigma = st.sidebar.slider(
    "Neutral band (± x horizon sigma)", 0.20, 1.00, 0.50, 0.05,
    help="'Neutral' = the price ends the forecast horizon within +/- this many standard deviations "
         "of the horizon return. Wider band = more probability mass in Neutral. The band is shown "
         "in rupiah in the engine panel.")
show_forecast_line = st.sidebar.checkbox(
    "Forecast line (pattern projection)", value=True,
    help="A dashed line: the projection of the detected pattern (Bullish Flag etc.) to its Fibonacci / measured-move target. "
         "It also gives the dates at which the line reaches the trendlines, the pattern lines and the Fibonacci levels (see the Forecast line panel).")

# DEEP FORECAST (v1.5.0)
st.sidebar.subheader("🤖 Deep Forecast (ML)")
show_deep, ml_use_rf, ml_use_xgb, ml_use_sarimax = False, True, True, True
ml_blend_cap, use_blend = 40, True
ml_trees, ml_splits, ml_order, ml_seasonal = 200, 5, (1, 1, 1), (0, 0, 0, 0)
if dfc is None:
    st.sidebar.warning(f"Deep Forecast is unavailable ({DEEP_IMPORT_ERR}). Install it with: pip install scikit-learn xgboost statsmodels")
else:
    show_deep = st.sidebar.checkbox(
        "Enable Deep Forecast (Random Forest · XGBoost · SARIMAX)", value=True,
        help="Trains on this stock's own ~3-year daily history, validates walk-forward against a naive baseline, and predicts the "
             "same horizon as the Forecast Engine. Cached for 15 minutes; the first run takes roughly 15-30 seconds.")
    if show_deep:
        with st.sidebar.expander("Deep Forecast settings", expanded=False):
            ml_use_rf = st.checkbox("Random Forest", value=True, key="ml_rf")
            ml_use_xgb = st.checkbox("XGBoost", value=True, key="ml_xgb")
            ml_use_sarimax = st.checkbox("SARIMAX (statistical path + 80% band)", value=True, key="ml_sx")
            ml_trees = st.slider("Trees / boosting rounds", 100, 600, 200, 50, key="ml_trees",
                                 help="More rounds = slower, rarely better on ~600 samples.")
            ml_splits = st.slider("Walk-forward folds", 3, 8, 5, key="ml_splits")
            st.caption("SARIMAX order (p, d, q) on the log price")
            _c1, _c2, _c3 = st.columns(3)
            _sp, _sd, _sq = (_c1.number_input("p", 0, 4, 1, key="ml_p"), _c2.number_input("d", 0, 2, 1, key="ml_d"),
                             _c3.number_input("q", 0, 4, 1, key="ml_q"))
            ml_order = (int(_sp), int(_sd), int(_sq))
            ml_seasonal = (1, 0, 0, 5) if st.checkbox("Weekly seasonality (5 trading days)", value=False, key="ml_seas") else (0, 0, 0, 0)
            use_blend = st.checkbox("Blend ML into the scenario probabilities", value=True, key="ml_blend",
                                    help="Skill-weighted: the blended probabilities feed the probability-weighted path and the scenario labels. "
                                         "With no out-of-sample ML skill the weight is 0 and the engine is left untouched.")
            ml_blend_cap = st.slider("Max ML weight in the blend (%)", 0, 60, 40, 5, key="ml_blend_cap",
                                     help="The actual weight = this cap x the classifiers' out-of-sample advantage (0-100%).")

with st.sidebar.expander("🖼️ Stock logo", expanded=False):
    show_logo = st.checkbox("Show the company logo and name in the header", value=True, key="show_logo_v146")
    logo_dir_override = st.text_input(
        "Logo folder (optional)", value="", key="logo_dir_v146",
        help="Folder with <CODE>.svg files. Leave empty: the app looks in <IDX-API folder>/sample/img, next to your IDX-API database.")
    _lg_found = idx_find_logo(clean_ticker, idx_asset_roots(engine_cfg.get("idx_api_db", "")), logo_dir_override)
    if _lg_found:
        st.caption(f"✅ {clean_ticker}: {_lg_found}")
    elif IDX_CODE_RE.match(clean_ticker.upper()):
        st.caption(f"ℹ️ No logo file for {clean_ticker}. The logos are the SVG files in <IDX-API folder>/sample/img "
                   "(STOCKLIST.md only lists them; a few stocks have none).")
    else:
        st.caption("ℹ️ Logos exist only for IDX stock codes.")

plotly_date_format_map = {
    "%b %Y": "%b %Y",
    "%d-%b-%Y": "%d-%b-%Y",
    "%Y-%m-%d": "%Y-%m-%d"
}

# ============================================================
# DYNAMIC ENGLISH TITLE
# ============================================================
_header_slot = st.empty()   # replaced by the logo header below once the data is loaded; this plain title stays if anything fails first
_header_slot.title(f"📈 {clean_ticker} Macro Pattern, Fibonacci & Cycle Projection · v1.4.6")
st.caption(f"Automated Macro Pattern Detection, Dynamic Fibonacci Retracement, and Wave-Based Projection App for {clean_ticker} · Build {APP_VERSION}")
if engine_cfg.get("idx_api_enabled"):
    st.caption(f"🇮🇩 IDX-API v1.4.6: {'configured' if engine_cfg.get('idx_api_db') else 'enabled but DB path is empty'} | price source is shown under the chart title once data loads")

# ============================================================
# LOAD STOCK DATA
# ============================================================
def bar_is_forming(last_bar_date, now_wib=None):
    """True when the last daily bar is today's and the IDX cash market is still open (Mon-Fri, ~09:00-15:50 WIB,
    with a small margin for the data feed). Pure function so it can be tested with explicit times."""
    now = now_wib if now_wib is not None else pd.Timestamp.now(tz="Asia/Jakarta")
    if pd.Timestamp(last_bar_date).normalize() != pd.Timestamp(now.year, now.month, now.day):
        return False
    return now.weekday() < 5 and (9, 0) <= (now.hour, now.minute) < (16, 15)


@st.cache_data(ttl=300)
def load_stock_data(symbol, start, end, tf):
    # yfinance's `end` is EXCLUSIVE, so End Date = 6 Oct used to stop at the 5 Oct bar. Ask for end + 1 day
    # so the date shown in the sidebar is included (the latest bar, which may still be forming intraday).
    end_incl = pd.Timestamp(end) + pd.Timedelta(days=1)
    df = yf.download(symbol, start=start, end=end_incl, interval=tf, auto_adjust=False, progress=False)
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    df.index.name = "Date"   # every consumer (pattern detectors call reset_index() then read cycle["Date"]) gets the same index name
    return df


# ============================================================
# FUNCTION: HISTORICAL S/R ZONES (polarity principle)
# ============================================================
def detect_sr_zones(df, pivot_distance=4, tolerance_pct=0.015, min_touches=2,
                     extend_bars=3, max_span_bars=40):
    """
    Detect horizontal support/resistance zones following the POLARITY principle
    (old support -> new resistance, and so on; see the reference picture "Support
    Line Turns Into Resistance"):

    1. Find all pivot highs and lows in the data.
    2. Cluster by PRICE PROXIMITY (not by time) -> one "price zone" touched many
       times over the years, then split again per max_span_bars so that each zone
       represents a single active period (not one giant line spanning years).
    3. Each zone is drawn ONLY over the date range in which it was actively touched
       (+ a few extend_bars to the right) -- exactly the style of the reference:
       a short horizontal segment per zone, not one full-width line.
    4. The role (support/resistance) is decided from the median Close AFTER the
       zone formed: majority above -> SUPPORT, majority below -> RESISTANCE. If the
       SAME level reappears in another cluster at a different period with a
       different role, that is the polarity flip being captured automatically.

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
        # split the cluster into sub-groups by time when the span > max_span_bars
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


# ============================================================
# IDX-API SQLITE ADAPTER · v1.4.6
# Reads the SQLite database written by NeaByteLab/IDX-API. Table and column names are the exact
# ones from that project's Drizzle schemas (looked up by normalised equality, never by substring —
# substring matching silently picked the wrong column: 'id' as an ex-date, 'eps' as EPS growth).
# Every method opens its own short-lived connection (see idx_connect) and fails soft.
# ============================================================
IDX_TABLES = {
    "profile": "company_profile",     # code, name (full company name), listing_date
    "foreign": "foreign_trading",     # market-wide foreign buy/sell VALUE (Rp), one row per day
    "stock": "stock_summary",         # per-stock daily OHLC/volume/value; foreign_* columns are in SHARES
    "ratio": "financial_ratio",       # per-company ratios by report period; ROE/ROA/NPM already in PERCENT
    "dividend": "company_dividend",   # dividend announcements; ex_dividend date, cash_dividend per share
    "index_summary": "index_summary", # daily OHLCV per index code ('COMPOSITE' = JCI)
    "index_daily": "daily_index",     # daily close per index NAME ('Composite Index')
}
IDX_WIB = pd.Timedelta(hours=7)


def _idx_norm(x):
    return "".join(ch for ch in str(x).lower() if ch.isalnum())


def idx_to_datetime(values):
    """IDX-API stores dates as integers: epoch MILLISECONDS (pd.to_datetime on them reads
    nanoseconds and lands in 1970). Some tables hold local-midnight WIB (17:00Z of the previous day),
    others UTC midnight, so shift by +7h and floor to the day: both land on the right trading date.
    Also accepts epoch seconds, YYYYMMDD integers and ISO strings. Returns naive Timestamps."""
    raw = pd.Series(values).reset_index(drop=True)
    num = pd.to_numeric(raw, errors="coerce")
    out = pd.Series(pd.NaT, index=raw.index, dtype="datetime64[ns]")
    m_ms, m_s = num >= 1e11, (num >= 1e8) & (num < 1e11)
    m_ymd = (num >= 19000101) & (num <= 21001231)
    if m_ms.any():
        out[m_ms] = pd.to_datetime(num[m_ms], unit="ms") + IDX_WIB
    if m_s.any():
        out[m_s] = pd.to_datetime(num[m_s], unit="s") + IDX_WIB
    if m_ymd.any():
        out[m_ymd] = pd.to_datetime(num[m_ymd].astype("int64").astype(str), format="%Y%m%d", errors="coerce")
    rest = num.isna() & raw.notna()
    if rest.any():
        out[rest] = pd.to_datetime(raw[rest].astype(str), errors="coerce")
    return out.dt.normalize()


def _idx_num(v):
    v = pd.to_numeric(pd.Series([v]), errors="coerce").iloc[0]
    return float(v) if pd.notna(v) else None


class IDXAPIAdapter:
    """Read-only, schema-aware adapter over the SQLite output of NeaByteLab/IDX-API."""

    def __init__(self, db_path):
        self.db_path = idx_clean_path(db_path)
        self.diag = idx_diagnose_db(self.db_path)
        self.tables, self._schemas, self.notes = [], {}, []
        if self.diag["ok"]:
            try:
                with closing(idx_connect(self.db_path)) as c:
                    self.tables = [r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'").fetchall()]
                    for t in self.tables:
                        self._schemas[t] = [r[1] for r in c.execute(f'PRAGMA table_info("{t.replace(chr(34), chr(34) * 2)}")').fetchall()]
            except Exception as e:
                self.notes.append(f"IDX-API SQLite open failed: {type(e).__name__}: {e}")
                self.tables, self._schemas = [], {}

    @property
    def available(self):
        return bool(self.tables)

    # ---------- plumbing ----------
    def _note(self, msg):
        self.notes = (self.notes + [msg])[-20:]

    def _query(self, sql, params=()):
        try:
            with closing(idx_connect(self.db_path)) as c:
                return pd.read_sql_query(sql, c, params=params)
        except Exception as e:
            self._note(f"IDX-API query failed: {type(e).__name__}: {e}")
            return pd.DataFrame()

    def _table(self, key):
        n = _idx_norm(IDX_TABLES[key])
        for t in self._schemas:
            if _idx_norm(t) == n:
                return t
        for t in self._schemas:
            if n in _idx_norm(t):
                return t
        return None

    def _col(self, table, *names):
        norm = {_idx_norm(c): c for c in self._schemas.get(table, [])}
        for nm in names:
            if _idx_norm(nm) in norm:
                return norm[_idx_norm(nm)]
        return None

    @staticmethod
    def _tk(ticker):
        return str(ticker).upper().replace(".JK", "").replace("IDX:", "").strip()

    @staticmethod
    def _recent(df, days):
        """Last `days` sessions, but only if they sit inside a sane calendar window (no sync gaps)."""
        df = df.sort_values("Date")
        latest = df["Date"].max()
        win = df[df["Date"] >= latest - pd.Timedelta(days=int(days * 7 / 5) + 3)].tail(days)
        return win, latest

    def inventory(self):
        """Row count and latest date of every table this app reads — shows how far the sync reached."""
        rows = []
        for key, date_names in (("stock", ("date",)), ("index_summary", ("date",)), ("index_daily", ("date",)),
                                ("foreign", ("date",)), ("ratio", ("period",)), ("dividend", ("period",))):
            t = self._table(key)
            if not t:
                rows.append({"table": IDX_TABLES[key], "rows": None, "latest": None, "periodic": key in ("ratio", "dividend")})
                continue
            dc = self._col(t, *date_names)
            d = self._query(f'SELECT COUNT(*) AS n, MAX("{dc}") AS mx FROM "{t}"') if dc else pd.DataFrame()
            n = int(d["n"].iloc[0]) if len(d) else None
            mx = idx_to_datetime(d["mx"]).iloc[0] if (len(d) and pd.notna(d["mx"].iloc[0])) else None
            rows.append({"table": t, "rows": n, "latest": mx if pd.notna(mx) else None, "periodic": key in ("ratio", "dividend")})
        return rows

    # ---------- price series ----------
    def ohlcv(self, ticker, start=None, end=None):
        t = self._table("stock")
        if not t:
            return pd.DataFrame(), "stock_summary table not found"
        cc, dc = self._col(t, "code", "ticker", "symbol"), self._col(t, "date")
        cols = {k: self._col(t, k) for k in ("open", "high", "low", "close", "volume")}
        if not (cc and dc and cols["close"]):
            return pd.DataFrame(), "stock_summary lacks code/date/close columns"
        sel = ", ".join(f'"{c}"' for c in [dc] + [v for v in cols.values() if v])
        d = self._query(f'SELECT {sel} FROM "{t}" WHERE UPPER("{cc}") = ? ORDER BY "{dc}"', (self._tk(ticker),))
        if d.empty:
            return pd.DataFrame(), f"no rows for {self._tk(ticker)} in {t}"
        out = pd.DataFrame({"Date": idx_to_datetime(d[dc]).values})
        for tgt, key in (("Open", "open"), ("High", "high"), ("Low", "low"), ("Close", "close"), ("Volume", "volume")):
            out[tgt] = pd.to_numeric(d[cols[key]], errors="coerce").values if cols[key] else np.nan
        out = out.dropna(subset=["Date", "Close"]).drop_duplicates("Date", keep="last").sort_values("Date").set_index("Date")
        out = out[out["Close"] > 0]
        if out.empty:
            return pd.DataFrame(), f"no priced rows for {self._tk(ticker)}"
        # IDX stores 0 for fields it has no value for: no-trade days have O/H/L = 0, and many traded days
        # have Open = 0. Repair so charts and indicators stay sane.
        for c in ("High", "Low"):
            out.loc[out[c].fillna(0) <= 0, c] = out["Close"]
        no_trade = out["Volume"].fillna(0) <= 0
        out.loc[no_trade & (out["Open"].fillna(0) <= 0), "Open"] = out["Close"]
        open_missing = out["Open"].fillna(0) <= 0
        n_open = int(open_missing.sum())
        out.loc[open_missing, "Open"] = out["Close"].shift(1).where(lambda s: s.notna(), out["Close"]).clip(lower=out["Low"], upper=out["High"])[open_missing]
        db_first, db_last = out.index.min(), out.index.max()
        if start is not None:
            out = out[out.index >= pd.Timestamp(start)]
        if end is not None:
            out = out[out.index <= pd.Timestamp(end)]
        if out.empty:
            return out, (f"the database has {self._tk(ticker)} prices only for {db_first:%Y-%m-%d} to {db_last:%Y-%m-%d}, "
                         "none inside the chart's date range — sync newer data")
        extra = f"; {n_open} missing Open values approximated by the previous close" if n_open else ""
        return out[["Open", "High", "Low", "Close", "Volume"]], f"IDX-API SQLite:{t} (last bar {out.index.max():%Y-%m-%d}{extra})"

    def index_ohlcv(self, index_code="COMPOSITE", start=None, end=None, max_age_days=10):
        """JCI series. index_summary (one request per DAY, has High/Low/Volume) is usually short, while
        daily_index (one request per MONTH, close only) is cheap to sync far back — so merge them: close
        from index_summary where it exists, otherwise from daily_index; Volume only where index_summary has it."""
        today = pd.Timestamp.now().normalize()
        summ, dly, parts = pd.DataFrame(), pd.DataFrame(), []
        t = self._table("index_summary")
        if t:
            cc, dc = self._col(t, "code"), self._col(t, "date")
            vals = {k: self._col(t, k) for k in ("open", "high", "low", "close", "volume")}
            if cc and dc and vals["close"]:
                sel = ", ".join(f'"{c}"' for c in [dc] + [v for v in vals.values() if v])
                d = self._query(f'SELECT {sel} FROM "{t}" WHERE UPPER("{cc}") = ? ORDER BY "{dc}"', (index_code.upper(),))
                if len(d):
                    summ = pd.DataFrame({"Date": idx_to_datetime(d[dc]).values})
                    for tgt, key in (("Open", "open"), ("High", "high"), ("Low", "low"), ("Close", "close"), ("Volume", "volume")):
                        summ[tgt] = pd.to_numeric(d[vals[key]], errors="coerce").values if vals[key] else np.nan
                    summ = summ.dropna(subset=["Date", "Close"]).drop_duplicates("Date", keep="last").set_index("Date").sort_index()
                    if vals["open"] is None:   # index_summary has no Open column
                        summ["Open"] = summ["Close"].shift(1)
        t = self._table("index_daily")
        if t:
            nc, dc, cl = self._col(t, "name"), self._col(t, "date"), self._col(t, "close")
            if nc and dc and cl:
                d = self._query(f'SELECT "{dc}", "{cl}" FROM "{t}" WHERE LOWER("{nc}") IN (?, ?, ?) ORDER BY "{dc}"',
                                ("composite index", index_code.lower(), "ihsg"))
                if len(d):
                    dly = pd.DataFrame({"Date": idx_to_datetime(d[dc]).values, "Close": pd.to_numeric(d[cl], errors="coerce").values})
                    dly = dly.dropna(subset=["Date", "Close"]).drop_duplicates("Date", keep="last").set_index("Date").sort_index()
        if summ.empty and dly.empty:
            return pd.DataFrame(), "no JCI rows found (looked for index_summary code COMPOSITE, then daily_index 'Composite Index')"
        if not summ.empty and not dly.empty:
            out = summ.combine_first(dly)
            extra = int((~dly.index.isin(summ.index)).sum())
            src = f"IDX-API SQLite:index_summary ({len(summ)} rows, with volume) + daily_index ({extra} older close-only rows)"
        elif not summ.empty:
            out, src = summ, f"IDX-API SQLite:index_summary ({len(summ)} rows)"
        else:
            out, src = dly, f"IDX-API SQLite:daily_index ({len(dly)} rows, close only)"
        out = out.sort_index()
        last = out.index.max()
        if max_age_days is not None and (today - last).days > max_age_days:
            return pd.DataFrame(), f"STALE: last JCI row is {last:%Y-%m-%d} ({(today - last).days} days old; limit {max_age_days})"
        if start is not None:
            out = out[out.index >= pd.Timestamp(start)]
        if end is not None:
            out = out[out.index <= pd.Timestamp(end)]
        return out, f"{src}, last {last:%Y-%m-%d}"

    # ---------- dividends ----------
    def company_name(self, ticker):
        """Full company name from company_profile (the names in STOCKLIST.md are cut at 30 characters); None when the table or row is missing."""
        t = self._table("profile")
        if not t:
            return None
        cc, nc = self._col(t, "code", "ticker", "symbol"), self._col(t, "name")
        if not (cc and nc):
            return None
        d = self._query(f'SELECT "{nc}" FROM "{t}" WHERE UPPER("{cc}") = ? LIMIT 1', (self._tk(ticker),))
        if len(d) == 0 or pd.isna(d.iloc[0, 0]):
            return None
        return str(d.iloc[0, 0]).strip() or None

    def dividends(self, ticker):
        t = self._table("dividend")
        if not t:
            return pd.Series(dtype=float), "company_dividend table not found"
        cc, xc, vc = self._col(t, "code", "ticker", "symbol"), self._col(t, "ex_dividend"), self._col(t, "cash_dividend")
        if not (cc and xc and vc):
            return pd.Series(dtype=float), "company_dividend lacks code/ex_dividend/cash_dividend columns"
        d = self._query(f'SELECT "{xc}", "{vc}" FROM "{t}" WHERE UPPER("{cc}") = ?', (self._tk(ticker),))
        if d.empty:
            return pd.Series(dtype=float), f"no dividend rows for {self._tk(ticker)}"
        s = pd.Series(pd.to_numeric(d[vc], errors="coerce").values, index=pd.DatetimeIndex(idx_to_datetime(d[xc]).values))
        s = s[s.index.notna() & s.notna()]
        s = s.groupby(level=0).sum().sort_index()
        return s, f"IDX-API SQLite:{t} ({len(s)} ex-dates, latest {s.index.max():%Y-%m-%d})" if len(s) else "no valid ex-dividend dates"

    # ---------- foreign flow ----------
    def foreign_flow(self, ticker, days=20, max_age_days=10):
        """Returns (stock_bn, market_bn, info, problems). Rp billions, net over the last `days` sessions.
        Stock level: stock_summary.foreign_net is in SHARES, so it is converted at each day's VWAP
        (value / volume). Market level: foreign_trading buy_value - sell_value (already rupiah), falling
        back to the sum of the per-stock figures when that table is empty or stale."""
        today, tk = pd.Timestamp.now().normalize(), self._tk(ticker)
        need = int(np.ceil(0.8 * days))
        info, problems, fs, fm = [], [], None, None

        t = self._table("stock")
        if t:
            cc, dc = self._col(t, "code"), self._col(t, "date")
            fn, vo, va, cl = self._col(t, "foreign_net"), self._col(t, "volume"), self._col(t, "value"), self._col(t, "close")
            if all((cc, dc, fn, vo, va, cl)):
                d = self._query(f'SELECT "{dc}", "{fn}", "{vo}", "{va}", "{cl}" FROM "{t}" WHERE UPPER("{cc}") = ? ORDER BY "{dc}" DESC LIMIT ?',
                                (tk, int(days * 3)))
                if d.empty:
                    problems.append(f"stock flow: no rows for {tk}")
                else:
                    d["Date"] = idx_to_datetime(d[dc]).values
                    win, latest = self._recent(d.dropna(subset=["Date"]), days)
                    if (today - latest).days > max_age_days:
                        problems.append(f"stock flow STALE: last {tk} row is {latest:%Y-%m-%d} ({(today - latest).days} days old; limit {max_age_days})")
                    elif len(win) < need:
                        problems.append(f"stock flow: only {len(win)} of {days} sessions for {tk} (gaps in the sync)")
                    else:
                        vol, val, close = (pd.to_numeric(win[c], errors="coerce") for c in (vo, va, cl))
                        vwap = (val / vol.where(vol > 0)).fillna(close)
                        fs = float((pd.to_numeric(win[fn], errors="coerce") * vwap).sum() / 1e9)
                        info.append(f"{tk} stock flow Rp {fs:+,.0f} bn over {len(win)} sessions to {latest:%Y-%m-%d} (net foreign shares × daily VWAP)")
            else:
                problems.append("stock flow: stock_summary lacks foreign_net/volume/value/close")
        else:
            problems.append("stock flow: stock_summary table not found")

        t = self._table("foreign")
        if t:
            dc, bv, sv = self._col(t, "date"), self._col(t, "buy_value"), self._col(t, "sell_value")
            if all((dc, bv, sv)):
                d = self._query(f'SELECT "{dc}", "{bv}", "{sv}" FROM "{t}" ORDER BY "{dc}" DESC LIMIT ?', (int(days * 3),))
                if len(d):
                    d["Date"] = idx_to_datetime(d[dc]).values
                    win, latest = self._recent(d.dropna(subset=["Date"]), days)
                    if (today - latest).days > max_age_days:
                        problems.append(f"market flow STALE: foreign_trading ends {latest:%Y-%m-%d} ({(today - latest).days} days old; limit {max_age_days})")
                    elif len(win) < need:
                        problems.append(f"market flow: only {len(win)} of {days} sessions in foreign_trading (gaps in the sync)")
                    else:
                        net = pd.to_numeric(win[bv], errors="coerce") - pd.to_numeric(win[sv], errors="coerce")
                        fm = float(net.sum() / 1e9)
                        info.append(f"market flow Rp {fm:+,.0f} bn over {len(win)} sessions to {latest:%Y-%m-%d} (foreign_trading buy_value - sell_value)")
        if fm is None and self._table("stock"):   # fallback: add up every stock's net foreign
            t = self._table("stock")
            cc, dc = self._col(t, "code"), self._col(t, "date")
            fn, vo, va, cl = self._col(t, "foreign_net"), self._col(t, "volume"), self._col(t, "value"), self._col(t, "close")
            if all((cc, dc, fn, vo, va, cl)):
                cutoff = int((today - pd.Timedelta(days=int(days * 7 / 5) + 10)).timestamp() * 1000)
                d = self._query(f'SELECT "{dc}", "{fn}", "{vo}", "{va}", "{cl}" FROM "{t}" WHERE "{dc}" >= ?', (cutoff,))
                if len(d):
                    d["Date"] = idx_to_datetime(d[dc]).values
                    vol, val, close = (pd.to_numeric(d[c], errors="coerce") for c in (vo, va, cl))
                    d["rp"] = pd.to_numeric(d[fn], errors="coerce") * (val / vol.where(vol > 0)).fillna(close)
                    g = d.dropna(subset=["Date"]).groupby("Date").agg(rp=("rp", "sum"), n=("rp", "size")).reset_index()
                    win, latest = self._recent(g, days)
                    if len(win) >= need and (today - latest).days <= max_age_days and win["n"].median() >= 300:
                        fm = float(win["rp"].sum() / 1e9)
                        info.append(f"market flow Rp {fm:+,.0f} bn over {len(win)} sessions to {latest:%Y-%m-%d} (sum of {int(win['n'].median())} stocks, shares × VWAP)")
                    elif len(win):
                        problems.append(f"market flow fallback not used: {int(win['n'].median())} stocks/day (needs >= 300), {len(win)} sessions")
        return fs, fm, "; ".join(info), "; ".join(problems)

    # ---------- financial ratios ----------
    def financial_ratio(self, ticker, max_age_months=15):
        """Latest report-period ratios for one company. IDX stores ROE/ROA/NPM already in PERCENT (so no
        x100 guess), and the figures belong to the REPORT PERIOD — for Mar/Jun/Sep that is a cumulative
        year-to-date period and IDX-API does not document annualisation."""
        t = self._table("ratio")
        if not t:
            return {}, "financial_ratio table not found"
        cc, pc = self._col(t, "code", "ticker", "symbol"), self._col(t, "period")
        if not (cc and pc):
            return {}, "financial_ratio lacks code/period columns"
        d = self._query(f'SELECT * FROM "{t}" WHERE UPPER("{cc}") = ?', (self._tk(ticker),))
        if d.empty:
            return {}, f"no financial_ratio rows for {self._tk(ticker)}"
        d["__p"] = idx_to_datetime(d[pc]).values
        d = d.dropna(subset=["__p"]).sort_values("__p")
        if d.empty:
            return {}, "financial_ratio rows have no valid period"
        last, period = d.iloc[-1], d["__p"].iloc[-1]
        age_m = (pd.Timestamp.now().normalize() - period).days / 30.4
        if age_m > max_age_months:
            return {}, f"STALE: latest ratio period for {self._tk(ticker)} is {period:%Y-%m-%d} ({age_m:.0f} months old)"
        out = {}
        for key in ("roe", "roa", "pbv", "per", "eps"):
            c = self._col(t, key)
            if c:
                v = _idx_num(last[c])
                if v is not None:
                    out[key] = v
        ec = self._col(t, "eps")
        if ec and out.get("eps") is not None:   # year-on-year EPS growth on the SAME cumulative basis (same month, a year earlier)
            prev = d[(d["__p"] >= period - pd.Timedelta(days=380)) & (d["__p"] <= period - pd.Timedelta(days=350))]
            pe = _idx_num(prev[ec].iloc[-1]) if len(prev) else None
            if pe and pe > 0 and out["eps"] > 0:
                out["eps_growth"] = (out["eps"] / pe - 1) * 100
        span = "full-year" if period.month == 12 else f"{period.month}-month cumulative, annualisation not documented"
        out["_period"] = period
        out["_source"] = f"IDX-API SQLite:{t} ({self._tk(ticker)} period {period:%Y-%m-%d}, {span})"
        return out, out["_source"]


@st.cache_resource(ttl=300)
def get_idx_adapter(db_path):
    """Cached adapter. It holds only the path and the table/column catalogue — never a connection —
    so it is safe to share between Streamlit's per-rerun threads."""
    return IDXAPIAdapter(db_path)


if st.session_state.pop("idx_reload_requested", False):
    get_idx_adapter.clear()


def _idx_unusable(db_path):
    """The adapter if it can serve data, otherwise (None, the real reason from the path diagnosis)."""
    a = get_idx_adapter(db_path)
    return (a, "") if a.available else (None, a.diag["message"])


def idx_api_fundamentals(ticker, db_path):
    if not db_path: return {}, "disabled"
    a, why = _idx_unusable(db_path)
    return a.financial_ratio(ticker) if a else ({}, why)


def idx_api_dividends(ticker, db_path):
    if not db_path: return pd.Series(dtype=float), "disabled"
    a, why = _idx_unusable(db_path)
    return a.dividends(ticker) if a else (pd.Series(dtype=float), why)


def idx_api_foreign_flow(ticker, db_path, days=20, max_age_days=10):
    if not db_path: return None, None, "", "disabled"
    a, why = _idx_unusable(db_path)
    return a.foreign_flow(ticker, days, max_age_days) if a else (None, None, "", why)


def idx_api_ohlcv(ticker, db_path, start=None, end=None):
    if not db_path: return pd.DataFrame(), "disabled"
    a, why = _idx_unusable(db_path)
    return a.ohlcv(ticker, start, end) if a else (pd.DataFrame(), why)


def idx_api_ihsg(db_path, start=None, end=None, max_age_days=10):
    if not db_path: return pd.DataFrame(), "disabled"
    a, why = _idx_unusable(db_path)
    return a.index_ohlcv("COMPOSITE", start, end, max_age_days) if a else (pd.DataFrame(), why)


def infer_dividend_season(dividend_series, lookback_years=3):
    """From real historical ex-dates, infer the typical MONTH(S) this stock
    tends to pay dividends (e.g. Indonesian banks often pay once in Q1/Q2
    after the annual shareholders' meeting (AGM)). This is a PATTERN from real data, not a
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
    # (decision announcement date — day 2 of the 2-day BI meeting, when BI Rate is announced)
    ("2026-01-21", "BI Meeting Jan (Quarterly)"), ("2026-02-19", "BI Meeting Feb"),
    ("2026-03-17", "BI Meeting Mar"), ("2026-04-22", "BI Meeting Apr (Quarterly)"),
    ("2026-05-20", "BI Meeting May"), ("2026-06-18", "BI Meeting Jun"),
    ("2026-07-22", "BI Meeting Jul (Quarterly)"), ("2026-08-19", "BI Meeting Aug"),
    ("2026-09-23", "BI Meeting Sep"), ("2026-10-21", "BI Meeting Oct (Quarterly+Annual)"),
    ("2026-11-18", "BI Meeting Nov"), ("2026-12-16", "BI Meeting Dec"),
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
            st.sidebar.warning(f"Could not parse AGM/EGM date: '{line}' — use YYYY-MM-DD format.")
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


@st.cache_data(ttl=900, show_spinner=False)
def run_deep_forecast(ohlcv, horizon, band_sigma, n_estimators, n_splits, use_rf, use_xgb, use_sarimax, sarimax_order, sarimax_seasonal):
    """Cached wrapper around deep_forecast.run (the DataFrame and every setting are part of the cache key)."""
    return dfc.run(ohlcv, horizon, band_sigma=band_sigma, n_estimators=n_estimators, n_splits=n_splits, use_rf=use_rf, use_xgb=use_xgb,
                   use_sarimax=use_sarimax, sarimax_order=sarimax_order, sarimax_seasonal=sarimax_seasonal)


@st.cache_data(ttl=300)
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


@st.cache_data(ttl=300)
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
    """ROE / ROA / PBV / PER / EPS / EV-EBITDA from yfinance .info. Fields are
    frequently missing or stale for Indonesian tickers, and EV/EBITDA is generally
    not reported for banks — anything unavailable is returned as None."""
    out = {"roe": None, "roa": None, "pbv": None, "ev_ebitda": None,
           "pe": None, "eps_ttm": None, "eps_fwd": None, "eps_growth": None}
    try:
        info = yf.Ticker(symbol).info or {}
    except Exception:
        return out

    def num(k):
        v = info.get(k)
        return float(v) if isinstance(v, (int, float)) and np.isfinite(v) else None

    roe, roa, eg = num("returnOnEquity"), num("returnOnAssets"), num("earningsGrowth")
    out["roe"] = roe * 100 if roe is not None else None   # yfinance gives fractions
    out["roa"] = roa * 100 if roa is not None else None
    out["eps_growth"] = eg * 100 if eg is not None else None
    out["pbv"] = num("priceToBook")
    out["ev_ebitda"] = num("enterpriseToEbitda")
    out["pe"] = num("trailingPE") or num("forwardPE")
    out["eps_ttm"], out["eps_fwd"] = num("trailingEps"), num("forwardEps")
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
    now = pd.Timestamp.now().normalize()
    # ex-date <= today only: company_dividend lists ANNOUNCED dividends too, and one that has not gone ex yet is not trailing income
    ttm = float(d[(d.index >= now - pd.DateOffset(years=1)) & (d.index <= now)].sum())
    return ttm / float(last_price) * 100 if ttm > 0 else None


def justified_pbv(roe_pct, coe_pct, g_pct):
    """Gordon-growth justified P/B = (ROE - g) / (COE - g). Very sensitive to
    its inputs; returns None when COE - g is too small for the formula to be stable."""
    if roe_pct is None or coe_pct is None or (coe_pct - g_pct) <= 0.5:
        return None
    return max((roe_pct - g_pct) / (coe_pct - g_pct), 0.0)


# ---------- technical indicators (pure functions) ----------
def _f(x):
    """float or None (never NaN/inf)."""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None


def rsi_wilder(close, n=14):
    d = close.diff()
    up, dn = d.clip(lower=0), (-d).clip(lower=0)
    au = up.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    ad = dn.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    rsi = 100 - 100 / (1 + au / ad.replace(0, np.nan))
    return rsi.where(ad != 0, 100.0).where(au.notna())


def macd_lines(close, fast=12, slow=26, signal=9):
    macd = close.ewm(span=fast, adjust=False).mean() - close.ewm(span=slow, adjust=False).mean()
    sig = macd.ewm(span=signal, adjust=False).mean()
    return macd, sig, macd - sig


def bollinger_bands(close, n=20, k=2.0):
    mid, sd = close.rolling(n).mean(), close.rolling(n).std(ddof=0)
    return mid, mid + k * sd, mid - k * sd


def atr_wilder(high, low, close, n=14):
    pc = close.shift(1)
    tr = pd.concat([high - low, (high - pc).abs(), (low - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def technical_stats(ohlcv):
    """Latest RSI / MACD / SMA / Bollinger / ATR readings. Any field is None when
    the history is too short for it; the whole result is None below ~35 bars."""
    if ohlcv is None or len(ohlcv) < 35:
        return None
    c, h, l = ohlcv["Close"], ohlcv["High"], ohlcv["Low"]
    price = _f(c.iloc[-1])
    out = {"price": price, "rsi": _f(rsi_wilder(c).iloc[-1])}
    m, sg, hist = macd_lines(c)
    out.update({"macd": _f(m.iloc[-1]), "macd_signal": _f(sg.iloc[-1]), "macd_hist": _f(hist.iloc[-1])})
    for n in (20, 50, 200):
        out[f"sma{n}"] = _f(c.rolling(n).mean().iloc[-1]) if len(c) >= n else None
    mid, up, lo = bollinger_bands(c)
    width = (up - lo).iloc[-1]
    out["pctb"] = _f((c.iloc[-1] - lo.iloc[-1]) / width) if _f(width) and width > 0 else None
    bw = ((up - lo) / mid).dropna()
    out["bandwidth"] = _f(bw.iloc[-1] * 100) if len(bw) else None
    out["bw_pctile"] = _f((bw.iloc[-250:] <= bw.iloc[-1]).mean() * 100) if len(bw) >= 60 else None
    atr = _f(atr_wilder(h, l, c).iloc[-1])
    out["atr"], out["atr_pct"] = atr, (atr / price * 100 if (atr and price) else None)
    return out


# ---------- signal rules: each returns a float in [-1, +1] or None ----------
def rsi_signal(r):
    """Momentum with a contrarian dampener at the extremes."""
    if r is None:
        return None
    if r >= 70:
        return -0.5      # overbought: stretched
    if r >= 55:
        return 1.0       # healthy upside momentum
    if r > 45:
        return 0.0
    if r > 30:
        return -1.0      # downside momentum
    return 0.5           # oversold: bounce potential


def macd_signal_score(t):
    m, h = t.get("macd"), t.get("macd_hist")
    if m is None or h is None:
        return None
    if h > 0:
        return 1.0 if m > 0 else 0.5     # above signal line; below zero = still recovering
    return -1.0 if m < 0 else -0.5


def ma_signal(t):
    """Mean of: price vs SMA20/50/200 and SMA50 vs SMA200 (needs >= 2 components)."""
    p, comps = t.get("price"), []
    for k in ("sma20", "sma50", "sma200"):
        if t.get(k) is not None and p is not None:
            comps.append(1.0 if p > t[k] else -1.0)
    if t.get("sma50") is not None and t.get("sma200") is not None:
        comps.append(1.0 if t["sma50"] > t["sma200"] else -1.0)
    return float(np.mean(comps)) if len(comps) >= 2 else None


def bollinger_signal(pctb):
    """Inside the bands the side of the midline gives a weak trend read; outside
    the bands the move is treated as stretched, so the read reverses."""
    if pctb is None:
        return None
    if pctb > 1.0:
        return -0.5
    if pctb < 0.0:
        return 0.5
    return 0.5 if pctb >= 0.5 else -0.5


def trend_stats(close):
    s = close.dropna()
    if len(s) < 64:
        return None
    out = {"last": _f(s.iloc[-1]), "ret63": _f((s.iloc[-1] / s.iloc[-64] - 1) * 100),
           "ret126": _f((s.iloc[-1] / s.iloc[-127] - 1) * 100) if len(s) >= 127 else None}
    out["above50"] = bool(s.iloc[-1] > s.rolling(50).mean().iloc[-1]) if len(s) >= 50 else None
    out["above200"] = bool(s.iloc[-1] > s.rolling(200).mean().iloc[-1]) if len(s) >= 200 else None
    return out


def trend_signal(ts, ret_thr=3.0, invert=False):
    """Mean of: 63d return beyond +/-ret_thr, price vs SMA50, price vs SMA200.
    invert=True for series where a RISE is adverse (USD/IDR, the dollar index)."""
    if ts is None or ts.get("ret63") is None:
        return None
    comps = [1.0 if ts["ret63"] > ret_thr else (-1.0 if ts["ret63"] < -ret_thr else 0.0)]
    for k in ("above50", "above200"):
        if ts.get(k) is not None:
            comps.append(1.0 if ts[k] else -1.0)
    if len(comps) < 2:
        return None
    v = float(np.mean(comps))
    return -v if invert else v


def vix_signal(v):
    if v is None:
        return None
    return -1.0 if v >= 25 else (1.0 if v <= 16 else 0.0)


def flow_signal(value_bn, deadband):
    """Foreign flow: sign of the 20d net flow, ignoring flows inside +/-deadband. 0 = not provided."""
    if value_bn is None or value_bn == 0:
        return None
    if abs(value_bn) <= deadband:
        return 0.0
    return 1.0 if value_bn > 0 else -1.0


def sig_txt(v):
    return "⚪ n/a" if v is None else (f"🟢 {v:+.2f}" if v > 0.05 else (f"🔴 {v:+.2f}" if v < -0.05 else "🟡 0.00"))


def collect_engine_inputs(cfg, dividend_history, last_price):
    """Gather every raw input for the three layers. Missing -> None. Manual overrides win."""
    f = {"is_lq45": cfg["is_lq45"], "is_idx30": cfg["is_idx30"], "notes": [], "sources": []}

    def series(sym, days_back=420):
        return load_market_series(sym, days_back=days_back) if sym else pd.DataFrame()

    def trend(sym):
        d = series(sym)
        return trend_stats(d["Close"]) if len(d) else None

    # ---- layer 1: market regime ----
    ihsg = series(cfg["ihsg_symbol"])
    # JCI trend / volatility / volume always come from yfinance. IDX-API is additional information only (see the data-health panel).
    f["ihsg_trend"] = trend_stats(ihsg["Close"]) if len(ihsg) else None
    f["ihsg_vol"] = realized_vol_stats(ihsg["Close"]) if len(ihsg) else None
    f["ihsg_volume"] = volume_stats(ihsg) if len(ihsg) else None
    if not len(ihsg):
        f["notes"].append(f"No JCI data for '{cfg['ihsg_symbol']}' (JCI trend, volume & volatility skipped).")
    f["breadth"] = breadth_stats(load_basket_closes(tuple(cfg["basket"])))
    f["usdidr"], f["dxy"] = trend(cfg["usdidr_symbol"]), trend(cfg["dxy_symbol"])
    f["spx"], f["em"] = trend(cfg["spx_symbol"]), trend(cfg["em_symbol"])
    vx = series(cfg["vix_symbol"])
    f["vix"] = _f(vx["Close"].iloc[-1]) if len(vx) else None
    for key, label in (("usdidr", "USD/IDR"), ("dxy", "DXY"), ("spx", "S&P 500"), ("em", "EM equities")):
        if f[key] is None:
            f["notes"].append(f"{label} data unavailable (check the symbol in the sidebar).")

    stock = series(cfg["ticker"], 800)
    f["stock_trend"] = trend_stats(stock["Close"]) if len(stock) else None
    f["stock_volume"] = volume_stats(stock) if len(stock) else None
    idx_fs = idx_fm = None
    if cfg.get("idx_api_enabled") and cfg.get("idx_api_db"):
        try:
            idx_fs, idx_fm, idx_flow_info, idx_flow_problems = idx_api_foreign_flow(
                cfg["ticker"].split(".")[0], cfg["idx_api_db"], cfg.get("idx_api_days", 20), cfg.get("idx_api_max_age", 10))
            if idx_flow_info:
                f["sources"].append(f"Foreign flow: {idx_flow_info}")
            if idx_flow_problems:
                f["notes"].append(f"IDX-API foreign flow — {idx_flow_problems}")
        except Exception as e:
            f["notes"].append(f"IDX-API foreign flow unavailable: {type(e).__name__}: {e}")
    f["foreign_stock_bn"] = cfg["foreign_stock_bn"] or idx_fs
    f["foreign_mkt_bn"] = cfg["foreign_mkt_bn"] or idx_fm

    # ---- layer 2: technical ----
    f["tech"] = technical_stats(stock) if len(stock) else None
    if f["tech"] is None:
        f["notes"].append(f"Not enough price history for '{cfg['ticker']}' to compute RSI/MACD/MA/Bollinger/ATR.")

    def _rs(sym):
        b = series(sym)
        return relative_strength_pct(stock["Close"], b["Close"]) if (len(stock) and len(b)) else None

    f["rs_ihsg"], f["rs_lq45"], f["rs_idx30"] = _rs(cfg["ihsg_symbol"]), _rs(cfg["lq45_symbol"]), _rs(cfg["idx30_symbol"])
    f["has_idx30_symbol"] = bool(cfg["idx30_symbol"])

    # ---- layer 3: fundamental ----
    fund = load_fundamentals(cfg["ticker"])
    idx_fund, idx_fund_src = ({}, "disabled")
    if cfg.get("idx_api_enabled") and cfg.get("idx_api_db"):
        try:
            idx_fund, idx_fund_src = idx_api_fundamentals(cfg["ticker"].split(".")[0], cfg["idx_api_db"])
            if idx_fund:
                f["sources"].append(f"Fundamentals (IDX-API): {idx_fund_src}")
            else:
                f["notes"].append(f"IDX-API financial ratios not used: {idx_fund_src}")
        except Exception as e:
            f["notes"].append(f"IDX-API financial ratios unavailable: {type(e).__name__}: {e}")
    idx_first = str(cfg.get("idx_api_ratio_priority", "")).startswith("IDX-API first")

    def auto_fund(key, yf_key=None):
        """Ratio from the preferred source, the other one filling the gap. yfinance calls PER 'pe'."""
        a, b = idx_fund.get(key), fund.get(yf_key or key)
        return (a if a is not None else b) if idx_first else (b if b is not None else a)
    f["roe"] = cfg["roe_manual"] if cfg["roe_manual"] > 0 else auto_fund("roe")
    f["roa"] = cfg["roa_manual"] if cfg["roa_manual"] > 0 else auto_fund("roa")
    f["pbv"] = cfg["pbv_manual"] if cfg["pbv_manual"] > 0 else auto_fund("pbv")
    f["pe"] = cfg["per_manual"] if cfg["per_manual"] > 0 else auto_fund("per", "pe")
    f["ev_ebitda"] = cfg["evebitda_manual"] if cfg["evebitda_manual"] > 0 else auto_fund("ev_ebitda")
    if cfg["eps_growth_manual"] != 0:
        f["eps_growth"], f["eps_src"] = cfg["eps_growth_manual"], "manual"
    elif idx_first and idx_fund.get("eps_growth") is not None:
        f["eps_growth"], f["eps_src"] = idx_fund["eps_growth"], "IDX-API year-on-year EPS"
    elif fund["eps_growth"] is not None:
        f["eps_growth"], f["eps_src"] = fund["eps_growth"], "yfinance earningsGrowth"
    elif fund["eps_ttm"] and fund["eps_fwd"] and fund["eps_ttm"] > 0:
        f["eps_growth"], f["eps_src"] = (fund["eps_fwd"] / fund["eps_ttm"] - 1) * 100, "forward vs trailing EPS"
    elif idx_fund.get("eps_growth") is not None:
        f["eps_growth"], f["eps_src"] = idx_fund["eps_growth"], "IDX-API year-on-year EPS"
    else:
        f["eps_growth"], f["eps_src"] = None, "unavailable"
    div = dividend_history if (dividend_history is not None and len(dividend_history) > 0) else load_dividend_history(cfg["ticker"])
    f["div_yield"] = trailing_dividend_yield_pct(div, last_price)

    # ---- rates (part of the regime layer) ----
    f["bi_rate"], f["bi_expect_bps"] = cfg["bi_rate"], cfg["bi_expect_bps"]
    ust = None
    tnx = load_market_series("^TNX", days_back=420)
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


def technical_evidence(pattern_result, price_diff, support_diff):
    """Chart-pattern signals taken from what the app already detects. Each is in
    [-1, +1] (continuation patterns = +/-1, 'reversal watch' = +/-0.5) or None.
    Returns (rows, score or None, n_scored)."""
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


LAYERS = ("Regime", "Technical", "Fundamental")
LAYER_LABELS = {"Regime": "1 · Market regime", "Technical": "2 · Technical", "Fundamental": "3 · Fundamental"}
# (minimum scored signals, minimum distinct groups) before a layer is allowed to influence the forecast
LAYER_MIN = {"Regime": (4, 2), "Technical": (3, 2), "Fundamental": (3, 2)}


def build_forecast_scorecard(f, pattern_rows):
    """Rule-based scorecard organised as the three pipeline layers. Each row ->
    a signal in [-1, +1] (supportive/adverse) or None (unavailable / info only).
    Rows are grouped inside each layer so correlated factors (e.g. the five rates
    rows) are averaged before the layer score is formed."""
    rows = []

    def add(layer, group, factor, latest, signal, rule, info=False):
        rows.append({"Layer": layer, "Group": group, "Factor": factor, "Latest": latest,
                     "Signal": signal, "Rule": rule, "Info": info})

    def x(v, nd=2, suf=""):
        return "n/a" if v is None else f"{v:.{nd}f}{suf}"

    def trend_txt(ts):
        if ts is None:
            return "n/a (data unavailable)"
        yn = lambda b: "n/a" if b is None else ("yes" if b else "no")
        return f"63d {ts['ret63']:+.1f}%; above SMA50: {yn(ts['above50'])}; above SMA200: {yn(ts['above200'])}"

    TREND_RULE = "mean of: 63d return beyond +/-{t}%, price vs SMA50, price vs SMA200{inv}"
    R = "Regime"

    # ======== LAYER 1 · MARKET REGIME ========
    g = "Domestic market"
    add(R, g, "JCI trend", trend_txt(f.get("ihsg_trend")), trend_signal(f.get("ihsg_trend"), 3.0), TREND_RULE.format(t=3, inv=""))
    br = f.get("breadth")
    if br is None:
        add(R, g, "Market breadth (advance/decline)", "n/a (basket data unavailable)", None, ">60% above 20DMA = +1, <40% = -1")
    else:
        p = br["pct_above_ma20"]
        add(R, g, "Market breadth (advance/decline)",
            f"today {br['adv']} adv / {br['dec']} dec; 10d net {br['net10']:+d}; {p:.0f}% of {br['n']} names above 20DMA",
            1.0 if p > 60 else (-1.0 if p < 40 else 0.0), ">60% above 20DMA = +1, <40% = -1 (proxy basket, not the full exchange)")
    vs = f.get("ihsg_volume")
    if vs is None:
        add(R, g, "JCI volume", "n/a", None, "5d/20d volume > 1.15x: direction of the 5d move decides +1/-1")
    else:
        sig = (1.0 if vs["ret5"] > 0 else -1.0) if (vs["ratio"] > 1.15 and abs(vs["ret5"]) >= 0.3) else 0.0
        add(R, g, "JCI volume", f"5d/20d volume {vs['ratio']:.2f}x; JCI 5d {vs['ret5']:+.1f}%", sig,
            "5d/20d > 1.15x: volume expansion on an up-move = +1, on a down-move = -1, else 0")
    vo = f.get("ihsg_vol")
    if vo is None or vo["ratio"] is None:
        add(R, g, "JCI volatility", "n/a", None, "20d realized vol vs 1y median: >1.3x = -1, <0.8x = +1")
    else:
        add(R, g, "JCI volatility", f"20d realized {vo['vol']:.1f}% ann. (1y median {vo['median']:.1f}%, ratio {vo['ratio']:.2f}x)",
            -1.0 if vo["ratio"] > 1.3 else (1.0 if vo["ratio"] < 0.8 else 0.0), "ratio >1.3x = -1 (stress), <0.8x = +1 (calm)")

    g = "Rates & policy"
    add(R, g, "BI Rate (level)", f"{f.get('bi_rate'):.2f}% (manual input)", None,
        "Level shown for context; the scored signal is the expected direction below.", info=True)
    bps = f.get("bi_expect_bps")
    add(R, g, "BI-Rate expectation", f"{bps:+.0f} bps over ~6 months (manual input)",
        None if bps is None else (1.0 if bps <= -25 else (-1.0 if bps >= 25 else 0.0)), "<= -25bps = +1, >= +25bps = -1")
    fed, fchg = f.get("fed"), f.get("fed_chg6m")
    add(R, g, "Fed Funds Rate", x(fed, 2, "%") + (f" (6M change {fchg:+.2f}pp)" if fchg is not None else ""),
        None if fchg is None else (-1.0 if fchg >= 0.25 else (1.0 if fchg <= -0.25 else 0.0)),
        "6M change >= +0.25pp = -1, <= -0.25pp = +1 (needs the automatic FRED series to know the change)")
    ust, uchg = f.get("ust10"), f.get("ust10_chg3m")
    add(R, g, "US Treasury 10Y yield", x(ust, 2, "%") + (f" (3M change {uchg:+.2f}pp)" if uchg is not None else ""),
        None if uchg is None else (-1.0 if uchg >= 0.30 else (1.0 if uchg <= -0.30 else 0.0)), "3M change >= +0.30pp = -1, <= -0.30pp = +1")
    idy, ichg = f.get("id10y"), f.get("id10y_chg3m")
    add(R, g, "Indonesia 10Y yield", f"{x(idy, 2, '%')} [{f.get('id10y_src')}]" + (f" (3M change {ichg:+.2f}pp)" if ichg is not None else ""),
        None if ichg is None else (-1.0 if ichg >= 0.30 else (1.0 if ichg <= -0.30 else 0.0)), "3M change >= +0.30pp = -1, <= -0.30pp = +1")
    sp = f.get("spread")
    add(R, g, "Yield spread (ID10Y - UST10Y)", x(sp, 2, "pp"),
        None if sp is None else (-1.0 if sp < 2.0 else (1.0 if sp > 3.0 else 0.0)),
        "< 2.0pp = -1 (thin compensation for EM risk, outflow risk), > 3.0pp = +1")

    g = "FX & dollar"
    add(R, g, "USD/IDR", trend_txt(f.get("usdidr")), trend_signal(f.get("usdidr"), 2.0, invert=True),
        TREND_RULE.format(t=2, inv="; INVERTED - a weaker rupiah is adverse"))
    add(R, g, "Dollar index (DXY)", trend_txt(f.get("dxy")), trend_signal(f.get("dxy"), 2.0, invert=True),
        TREND_RULE.format(t=2, inv="; INVERTED - a stronger dollar is adverse for EM"))

    g = "Global risk"
    add(R, g, "Global equities (S&P 500)", trend_txt(f.get("spx")), trend_signal(f.get("spx"), 3.0), TREND_RULE.format(t=3, inv=""))
    vix = f.get("vix")
    add(R, g, "VIX", x(vix, 1), vix_signal(vix), ">= 25 = -1 (fear), <= 16 = +1 (calm), else 0")
    add(R, g, "EM equities (EEM)", trend_txt(f.get("em")), trend_signal(f.get("em"), 3.0), TREND_RULE.format(t=3, inv=""))

    g = "Price, volume & foreign flow"
    st_ = f.get("stock_trend")
    r126 = None if st_ is None else st_.get("ret126")
    add(R, g, "Stock price momentum (126d return)", x(r126, 1, "%"),
        None if r126 is None else (1.0 if r126 >= 10 else (-1.0 if r126 <= -10 else 0.0)), ">= +10% = +1, <= -10% = -1")
    sv = f.get("stock_volume")
    if sv is None:
        add(R, g, "Stock volume", "n/a", None, "5d/20d volume > 1.15x: direction of the 5d move decides +1/-1")
    else:
        sig = (1.0 if sv["ret5"] > 0 else -1.0) if (sv["ratio"] > 1.15 and abs(sv["ret5"]) >= 0.3) else 0.0
        add(R, g, "Stock volume", f"5d/20d volume {sv['ratio']:.2f}x; stock 5d {sv['ret5']:+.1f}%", sig,
            "5d/20d > 1.15x: volume expansion on an up-move = +1, on a down-move = -1, else 0")
    fs, fm = f.get("foreign_stock_bn"), f.get("foreign_mkt_bn")
    add(R, g, "Foreign flow — stock (20d)", "not provided (manual input)" if fs is None else f"{fs:+,.0f} Rp bn",
        flow_signal(fs, 50), "net buy > +50bn = +1, net sell < -50bn = -1, inside the +/-50bn dead-band = 0 (manual)")
    add(R, g, "Foreign flow — market (20d)", "not provided (manual input)" if fm is None else f"{fm:+,.0f} Rp bn",
        flow_signal(fm, 500), "net buy > +500bn = +1, net sell < -500bn = -1, inside the +/-500bn dead-band = 0 (manual)")

    # ======== LAYER 2 · TECHNICAL ========
    T = "Technical"
    t = f.get("tech") or {}
    g = "Momentum & trend"
    add(T, g, "RSI (14)", x(t.get("rsi"), 1), rsi_signal(t.get("rsi")),
        ">=70 = -0.5 (stretched), 55-70 = +1, 45-55 = 0, 30-45 = -1, <=30 = +0.5 (oversold bounce)")
    add(T, g, "MACD (12,26,9)", ("n/a" if t.get("macd") is None else
        f"MACD {t['macd']:+.1f} vs signal {t['macd_signal']:+.1f} (hist {t['macd_hist']:+.1f})"), macd_signal_score(t),
        "hist>0 & MACD>0 = +1, hist>0 & MACD<0 = +0.5, hist<0 & MACD<0 = -1, hist<0 & MACD>0 = -0.5")
    mtxt = "n/a" if not t else " | ".join(
        f"SMA{n} {t[f'sma{n}']:,.0f}" if t.get(f"sma{n}") is not None else f"SMA{n} n/a" for n in (20, 50, 200))
    add(T, g, "Moving averages", mtxt, ma_signal(t) if t else None,
        "mean of: price vs SMA20/50/200 and SMA50 vs SMA200 (needs >= 2 averages)")
    g = "Volatility & bands"
    add(T, g, "Bollinger %B (20,2)", x(t.get("pctb"), 2), bollinger_signal(t.get("pctb")),
        "inside bands: above midline = +0.5, below = -0.5; outside the bands the move is stretched, so the read reverses (+/-0.5)")
    add(T, g, "Bollinger bandwidth", "n/a" if t.get("bandwidth") is None else
        f"{t['bandwidth']:.1f}% of price" + ("" if t.get("bw_pctile") is None else f" ({t['bw_pctile']:.0f}th percentile of the last year)"),
        None, "Context: a low percentile = a volatility squeeze, which says nothing about direction.", info=True)
    add(T, g, "ATR (14)", "n/a" if t.get("atr") is None else f"Rp {t['atr']:,.0f} ({t['atr_pct']:.1f}% of price)",
        None, "Context: typical daily range; used to sanity-check the forecast range.", info=True)
    g = "Relative strength"
    for key, label, always in (("rs_ihsg", "RS vs JCI (63d)", True), ("rs_lq45", "RS vs LQ45 (63d)", True),
                               ("rs_idx30", "RS vs IDX30 (63d)", f.get("has_idx30_symbol"))):
        rs = f.get(key)
        if rs is None:
            if always:
                add(T, g, label, "n/a (index data unavailable)", None, "excess return > +3pp = +1, < -3pp = -1")
            continue
        ex, s_, b_ = rs
        add(T, g, label, f"stock {s_:+.1f}% vs index {b_:+.1f}% (excess {ex:+.1f}pp)",
            1.0 if ex > 3 else (-1.0 if ex < -3 else 0.0), "excess return > +3pp = +1, < -3pp = -1")
    for r in pattern_rows:
        add(T, "Chart patterns", r["Factor"], r["Latest"], r["Signal"], r["Rule"])

    # ======== LAYER 3 · FUNDAMENTAL ========
    F = "Fundamental"
    g = "Earnings & quality"
    eg = f.get("eps_growth")
    add(F, g, "EPS growth", ("n/a" if eg is None else f"{eg:+.1f}% [{f.get('eps_src')}]"),
        None if eg is None else (1.0 if eg >= 5 else (-1.0 if eg <= -5 else 0.0)), ">= +5% = +1, <= -5% = -1")
    roe, roa = f.get("roe"), f.get("roa")
    add(F, g, "ROE", x(roe, 1, "%"), None if roe is None else (1.0 if roe >= 15 else (-1.0 if roe < 10 else 0.0)), ">=15% = +1, <10% = -1")
    add(F, g, "ROA", x(roa, 2, "%"), None if roa is None else (1.0 if roa >= 2 else (-1.0 if roa < 1 else 0.0)),
        ">=2% = +1, <1% = -1 (generic thresholds; bank ROA norms differ from other sectors)")

    g = "Valuation"
    pbv, jp = f.get("pbv"), f.get("justified_pbv")
    if pbv is None:
        add(F, g, "PBV vs justified PBV", "n/a", None, "actual/justified <0.8 = +1, >1.2 = -1")
    elif jp is None or jp <= 0:
        add(F, g, "PBV vs justified PBV", f"PBV {pbv:.2f}x; justified PBV n/a (needs ROE and Indonesia 10Y)", None, "actual/justified <0.8 = +1, >1.2 = -1")
    else:
        ratio = pbv / jp
        add(F, g, "PBV vs justified PBV",
            f"PBV {pbv:.2f}x vs justified {jp:.2f}x (ROE {x(f.get('roe'), 1, '%')}, COE {x(f.get('coe'), 1, '%')}, g {f.get('g'):.1f}%) -> {ratio:.2f}",
            1.0 if ratio < 0.8 else (-1.0 if ratio > 1.2 else 0.0),
            "actual/justified <0.8 = +1, >1.2 = -1. Justified P/B = (ROE-g)/(COE-g); highly assumption-sensitive")
    pe = f.get("pe")
    if pe is None or pe <= 0:
        add(F, g, "PER (earnings yield vs ID10Y)", "n/a (no positive PER)", None, "earnings yield minus Indonesia 10Y: >= +3pp = +1, <= 0 = -1")
    elif idy is None:
        add(F, g, "PER (earnings yield vs ID10Y)", f"PER {pe:.1f}x (earnings yield {100 / pe:.1f}%); Indonesia 10Y unavailable", None,
            "earnings yield minus Indonesia 10Y: >= +3pp = +1, <= 0 = -1")
    else:
        ey_gap = 100.0 / pe - idy
        add(F, g, "PER (earnings yield vs ID10Y)", f"PER {pe:.1f}x -> earnings yield {100 / pe:.1f}% vs ID10Y {idy:.2f}% (gap {ey_gap:+.1f}pp)",
            1.0 if ey_gap >= 3 else (-1.0 if ey_gap <= 0 else 0.0), "earnings yield minus Indonesia 10Y: >= +3pp = +1, <= 0 = -1")
    ev = f.get("ev_ebitda")
    if ev is None:
        add(F, g, "EV/EBITDA", "n/a (not reported / not meaningful for banks)", None, "<6x = +1, >12x = -1")
    else:
        add(F, g, "EV/EBITDA", f"{ev:.1f}x", 1.0 if ev < 6 else (-1.0 if ev > 12 else 0.0), "<6x = +1, >12x = -1 (generic thresholds)")

    g = "Income"
    dy = f.get("div_yield")
    if dy is None:
        add(F, g, "Dividend yield (TTM)", "n/a (no dividend in the last 12 months / data unavailable)", None, "yield minus Indonesia 10Y: >=0 = +1, < -3pp = -1")
    elif idy is None:
        add(F, g, "Dividend yield (TTM)", f"{dy:.1f}% (Indonesia 10Y unavailable for comparison)", None, "yield minus Indonesia 10Y: >=0 = +1, < -3pp = -1")
    else:
        d_ = dy - idy
        add(F, g, "Dividend yield (TTM)", f"{dy:.1f}% vs Indonesia 10Y {idy:.2f}% (gap {d_:+.1f}pp)",
            1.0 if d_ >= 0 else (-1.0 if d_ < -3 else 0.0), "gap >= 0 = +1, < -3pp = -1")

    add("Info", "Info", "LQ45 / IDX30 membership",
        f"LQ45: {'yes' if f.get('is_lq45') else 'no'} | IDX30: {'yes' if f.get('is_idx30') else 'no'} (manual — verify on idx.co.id)",
        None, "Context only. Membership is a constant, so scoring it would add a permanent bias; "
              "index-fund flow relevance is captured by relative strength.", info=True)
    return rows


def layer_summary(rows):
    """Per layer: score = mean of the GROUP means (so correlated rows in one group
    count once), plus how many signals / groups back it."""
    out = {}
    for L in LAYERS:
        groups, n_rows = {}, 0
        for r in rows:
            if r["Layer"] != L:
                continue
            n_rows += 0 if r["Info"] else 1
            if r["Signal"] is not None:
                groups.setdefault(r["Group"], []).append(r["Signal"])
        gm = {g_: float(np.mean(v)) for g_, v in groups.items()}
        out[L] = {"score": float(np.mean(list(gm.values()))) if gm else None,
                  "n": sum(len(v) for v in groups.values()), "n_groups": len(gm), "groups": gm, "n_rows": n_rows}
    return out


def layer_counts(stats, layer):
    mn_n, mn_g = LAYER_MIN[layer]
    s_ = stats.get(layer) or {}
    return s_.get("score") is not None and s_.get("n", 0) >= mn_n and s_.get("n_groups", 0) >= mn_g


def engine_evidence(layer_stats, weights):
    """Weighted mean of the layer scores, using only layers with enough independent
    signals (LAYER_MIN) and a positive weight; weights are renormalised over the layers used.
    Returns (evidence in [-1, 1], {layer: effective weight})."""
    used = {L: weights.get(L, 0.0) for L in LAYERS if layer_counts(layer_stats, L) and weights.get(L, 0.0) > 0}
    if not used:
        return 0.0, {}
    tot = sum(used.values())
    eff = {L: w / tot for L, w in used.items()}
    return float(sum(eff[L] * layer_stats[L]["score"] for L in eff)), eff


def regime_label(score):
    if score is None:
        return "Unknown"
    return "Risk-on" if score > 0.25 else ("Risk-off" if score < -0.25 else "Mixed / neutral")


def scenario_probabilities(log_returns, last_price, horizon_days, evidence, strength,
                           band_sigma=0.5, n_sim=5000, seed=42, tilt_sigma_max=0.5):
    """Bullish / Neutral / Bearish probabilities for the price at the END of the horizon.

    1. Bootstrap `horizon_days` daily log-returns from the stock's own history, after
       DEMEANING them: the base case carries no directional view, so a past downtrend
       is not silently projected forward (see the regime-dependence issue we hit earlier).
    2. Classification on the horizon return R: Bullish if R > +band, Bearish if R < -band,
       Neutral otherwise, where band = band_sigma x (std of the simulated horizon return).
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
    Rt = R + shift
    price_end = last_price * np.exp(Rt)
    p10, p50, p90 = np.percentile(price_end, [10, 50, 90])

    def cond(mask):
        """Target / range among the simulated paths that END in this state."""
        if int(mask.sum()) < 20:
            return None
        x_ = price_end[mask]
        q25, q50, q75 = np.percentile(x_, [25, 50, 75])
        return {"median": float(q50), "p25": float(q25), "p75": float(q75), "n": int(mask.sum())}

    bull_m, bear_m = Rt > band, Rt < -band
    return {"p_bull": p_bull, "p_side": p_side, "p_bear": p_bear, "base": split(R),
            "sigma_h": sigma_h, "band": band, "shift": shift,
            "bull_price": last_price * float(np.exp(band)), "bear_price": last_price * float(np.exp(-band)),
            "p10": float(p10), "p50": float(p50), "p90": float(p90), "mean_price": float(price_end.mean()),
            "cond": {"bull": cond(bull_m), "side": cond(~(bull_m | bear_m)), "bear": cond(bear_m)},
            "n_returns": int(len(r)), "horizon_days": horizon_days, "n_sim": n_sim, "price_end": price_end}


try:
    df = load_stock_data(ticker, start_date, end_date, interval)
    price_source_label = "yfinance"   # prices, trends, indicators and Fibonacci always come from yfinance

    if df.empty:
        st.error(f"Stock data for {ticker} not found.")
        st.stop()
    df.index.name = "Date"   # the pattern detectors call reset_index() and then read cycle["Date"], whatever the data source named its index
    if show_logo:
        _roots = idx_asset_roots(engine_cfg.get("idx_api_db", ""))
        _logo = idx_find_logo(clean_ticker, _roots, logo_dir_override)
        _sl = idx_find_stocklist(_roots)
        _sl_info = idx_parse_stocklist(str(_sl), _sl.stat().st_mtime).get(clean_ticker.upper()) if _sl else None
        _full_name = None
        if engine_cfg.get("idx_api_enabled") and engine_cfg.get("idx_api_db"):
            try:
                _full_name = get_idx_adapter(engine_cfg["idx_api_db"]).company_name(clean_ticker)   # full name; STOCKLIST.md cuts at 30 characters
            except Exception:
                _full_name = None
        _name = _full_name or ((_sl_info["name"] + ("…" if _sl_info["name_truncated"] else "")) if _sl_info else None)
        _uri = idx_logo_data_uri(str(_logo), _logo.stat().st_mtime) if _logo else None
        with _header_slot.container():
            if _uri:
                try:
                    _c1, _c2 = st.columns([1, 12], vertical_alignment="center")
                except TypeError:   # Streamlit older than 1.36 has no vertical_alignment
                    _c1, _c2 = st.columns([1, 12])
                _c1.markdown(idx_logo_html(_uri, clean_ticker), unsafe_allow_html=True)
                _c2.title(f"{clean_ticker} Macro Pattern, Fibonacci & Cycle Projection · v{APP_VERSION}")
            else:   # no logo file for this ticker: the plain title, no empty frame
                st.title(f"📈 {clean_ticker} Macro Pattern, Fibonacci & Cycle Projection · v{APP_VERSION}")
            if _name:
                st.caption(f"**{_name}**" + (f" · listed {_sl_info['listed']}" if _sl_info else ""))
    _last_bar = df.index.max()
    st.caption(f"📡 Prices & trend: **yfinance** · last bar **{_last_bar:%a %d %b %Y}** · close Rp {float(df['Close'].iloc[-1]):,.0f}"
               + (f" · IDX-API supplies only foreign flow, dividends and fundamental ratios" if engine_cfg.get("idx_api_enabled") else ""))
    if interval == "1d" and bar_is_forming(_last_bar):
        st.caption("⏳ The last bar is dated today and the IDX session is still open (09:00–15:50 Jakarta time, Mon–Fri), so it is an INTRADAY bar "
                   "that will keep changing; RSI/MACD and the forecast anchor move with it. Click “Refresh market data” to re-fetch.")

    # --- Corporate & Macro Event Calendar data ---
    dividend_history = pd.Series(dtype=float)
    dividend_source_actual = dividend_source
    if show_dividend_cal:
        if dividend_source == "Manual input":
            dividend_history = parse_dividend_manual_input(dividend_manual_input)
            dividend_source_actual = "Manual input"
        elif engine_cfg.get("idx_api_enabled") and engine_cfg.get("idx_api_db"):
            idx_div, idx_div_src = idx_api_dividends(clean_ticker, engine_cfg["idx_api_db"])
            if len(idx_div):
                dividend_history = idx_div
                dividend_source_actual = idx_div_src
            else:
                dividend_history = load_dividend_history(ticker)
                dividend_source_actual = "yfinance fallback"
        else:
            dividend_history = load_dividend_history(ticker)
            dividend_source_actual = "yfinance (automatic)"
    dividend_months = infer_dividend_season(dividend_history) if show_dividend_cal else []
    rups_dates = parse_rups_input(rups_dates_input) if show_rups and rups_dates_input else []
    if engine_cfg.get("idx_api_enabled") and engine_cfg.get("idx_api_db"):
        idx_adapter = get_idx_adapter(engine_cfg["idx_api_db"])
        with st.expander("🇮🇩 IDX-API Data Health · v1.4.6", expanded=not idx_adapter.available):
            if idx_adapter.available:
                st.success(idx_adapter.diag["message"])
                _today, _lim = pd.Timestamp.now().normalize(), engine_cfg.get("idx_api_max_age", 10)
                _inv = []
                for r in idx_adapter.inventory():
                    age = None if r["latest"] is None else (_today - r["latest"]).days
                    if r["rows"] is None:
                        verdict = "table missing"
                    elif r["latest"] is None:
                        verdict = "empty"
                    elif r["periodic"]:
                        verdict = f"periodic · latest period {r['latest']:%Y-%m-%d}"
                    else:
                        verdict = "fresh" if age <= _lim else f"STALE ({age} days old)"
                    _inv.append({"Table": r["table"], "Rows": r["rows"], "Latest": "—" if r["latest"] is None else f"{r['latest']:%Y-%m-%d}", "Status": verdict})
                st.dataframe(pd.DataFrame(_inv), use_container_width=True, hide_index=True)
                if interval == "1d":
                    _xc = []
                    for _lab, _yf, _idx in (
                            (clean_ticker, df, idx_adapter.ohlcv(clean_ticker)[0]),
                            ("JCI", load_market_series(engine_cfg["ihsg_symbol"], days_back=60),
                             idx_adapter.index_ohlcv("COMPOSITE", max_age_days=None)[0])):
                        if len(_yf) == 0 or len(_idx) == 0:
                            _xc.append({"Series": _lab, "yfinance last bar": "—" if len(_yf) == 0 else f"{_yf.index.max():%Y-%m-%d}",
                                        "IDX-API last bar": "—" if len(_idx) == 0 else f"{_idx.index.max():%Y-%m-%d}", "Close on latest common day": "n/a", "Verdict": "no data to compare"})
                            continue
                        _common = _yf.index.intersection(_idx.index)
                        _gap = (_idx.index.max() - _yf.index.max()).days
                        if len(_common):
                            _d = _common.max(); _a, _b = float(_yf.loc[_d, "Close"]), float(_idx.loc[_d, "Close"])
                            _diff = (_b / _a - 1) * 100
                            _close_txt = f"{_d:%Y-%m-%d}: yfinance {_a:,.2f} · IDX-API {_b:,.2f} ({_diff:+.2f}%)"
                            _verdict = "match" if abs(_diff) <= 0.5 else "DIFFERENT — check splits / the IDX-API sync"
                        else:
                            _close_txt, _verdict = "no common date", "no common date"
                        if _gap > 0:
                            _verdict += f"; IDX-API is {_gap} day(s) NEWER (yfinance may lag)"
                        elif _gap < -3:
                            _verdict += f"; IDX-API is {-_gap} days behind (needs a sync, which only matters for the additional data)"
                        _xc.append({"Series": _lab, "yfinance last bar": f"{_yf.index.max():%Y-%m-%d}", "IDX-API last bar": f"{_idx.index.max():%Y-%m-%d}",
                                    "Close on latest common day": _close_txt, "Verdict": _verdict})
                    st.markdown("**Cross-check, yfinance vs IDX-API (information only: the forecast uses yfinance)**")
                    st.dataframe(pd.DataFrame(_xc), use_container_width=True, hide_index=True)
                _ls = IDXAutoSync(engine_cfg["idx_api_db"], _idx_sync_registry()).status()
                if _ls["state"] != "idle":
                    st.caption(f"Last sync attempt: {_ls.get('started_at', '?')[:16].replace('T', ' ')} ({_ls.get('trigger')}, {_ls.get('days')} days) → {_ls['state'].replace('_', ' ')}.")
                st.caption("Foreign-flow data older than the stale limit set in the sidebar is ignored by the engine. "
                           "Update the database with “⬇️ Update database now” in the sidebar, or run `deno run -A Cron.ts` in the IDX-API folder.")
                for note in idx_adapter.notes[-5:]:
                    st.warning(note)
            else:
                st.error(idx_adapter.diag["message"])
                if idx_adapter.diag["hint"]:
                    st.info(idx_adapter.diag["hint"])
                st.caption("Meanwhile the app falls back to yfinance and your manual inputs.")

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
        r2_val = None  # no real fit -> do not pretend to have an R²

    trendline_y = res_slope * np.arange(len(df)) + res_intercept
    current_trendline_val = float(res_slope * (len(df) - 1) + res_intercept)

    # ============================================================
    # SUPPORT TRENDLINE (regression on swing LOWS, same period as resistance)
    # Methodology identical to resistance, but fitted on LOW pivot points
    # after the ATH -- not a flat Fibonacci level. Together the two lines form
    # a macro channel (parallel) or a wedge/triangle (converging).
    # ============================================================
    low = df["Low"].to_numpy(dtype=float)
    low_idx, _ = find_peaks(-low, distance=int(pivot_distance))  # trough = peak of -low
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

    # Channel classification: slope ratio close to 1 -> parallel (a true channel)
    if res_slope != 0 and not res_fallback and not sup_fallback:
        slope_ratio = sup_slope / res_slope
        is_parallel_channel = 0.7 <= slope_ratio <= 1.3
    else:
        slope_ratio = None
        is_parallel_channel = None
    channel_width_now = current_trendline_val - current_support_val

    # BREAKOUT / RETEST STATUS (relative to resistance)
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
    # FORECAST ENGINE
    # Market regime -> Technical -> Fundamental -> evidence -> probabilities -> target / range
    # ============================================================
    engine_rows, layer_stats, engine_inputs = [], {}, {"notes": []}
    pattern_rows, _, _ = technical_evidence(pattern_result, price_diff, support_diff)
    if show_engine:
        try:
            engine_inputs = collect_engine_inputs(engine_cfg, dividend_history, last_price)
            engine_rows = build_forecast_scorecard(engine_inputs, pattern_rows)
            layer_stats = layer_summary(engine_rows)
        except Exception as e:  # never let an optional overlay take the whole app down
            engine_inputs = {"notes": [f"Forecast engine failed and was skipped: {type(e).__name__}: {e}"]}
            engine_rows, layer_stats = [], {}
    evidence, eff_weights = engine_evidence(layer_stats, engine_cfg["layer_weights"]) if layer_stats else (0.0, {})
    regime_score = (layer_stats.get("Regime") or {}).get("score")

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
                prob = scenario_probabilities(lr, last_price, forecast_weeks * 5, evidence, tilt_strength, prob_band_sigma)
                if prob is None:
                    prob_note = "Not enough usable daily returns (need >= 120) to estimate the probabilities."
        except Exception as e:
            prob, prob_note = None, f"Probability model failed and was skipped: {type(e).__name__}: {e}"
    pB, pS, pA = ((prob["p_bull"], prob["p_side"], prob["p_bear"]) if prob is not None else (None, None, None))

    # ============================================================
    # DEEP FORECAST (v1.5.0): Random Forest / XGBoost / SARIMAX on ~3 years of DAILY bars, same horizon as the engine
    # ============================================================
    deep = None
    if show_deep and dfc is not None and (ml_use_rf or ml_use_xgb or ml_use_sarimax):
        try:
            ml_ohlcv = load_market_series(ticker, days_back=1100)
            with st.spinner("Training the deep-forecast models (walk-forward validation, about 15-30 s the first time)…"):
                deep = run_deep_forecast(ml_ohlcv, int(forecast_weeks * 5), float(prob_band_sigma), int(ml_trees), int(ml_splits),
                                         bool(ml_use_rf), bool(ml_use_xgb), bool(ml_use_sarimax), tuple(ml_order), tuple(ml_seasonal))
        except Exception as e:   # an optional overlay must never take the whole app down
            deep = {"ok": False, "error": f"{type(e).__name__}: {e}", "warnings": [], "models": {}}

    # Engine + Deep Forecast: skill-weighted probability blend and a conviction score (see deep_forecast.combine)
    blend = None
    if deep is not None and deep.get("ok") and prob is not None:
        _cb, _cs, _ca = prob["cond"]["bull"], prob["cond"]["side"], prob["cond"]["bear"]
        blend = dfc.combine({"probs": {"bull": pB, "side": pS, "bear": pA},
                             "targets": {"bull": _cb["median"] if _cb else prob["bull_price"], "side": _cs["median"] if _cs else last_price,
                                         "bear": _ca["median"] if _ca else prob["bear_price"]},
                             "last_price": last_price, "range": (prob["p10"], prob["p90"])}, deep, ml_blend_cap / 100)
    blend_active = bool(blend is not None and use_blend and blend["weight"] > 0)

    # ============================================================
    # FORECAST LINE (v1.5.0): ONE consensus line = forecast engine + Fibonacci pattern + Random Forest + XGBoost + SARIMAX
    # (deep_forecast.consensus_line). It replaces the Bullish / Neutral / Bearish and weighted paths: the probabilities stay in the panels.
    # ============================================================
    target_fibo_recovery = fibo_levels["0.618 (Golden Ratio)"]
    target_fibo_bearish = fibo_levels["0.000 (Trend Low)"]
    targets_from_engine = prob is not None
    future_x_axis = pd.DatetimeIndex([last_date]).append(dates_future)

    # The Fibonacci forecast line always uses the LAST YEAR of daily data (pattern, Fibonacci levels), so it does not change with the chart period.
    _pat_fx, _fibo_fx = pattern_result, fibo_levels
    if show_forecast_line and dfc is not None:
        try:
            _dfx = load_stock_data(ticker, (pd.Timestamp(end_date) - pd.DateOffset(years=1)).date(), end_date, "1d")
            if len(_dfx) >= 120:
                _pat_fx = detect_chart_pattern(_dfx, int(_dfx["Low"].values.argmin()))
                _ph = ath_level if (use_custom_fibo and ath_level > 0) else float(_dfx["High"].max())
                _pl = crash_low_level if (use_custom_fibo and crash_low_level > 0) else float(_dfx["Low"].min())
                _dd = _ph - _pl
                _fibo_fx = {"1.000 (Trend High)": _ph, "0.786": _pl + 0.786 * _dd, "0.618 (Golden Ratio)": _pl + 0.618 * _dd, "0.500": _pl + 0.500 * _dd,
                            "0.382": _pl + 0.382 * _dd, "0.236": _pl + 0.236 * _dd, "0.000 (Trend Low)": _pl}
        except Exception:
            _pat_fx, _fibo_fx = pattern_result, fibo_levels
    fline = None
    _primary, _prim, _nc, _fib = None, None, 0, None          # defaults: the ML candles also need the levels when the pattern line is off
    if show_forecast_line and dfc is not None:
        _H = int(forecast_weeks * 5)
        _prim = _pat_fx["primary"] if _pat_fx is not None else None
        _primary, _nc = None, 0
        if _prim and _prim.get("pattern"):
            _imp = str(_prim.get("implication", "")).lower()
            _bias = 0 if "ambiguous" in _imp else (1 if _imp.startswith("bullish") else -1 if _imp.startswith("bearish") else 0)
            _nc = int(_prim["n_consol_bars"])
            _primary = {"bias": _bias, "height": abs(float(_prim["flagpole_end_price"]) - float(_prim["flagpole_start_price"])),
                        "upper": float(_prim["h_slope"] * (_nc - 1) + _prim["h_intercept"]),
                        "lower": float(_prim["l_slope"] * (_nc - 1) + _prim["l_intercept"]),
                        "days": max(1, int(np.busday_count(pd.Timestamp(_prim["flagpole_start"]).date(), pd.Timestamp(_prim["flagpole_end"]).date())))}
        _secs = []
        for _sp in (_pat_fx["secondary"] if _pat_fx is not None else []):
            _nk, _lv0 = float(_sp["neckline"]), float(_sp["level"])
            _bull = _sp["direction"] == "bullish"
            _secs.append({"bias": 1 if _bull else -1, "target": _nk + (_nk - _lv0) if _bull else _nk - (_lv0 - _nk),
                          "confirmed": last_price > _nk if _bull else last_price < _nk})
        _fib = dfc.fib_pattern_target(last_price, _primary, _secs, list(_fibo_fx.values()))
        try:
            _eng = float(prob["shift"]) if prob is not None else 0.0
            fline = dfc.pattern_line(last_price, _H, _eng, _fib)          # the candles follow the pattern projection ...
            fline["ref"] = dfc.consensus_line(last_price, _H, _eng, _fib, deep if (deep is not None and deep.get("ok")) else None)   # ... the all-methods consensus is a reference
        except Exception as e:
            fline = None
            st.warning(f"Forecast line failed: {type(e).__name__}: {e}")
    mlc = None                                            # candles built on the ML models (Random Forest + XGBoost + SARIMAX)
    if show_deep and dfc is not None and deep is not None and deep.get("ok"):
        try:
            _Hm = int(forecast_weeks * 5)
            mlc = dfc.consensus_line(last_price, _Hm, 0.0, None, deep, ml_only=True)
            if mlc is not None:
                mlc["candles"] = dfc.forecast_candles(mlc["line"], last_price, df["Open"], df["High"], df["Low"], df["Close"],
                                                      int(pd.Timestamp(last_date).strftime("%Y%m%d")))
                mlc["dates"] = pd.bdate_range(start=pd.Timestamp(last_date) + pd.Timedelta(days=1), periods=_Hm)
        except Exception as e:
            mlc = None
            st.warning(f"ML candles failed: {type(e).__name__}: {e}")
    if fline is not None or mlc is not None:
        # the levels the lines may cross: trendlines extended into the future, the pattern lines, the Fibonacci levels
        _H = int(forecast_weeks * 5)
        _bpd = {"1d": 1.0, "1wk": 1 / 5, "1mo": 1 / 21.7}[interval]
        _k = np.arange(1, _H + 1) * _bpd
        _lv = {}
        if not res_fallback:
            _lv["Macro resistance trendline"] = res_slope * ((len(df) - 1) + _k) + res_intercept
        if not sup_fallback:
            _lv["Macro support trendline"] = sup_slope * ((len(df) - 1) + _k) + sup_intercept
        if _primary:
            _lv["Pattern upper line"] = _prim["h_slope"] * ((_nc - 1) + _k) + _prim["h_intercept"]
            _lv["Pattern lower line"] = _prim["l_slope"] * ((_nc - 1) + _k) + _prim["l_intercept"]
        for _fn, _fv in _fibo_fx.items():
            _lv["Fibonacci " + _fn.split(" (")[0]] = float(_fv)
        if fline is not None:
            fline["events"] = dfc.timing_events(fline["line"], _lv, last_price)
            fline["dates"] = pd.bdate_range(start=pd.Timestamp(last_date) + pd.Timedelta(days=1), periods=_H)
            fline["fib"] = _fib
        if mlc is not None:
            mlc["events"] = dfc.timing_events(mlc["line"], _lv, last_price)

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

    # 2. Lower-High Peaks (only when resistance is active -- these are its fitting points)
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
            else "Macro Resistance Trendline (fallback — data lower-high < 2 points)"
        fig.add_trace(go.Scatter(
            x=dates_historical,
            y=trendline_y,
            mode="lines",
            line=dict(color="#f59e0b", width=2.5),
            name=res_label
        ))

    # 3b. Macro Support Trendline (toggle-able; regression on swing lows — NOT a static Fibonacci)
    if show_macro_support:
        sup_label = f"Macro Support Trendline (R²={sup_r2_val:.2f})" if sup_r2_val is not None \
            else "Macro Support Trendline (fallback — data swing-low < 2 points)"
        fig.add_trace(go.Scatter(
            x=dates_historical,
            y=support_trendline_y,
            mode="lines",
            line=dict(color="#34d399", width=2.5),
            name=sup_label
        ))

    # 3c. Lower-Low markers (only when support is active -- these are its fitting points)
    if show_macro_support and len(ll_idx) > 0:
        fig.add_trace(go.Scatter(
            x=dates_historical[ll_idx],
            y=df["Low"].iloc[ll_idx],
            mode="markers",
            marker=dict(symbol="triangle-up", size=9, color="#34d399"),
            name="Lower-Low Pivots"
        ))

    # 3d. Channel area shading — drawn only when BOTH lines are active
    #     (the area "between resistance & support" is meaningless when one of them is hidden)
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

    # 3e. Historical S/R Zones (a horizontal segment per zone, "polarity" style)
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
                fig.add_annotation(x=rd, y=df['High'].max(), text="AGM/EGM", showarrow=True, arrowhead=2,
                                   font=dict(color="#fb923c", size=10, family="Arial Black"))

    # 4. The forecast line (ONE line) with the dates at which it reaches the trendlines / pattern lines / Fibonacci levels
    if fline is not None:
        fig.add_trace(go.Scatter(x=[last_date] + list(fline["dates"]), y=[last_price] + fline["line"], mode="lines",
                                 line=dict(color="#f8fafc", width=2.5, dash="dash"), name="Forecast line"))
        _rk = lambda e: 3 if e["kind"] != "cross" else (0 if e["name"].startswith(("Macro", "Cycle")) else 1 if e["name"].startswith("Pattern") else 2)

        def _add_events(src, tpos_up, tpos_dn):
            _evs = sorted(sorted(src["events"], key=lambda e: (_rk(e), e["day"]))[:6], key=lambda e: e["day"])
            if _evs:
                fig.add_trace(go.Scatter(
                    x=[src["dates"][e["day"] - 1] for e in _evs], y=[e["price"] for e in _evs], mode="markers+text",
                    text=[f"{dfc.event_label(e)} · {src['dates'][e['day'] - 1]:%d %b}" for e in _evs],
                    textposition=[tpos_up if e["dir"] > 0 else tpos_dn for e in _evs], showlegend=False,
                    textfont=dict(size=11, color=["#34d399" if e["dir"] > 0 else "#f87171" for e in _evs]),
                    marker=dict(size=9, color="#111827", line=dict(width=2, color=["#34d399" if e["dir"] > 0 else "#f87171" for e in _evs])),
                    hovertemplate="%{text}<br>Rp %{y:,.0f}<extra></extra>"))
        _add_events(fline, "top center", "bottom center")
        _pw = blend["probs"] if blend_active else {"bull": pB, "side": pS, "bear": pA}
        fig.add_annotation(x=fline["dates"][-1], y=fline["line"][-1], xanchor="right", yanchor="bottom", showarrow=False, ay=-10,
                           text=(f"<b>Pattern line Rp {fline['line'][-1]:,.0f} ({fline['ret']:+.1%})</b>"
                                 + (f"<br>Bull {_pw['bull']:.0%} · Neutral {_pw['side']:.0%} · Bear {_pw['bear']:.0%}" if prob is not None else "")),
                           font=dict(size=12, color="#f8fafc"), bgcolor="rgba(0,0,0,0.55)")

    if mlc is not None:
        _mc = mlc["candles"]
        fig.add_trace(go.Candlestick(
            x=list(mlc["dates"]), open=[c["o"] for c in _mc], high=[c["h"] for c in _mc], low=[c["l"] for c in _mc], close=[c["c"] for c in _mc],
            increasing=dict(line=dict(color="#34d399", width=1.2), fillcolor="rgba(52,211,153,0.45)"),
            decreasing=dict(line=dict(color="#f87171", width=1.2), fillcolor="rgba(248,113,113,0.45)"),
            name="ML forecast candles"))
        fig.add_trace(go.Scatter(x=[last_date] + list(mlc["dates"]), y=[last_price] + mlc["line"], mode="lines", showlegend=False,
                                 line=dict(color="rgba(248,250,252,0.45)", width=1.2, dash="dot"), hoverinfo="skip"))
        if mlc.get("events"):
            _evm = sorted(sorted(mlc["events"], key=lambda e: (3 if e["kind"] != "cross" else 0 if e["name"].startswith(("Macro", "Cycle")) else 1 if e["name"].startswith("Pattern") else 2, e["day"]))[:6], key=lambda e: e["day"])
            fig.add_trace(go.Scatter(
                x=[mlc["dates"][e["day"] - 1] for e in _evm], y=[e["price"] for e in _evm], mode="markers+text", showlegend=False,
                text=[f"{dfc.event_label(e)} · {mlc['dates'][e['day'] - 1]:%d %b}" for e in _evm],
                textposition=["top center" if e["dir"] > 0 else "bottom center" for e in _evm],
                textfont=dict(size=11, color=["#34d399" if e["dir"] > 0 else "#f87171" for e in _evm]),
                marker=dict(size=9, color="#111827", line=dict(width=2, color=["#34d399" if e["dir"] > 0 else "#f87171" for e in _evm])),
                hovertemplate="ML: %{text}<br>Rp %{y:,.0f}<extra></extra>"))
        fig.add_annotation(x=mlc["dates"][-1], y=mlc["line"][-1], xanchor="right", yanchor="top", showarrow=False, ay=10,
                           text=f"<b>ML forecast Rp {mlc['line'][-1]:,.0f} ({mlc['ret']:+.1%})</b>", font=dict(size=12, color="#f8fafc"),
                           bgcolor="rgba(0,0,0,0.55)")

    if blend is not None:
        _dc = {"bull": "#22c9c3", "bear": "#ef4444", "side": "#9ca3af"}[blend["direction"]]
        fig.add_annotation(x=1.0, y=1.0, xref="paper", yref="paper", xanchor="right", yanchor="top", showarrow=False,
                           text=f"<b>Conviction {blend['conviction']}/100 · {blend['label']}</b>", font=dict(size=13, color=_dc),
                           bgcolor="rgba(0,0,0,0.55)")

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

    # Break-even price line (optional)
    if bep_level > 0:
        fig.add_hline(y=bep_level, line_dash="dashdot", line_color="#818cf8", annotation_text=f"Break-even price: {bep_level:,.0f}")

    # Layout Styling
    fig.update_layout(
        title=f"{clean_ticker} — Macro Cycle: Bull Market ➔ Descending Staircase ➔ Breakout/Retest",
        yaxis_title="Price (IDR)",
        xaxis_title="Date Period",
        height=700,
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis_rangeslider_visible=False,
        xaxis=dict(
            type='date',
            tickformat=plotly_date_format_map.get(date_format_option, "%b %Y"),
            # hoverformat is SEPARATE from tickformat: the axis stays compact (e.g. "Sep 2026"),
            # but the hover tooltip always shows the FULL DATE instead of following tickformat.
            hoverformat="%d %b %Y (%a)"
        )
    )

    mac_plotly(fig)
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

    # Second row: channel classification (parallel vs converging)
    if is_parallel_channel is not None:
        shape_label = "📐 PARALLEL CHANNEL" if is_parallel_channel else "📐 CONVERGING WEDGE/TRIANGLE"
        st.info(f"{shape_label} — support/resistance slope ratio = {slope_ratio:.2f} "
                f"(close to 1.0 = parallel; further from 1.0 = converging)")
    if res_fallback or sup_fallback:
        st.warning("⚠️ One of the lines (resistance/support) is using a FALLBACK value because "
                   "fewer than 2 pivot points were found — try lowering 'Pivot Distance' in the "
                   "sidebar if the historical data range is long enough.")

    # Forecast Engine panel: Regime -> Technical -> Fundamental -> Probabilities -> Target / Range
    if show_engine or show_probabilities:
        with st.expander("🧭 Forecast Engine — Regime → Technical → Fundamental → Probabilities → Target / Range", expanded=True):
            st.markdown(
                "`JCI · USD/IDR · BI Rate · US10Y · DXY · Global` + `Price · Volume · Foreign flow` → **1 · Market regime** → "
                "**2 · Technical** (RSI/MACD/MA · ATR/BB/RS) → **3 · Fundamental** (EPS/ROE/PBV · PER/Dividend) → "
                "**Forecast engine** → **Bullish / Neutral / Bearish** → **Target / Range**")

            # ---------- layers ----------
            if show_engine:
                if not engine_rows:
                    st.warning("The engine produced no scorecard (see the notes below). The probabilities fall back to "
                               "the no-view base rate.")
                else:
                    lc = st.columns(4)
                    for col, L in zip(lc[:3], LAYERS):
                        s_ = layer_stats[L]
                        mn_n, mn_g = LAYER_MIN[L]
                        counted = layer_counts(layer_stats, L)
                        val = "n/a" if s_["score"] is None else (f"{s_['score']:+.2f}" if counted else "excluded")
                        col.metric(LAYER_LABELS[L], val,
                                   help=f"{s_['n']} scored signals in {s_['n_groups']} group(s), out of {s_['n_rows']} rows. "
                                        f"A layer needs >= {mn_n} signals across >= {mn_g} groups to influence the forecast.")
                    lc[3].metric("Market regime",
                                 regime_label(regime_score) if layer_counts(layer_stats, "Regime") else "Unknown",
                                 help="Risk-on when the regime-layer score is above +0.25, Risk-off below -0.25 (heuristic cut-offs).")
                    if eff_weights:
                        parts = " + ".join(f"{eff_weights[L]:.0%} × {layer_stats[L]['score']:+.2f} ({LAYER_LABELS[L].split(' · ')[1]})"
                                           for L in eff_weights)
                        st.markdown(f"**Combined evidence = {evidence:+.2f}**  =  {parts}  ·  tilt strength {tilt_strength:.0%}")
                    else:
                        st.info("No layer has enough data, so the combined evidence is 0 and the probabilities are the no-view base rate.")

                    tabs = st.tabs([LAYER_LABELS[L] for L in LAYERS])
                    for tab, L in zip(tabs, LAYERS):
                        with tab:
                            gs = layer_stats[L]["groups"]
                            if gs:
                                for gc, (gn, gv) in zip(st.columns(len(gs)), gs.items()):
                                    gc.metric(gn, f"{gv:+.2f}")
                            lrows = [r for r in engine_rows if r["Layer"] == L or (L == "Technical" and r["Layer"] == "Info")]
                            st.dataframe(pd.DataFrame([{
                                "Group": r["Group"], "Factor": r["Factor"], "Latest": r["Latest"],
                                "Signal": "ℹ️ info" if r["Info"] else sig_txt(r["Signal"]), "Rule": r["Rule"]} for r in lrows]),
                                use_container_width=True, hide_index=True)
            else:
                st.info("Layered engine is switched off: the probabilities below use the no-view base rate (evidence = 0).")

            for note in engine_inputs.get("notes", []):
                st.warning(note)
            for src_line in engine_inputs.get("sources", []):
                st.caption(f"📡 {src_line}")

            # ---------- probabilities ----------
            if show_probabilities:
                st.markdown("##### Probabilities")
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
                    pc2.metric("🟡 Neutral", f"{pS:.0%}", delta=_dpp(pS - base_s), delta_color="off",
                               help=f"Price ends between Rp {prob['bear_price']:,.0f} and Rp {prob['bull_price']:,.0f}.")
                    pc3.metric("🔴 Bearish", f"{pA:.0%}", delta=_dpp(pA - base_a), delta_color="inverse",
                               help=f"Price ends the {hd}-trading-day horizon below Rp {prob['bear_price']:,.0f}.")

                    bar = go.Figure()
                    for nm, val, col in (("Bullish", pB, "#22c9c3"), ("Neutral", pS, "#9ca3af"), ("Bearish", pA, "#ef4444")):
                        bar.add_trace(go.Bar(x=[val * 100], y=[""], orientation="h", name=nm, marker_color=col,
                                             text=f"{nm} {val:.0%}", textposition="inside", insidetextanchor="middle",
                                             hoverinfo="skip"))
                    bar.update_layout(barmode="stack", height=90, showlegend=False,
                                      margin=dict(l=0, r=0, t=0, b=0),
                                      xaxis=dict(range=[0, 100], showticklabels=False, showgrid=False),
                                      yaxis=dict(showticklabels=False))
                    mac_plotly(bar)
                    st.plotly_chart(bar, use_container_width=True)

                    # ---------- target / range ----------
                    st.markdown("##### Target / Range")
                    cb, cs, ca = prob["cond"]["bull"], prob["cond"]["side"], prob["cond"]["bear"]

                    def _tgt(c):
                        return "n/a (too few paths)" if not c else f"Rp {c['median']:,.0f}"

                    def _rng(c):
                        return "n/a" if not c else f"Rp {c['p25']:,.0f} – Rp {c['p75']:,.0f}"

                    st.dataframe(pd.DataFrame([
                        {"Scenario": "🟢 Bullish", "Probability": f"{pB:.0%}", "Target (median)": _tgt(cb), "Range": _rng(cb),
                         "Defined as": f"ends above Rp {prob['bull_price']:,.0f} ({(prob['bull_price'] / last_price - 1) * 100:+.1f}%)"},
                        {"Scenario": "🟡 Neutral", "Probability": f"{pS:.0%}", "Target (median)": _tgt(cs), "Range": _rng(cs),
                         "Defined as": f"ends between Rp {prob['bear_price']:,.0f} and Rp {prob['bull_price']:,.0f}"},
                        {"Scenario": "🔴 Bearish", "Probability": f"{pA:.0%}", "Target (median)": _tgt(ca), "Range": _rng(ca),
                         "Defined as": f"ends below Rp {prob['bear_price']:,.0f} ({(prob['bear_price'] / last_price - 1) * 100:.1f}%)"},
                        {"Scenario": "⚖️ All paths", "Probability": "100%",
                         "Target (median)": f"Rp {prob['p50']:,.0f}  (mean Rp {prob['mean_price']:,.0f})",
                         "Range": f"Rp {prob['p10']:,.0f} – Rp {prob['p90']:,.0f}",
                         "Defined as": f"80% range (P10–P90) after {hd} trading days"},
                    ]), use_container_width=True, hide_index=True)
                    st.caption("Each scenario's target is the median final price among the simulated paths that END in that "
                               "state; its range is the middle 50% (P25–P75) of those paths. Horizon "
                               f"σ ≈ {prob['sigma_h'] * 100:.1f}%, neutral band ±{prob['band'] * 100:.1f}%. "
                               "The forecast line on the chart is a consensus of several methods; see the Forecast line panel.")

                    tch = engine_inputs.get("tech") or {}
                    if tch.get("atr"):
                        atr_rng = tch["atr"] * np.sqrt(hd)
                        st.caption(f"ATR cross-check: ATR(14) = Rp {tch['atr']:,.0f} ({tch['atr_pct']:.1f}% of price), so ATR × √{hd} "
                                   f"≈ ±Rp {atr_rng:,.0f} (±{atr_rng / last_price * 100:.0f}%) versus the simulated horizon σ of "
                                   f"{prob['sigma_h'] * 100:.0f}%. ATR measures the daily high-low range, not a standard "
                                   "deviation, so it is only a rough sanity check.")
                    p_hit_b = float((prob["price_end"] >= target_fibo_recovery).mean())
                    p_hit_a = float((prob["price_end"] <= target_fibo_bearish).mean())
                    st.caption(f"Fibonacci reference: P(price ends at or above the 0.618 level Rp {target_fibo_recovery:,.0f}) = "
                               f"{p_hit_b:.0%}; P(ends at or below the 0.000 level Rp {target_fibo_bearish:,.0f}) = {p_hit_a:.0%}.")
                    if not targets_from_engine:
                        if target_fibo_recovery < prob["bull_price"]:
                            st.info(f"⚠️ The Fibonacci target for Scenario B (Rp {target_fibo_recovery:,.0f}) is **below** the Bullish "
                                    f"threshold (Rp {prob['bull_price']:,.0f}), so the plotted recovery path would not by itself count "
                                    "as 'Bullish'. Switch 'Scenario path targets' to the forecast engine, widen the Start Date, or "
                                    "set custom High/Low levels.")
                        if target_fibo_bearish > prob["bear_price"]:
                            st.info(f"⚠️ The Fibonacci target for Scenario A (Rp {target_fibo_bearish:,.0f}) is **above** the Bearish "
                                    f"threshold (Rp {prob['bear_price']:,.0f}), so the plotted bearish path would not by itself "
                                    "count as 'Bearish'.")

            st.caption(
                "**How to read this.** Every factor is a rule-of-thumb signal between -1 (adverse) and +1 (supportive); the "
                "thresholds are heuristics, **not calibrated against this stock's own history**. Rows are averaged inside "
                "groups, groups inside layers, and the three layers are combined with the sidebar weights, so correlated "
                "factors (e.g. the five rates rows) cannot outvote everything else by sheer count. A layer with too few "
                "signals is excluded rather than trusted. The base rate bootstraps the stock's own daily returns after "
                "removing their average drift, so with no evidence the odds depend only on volatility and the neutral band "
                "(roughly 31/38/31 at the default band) — no directional view. The combined evidence then shifts the expected "
                "return by at most ±0.5 sigma at 100% tilt strength. These are **model-implied probabilities under those "
                "assumptions, not calibrated forecasts**: returns are assumed to resemble their recent past, and event "
                "risk (earnings, policy shocks) is not modelled. Technical and pattern signals depend on the selected date "
                "range; BI Rate, BI-Rate expectation and foreign flow are manual inputs; yfinance fundamentals are often "
                "missing or stale for IDX tickers; EV/EBITDA is generally not meaningful for banks.")

    # Forecast line panel (v1.5.0)
    if fline is not None:
        with st.expander("📈 Forecast line — pattern projection, with the other methods as reference", expanded=True):
            _Hn = len(fline["line"])
            _m1, _m2, _m3 = st.columns(3)
            _m1.metric(f"Pattern line at H+{_Hn}", f"Rp {fline['line'][-1]:,.0f}", delta=f"{fline['ret']:+.1%}", delta_color="off")
            _m2.metric(f"All-methods consensus at H+{_Hn}", f"Rp {fline['ref']['line'][-1]:,.0f}", delta=f"{fline['ref']['ret']:+.1%}", delta_color="off")
            _m3.metric("Projected events", str(len(fline["events"])), delta="crossings and turning points", delta_color="off")
            _cv = (deep or {}).get("curve") or {} if (deep is not None and deep.get("ok")) else {}
            _dm = deep.get("models", {}) if (deep is not None and deep.get("ok")) else {}

            def _note(p):
                if p["name"] == "Fibonacci pattern" and fline.get("fib"):
                    f_ = fline["fib"]
                    return f"{'Fibonacci level' if f_['snapped'] else 'measured move'} Rp {f_['target']:,.0f} reached in ~{max(10, f_['days'])} days"
                if p["name"] == "Forecast engine":
                    return "evidence tilt spread evenly: the engine carries no timing information"
                if p["name"] in _dm:
                    key_ = "rf" if p["name"] == "Random Forest" else "xgb"
                    sk_ = (_cv.get("skill") or {}).get(key_)
                    return (f"trained per horizon · skill {sk_:+.0f}% vs baseline" if (sk_ or 0) > 0 else "no out-of-sample skill → quarter weight")
                if p["name"] == "SARIMAX":
                    ho_ = (deep.get("sarimax") or {}).get("holdout") if deep else None
                    return ("worse than a flat price on hold-out → quarter weight" if ho_ and ho_["model_mae_pct"] > ho_["flat_mae_pct"] else "statistical drift path")
                return ""

            st.dataframe(pd.DataFrame([{"Component": p["name"], "Weight": f"{p['weight']:.0%}",
                                        f"Its own forecast at H+{_Hn}": f"Rp {p['end_price']:,.0f} ({p['end_ret']:+.1%})", "Basis": _note(p)}
                                       for p in fline["ref"]["parts"]]), use_container_width=True, hide_index=True)
            st.caption("The dashed line is the pattern projection only; the ML candles follow the ML models. This table is a reference: what each method says on its own, and its weight in the "
                       "all-methods consensus.")
            if show_deep and (deep is None or not deep.get("ok")):
                st.info("The ML models are not in the reference yet (Deep Forecast is off, still unavailable, or has too little data).")
            st.markdown("##### When could the breakout or the decline happen?")
            if fline["events"]:
                st.dataframe(pd.DataFrame([{"When": f"{fline['dates'][e['day'] - 1]:%d %b %Y} (H+{e['day']})",
                                            "Event": dfc.event_label(e) + (f" · {e['name']}" if e["kind"] == "cross" else ""),
                                            "Line price": f"Rp {e['price']:,.0f}"} for e in fline["events"]]), use_container_width=True, hide_index=True)
            else:
                st.caption("The forecast line does not cross the trendlines, the pattern lines or a Fibonacci level within this horizon, "
                           "and has no turning point of at least 1.5%.")
            st.caption("The dates are where the pattern line reaches the level (Fibonacci waves to the pattern target within the flagpole duration, then an A-B-C correction), not a "
                       "certainty. Compare with the all-methods consensus above: when it disagrees with the pattern line, the pattern is less likely to play out.")

    # Deep Forecast panel (v1.5.0)
    if deep is not None:
        with st.expander("🤖 Deep Forecast — Random Forest · XGBoost · SARIMAX (walk-forward validated)", expanded=True):
            if not deep.get("ok"):
                st.info(deep.get("error") or "Deep Forecast did not run.")
                for _w in deep.get("warnings", []):
                    st.warning(_w)
            else:
                _H, _last = deep["horizon"], deep["last_price"]
                st.caption(f"Horizon {_H} trading days ({forecast_weeks} weeks) · trained on {deep['n_samples']} daily samples × {deep['n_features']} "
                           f"technical features up to {deep['last_date']} · Neutral band ±{deep['band']:.1%} (= {prob_band_sigma:.2f} × horizon σ "
                           f"{deep['sigma_h']:.1%}) · base rate in history: Bearish {deep['class_share']['bear']:.0%} / Neutral "
                           f"{deep['class_share']['side']:.0%} / Bullish {deep['class_share']['bull']:.0%}")
                for _w in deep.get("warnings", []):
                    st.warning(_w)

                _rows = []
                for _nm, _m in deep["models"].items():
                    _cv = _m.get("cv") or {}
                    _rows.append({
                        "Model": _nm, "Expected return": f"{_m['ret']:+.1%}", "Target": f"Rp {_m['target']:,.0f}",
                        "80% range": f"{_m['lo']:,.0f} – {_m['hi']:,.0f}",
                        "Bullish / Neutral / Bearish": f"{_m['probs']['bull']:.0%} / {_m['probs']['side']:.0%} / {_m['probs']['bear']:.0%}",
                        "Skill vs baseline": (f"{_cv['skill_pct']:+.0f}%" if _cv.get("skill_pct") is not None else "n/a"),
                        "Direction hit-rate": (f"{_cv['dir_acc']:.0%}" if _cv.get("dir_acc") is not None else "n/a"),
                        "Class accuracy (base)": (f"{_cv['clf_acc']:.0%} ({_cv['clf_base_acc']:.0%})" if _cv.get("clf_acc") is not None else "n/a"),
                        "Folds": str(_cv.get("folds", 0))})
                _e = deep.get("ensemble")
                if _e and len(deep["models"]) > 1:
                    _rows.append({"Model": "Ensemble (mean)", "Expected return": f"{_e['ret']:+.1%}", "Target": f"Rp {_e['target']:,.0f}",
                                  "80% range": f"{_e['lo']:,.0f} – {_e['hi']:,.0f}",
                                  "Bullish / Neutral / Bearish": f"{_e['probs']['bull']:.0%} / {_e['probs']['side']:.0%} / {_e['probs']['bear']:.0%}",
                                  "Skill vs baseline": "", "Direction hit-rate": "", "Class accuracy (base)": "", "Folds": ""})
                _sx = deep.get("sarimax")
                if _sx and "path" in _sx:
                    _ho = _sx.get("holdout")
                    _rows.append({"Model": f"SARIMAX{_sx['order']}" + ("" if _sx.get("drift") else " (no drift)"), "Expected return": f"{np.exp(_sx['ret']) - 1:+.1%}",
                                  "Target": f"Rp {_sx['path'][_H - 1]:,.0f}", "80% range": f"{_sx['lo'][_H - 1]:,.0f} – {_sx['hi'][_H - 1]:,.0f}",
                                  "Bullish / Neutral / Bearish": "–",
                                  "Skill vs baseline": (f"hold-out error {_ho['model_mae_pct']:.1f}% vs flat {_ho['flat_mae_pct']:.1f}%" if _ho else "n/a"),
                                  "Direction hit-rate": "", "Class accuracy (base)": "", "Folds": "1" if _ho else ""})
                elif _sx and _sx.get("error"):
                    st.warning(f"SARIMAX failed: {_sx['error']}")
                if _rows:
                    st.dataframe(pd.DataFrame(_rows), use_container_width=True, hide_index=True)

                if _e and prob is not None:
                    st.markdown("##### Compared with the Forecast Engine")
                    _c1, _c2, _c3 = st.columns(3)
                    _c1.metric("🟢 Bullish · ML ensemble", f"{_e['probs']['bull']:.0%}", delta=f"{(_e['probs']['bull'] - pB) * 100:+.0f} pp vs engine {pB:.0%}", delta_color="off")
                    _c2.metric("🟡 Neutral · ML ensemble", f"{_e['probs']['side']:.0%}", delta=f"{(_e['probs']['side'] - pS) * 100:+.0f} pp vs engine {pS:.0%}", delta_color="off")
                    _c3.metric("🔴 Bearish · ML ensemble", f"{_e['probs']['bear']:.0%}", delta=f"{(_e['probs']['bear'] - pA) * 100:+.0f} pp vs engine {pA:.0%}", delta_color="off")
                    _eng_dir = "bullish" if pB > pA else "bearish"
                    _ml_dir = "bullish" if _e["probs"]["bull"] > _e["probs"]["bear"] else "bearish"
                    st.caption(f"The engine leans {_eng_dir} and the ML ensemble leans {_ml_dir}: "
                               + ("they agree on the direction." if _eng_dir == _ml_dir else "they DISAGREE. Treat the signal as weak until they line up."))

                if blend is not None:
                    st.markdown("##### Combined view — Forecast Engine + Deep Forecast")
                    _bp, _ep = blend["probs"], blend["probs_engine"]
                    _k1, _k2, _k3, _k4 = st.columns(4)
                    _k1.metric("Conviction", f"{blend['conviction']}/100", delta=blend["label"], delta_color="off")
                    _k2.metric("🟢 Bullish · blended", f"{_bp['bull']:.0%}", delta=f"{round((_bp['bull'] - _ep['bull']) * 100):+d} pp vs engine", delta_color="off")
                    _k3.metric("🟡 Neutral · blended", f"{_bp['side']:.0%}", delta=f"{round((_bp['side'] - _ep['side']) * 100):+d} pp vs engine", delta_color="off")
                    _k4.metric("🔴 Bearish · blended", f"{_bp['bear']:.0%}", delta=f"{round((_bp['bear'] - _ep['bear']) * 100):+d} pp vs engine", delta_color="off")
                    _rng = f" · 80% range {blend['range'][0]:,.0f} – {blend['range'][1]:,.0f}" if blend.get("range") else ""
                    st.caption(f"**ML weight {blend['weight']:.0%}** ({blend['weight_reason']}). Combined expected price **Rp {blend['expected']:,.0f}** "
                               f"({blend['expected'] / last_price - 1:+.1%}; engine alone Rp {blend['expected_engine']:,.0f}){_rng}. "
                               + ("The blend feeds the scenario probabilities on the chart." if blend_active else
                                  "The blend is NOT applied to the chart (switched off, or the ML weight is 0)."))
                    _vn = {1: "🟢 Bullish", 0: "🟡 Neutral", -1: "🔴 Bearish"}
                    st.dataframe(pd.DataFrame([{"Voter": v["name"], "Vote": _vn[v["dir"]], "Weight": f"{v['weight']:.2f}", "Why": v["note"]}
                                               for v in blend["votes"]]), use_container_width=True, hide_index=True)
                    st.caption("Conviction = 100 × the weighted vote (engine 0.40, each ML model 0.20, SARIMAX 0.20; a voter without out-of-sample skill counts "
                               "a quarter). The models read the same price history as the engine, so agreement means robustness, not independent confirmation.")

                if deep.get("importance"):
                    st.markdown("##### What drives the models (feature importance, Random Forest / XGBoost average)")
                    _imp = pd.DataFrame(deep["importance"], columns=["Feature", "Importance"]).iloc[::-1]
                    _fi = go.Figure(go.Bar(x=_imp["Importance"], y=_imp["Feature"], orientation="h", marker_color="#60a5fa"))
                    _fi.update_layout(height=max(220, 26 * len(_imp) + 60), margin=dict(l=0, r=10, t=10, b=0), xaxis_title="Share of total importance")
                    mac_plotly(_fi)
                    st.plotly_chart(_fi, use_container_width=True)

                st.caption(
                    "**How to read this.** The regressors predict the forward return over the horizon, the classifiers predict Bullish / Neutral / "
                    "Bearish (the same three states as the Forecast Engine). Features use only data up to the prediction day. **Skill vs baseline** = how "
                    "much smaller the out-of-sample error (MAE) is than simply predicting the training-set mean; zero or negative = no edge. "
                    "Validation is walk-forward with a gap of one horizon between train and test, because consecutive rows share most of their "
                    "target. Class probabilities are **shrunk toward the historical base rate** in proportion to the classifier's out-of-sample "
                    "advantage over always guessing the majority class. The 80% range comes from out-of-fold residuals, not from the in-sample fit. "
                    "With only a few independent horizon-length windows in ~3 years, the scores are noisy and regime-dependent (a model can "
                    "'learn' a downtrend that then reverses). This is a research aid, not a calibrated forecast, and it knows nothing about earnings, "
                    "policy shocks or foreign flow.")

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
                source_label = dividend_source_actual
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
                st.markdown("#### 🏦 Bank Indonesia Board of Governors' meetings 2026 (officially published schedule)")
                bi_df = pd.DataFrame(BI_RDG_2026, columns=['Decision Date', 'Meeting'])
                st.dataframe(bi_df, use_container_width=True, hide_index=True)
                st.caption("Source: Bank Indonesia official press releases (as announced Dec 2025). "
                          "BI Rate decisions are announced on the 2nd day of each meeting and directly "
                          "affect bank-sector stocks like BBRI/BBCA/BMRI.")

            if show_rups:
                st.markdown("#### 🗳️ AGM/EGM (shareholders' meeting) — manual input")
                if not rups_dates:
                    st.write("No AGM/EGM dates entered yet. Paste known dates in the sidebar — check "
                            "**www.idx.co.id** (IDX disclosures) or the company's investor "
                            "relations page for official schedules.")
                else:
                    st.write(f"{len(rups_dates)} AGM/EGM date(s) marked on the chart: " +
                             ", ".join(d.strftime('%d %b %Y') for d in sorted(rups_dates)))
                    st.caption("AGM/EGM can announce dividends, buybacks, rights issues, or management "
                              "changes — any of which can move the stock. No free public API covers "
                              "this, so dates must be entered manually.")

    st.write("")  # Spacer

    # Market Structure Summary card
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