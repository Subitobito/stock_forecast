"""IDX-API information for IDX Trend Studio: read-only SQLite adapter + background sync (Cron.ts via Deno).

Standalone port (no Streamlit) of IDXAPIAdapter / IDXAutoSync from stock_forecast_app_1.4.7.py, with the same schema handling:
 - dates are epoch MILLISECONDS (+7 h = WIB calendar day)
 - stock_summary.foreign_net is in SHARES (converted at the daily VWAP), foreign_trading is in rupiah
 - financial_ratio ROE/ROA are already in percent
The sync state/log files are the same ones the Streamlit app uses (data/autosync_state.json, data/autosync.log).
"""
import atexit
import json
import math
import os
import re
import shutil
import sqlite3
import subprocess
from contextlib import closing
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
IDX_WIB = pd.Timedelta(hours=7)
SQLITE_MAGIC = b"SQLite format 3\x00"
CRON_MARKER = "idx-cron-compat:"
REGISTRY = {}          # Popen handles of the syncs started by this process


# ---------------------------------------------------------------- location / connection
def find_db(explicit=None):
    """The live database <IDX-API>/data/database.sqlite (the sample snapshot is never updated by a sync, so it is not used)."""
    cands = [Path(str(explicit).strip().strip('"'))] if explicit else []
    for base in (HERE.parent, Path.cwd(), Path.home() / "Documents", Path.home()):
        for rel in ("Documents/IDX-API/data/database.sqlite", "IDX-API/data/database.sqlite", "data/database.sqlite"):
            cands.append(base / rel)
    for c in cands:
        try:
            c = c.expanduser().resolve()
            if c.is_file():
                with open(c, "rb") as fh:
                    if fh.read(16).startswith(SQLITE_MAGIC):
                        return c
        except OSError:
            continue
    return None


def connect(path):
    p = Path(path).expanduser().resolve()
    try:
        return sqlite3.connect(f"{p.as_uri()}?mode=ro", uri=True, timeout=5)
    except sqlite3.Error:
        return sqlite3.connect(str(p), timeout=5)


def to_datetime(values):
    """IDX-API integer dates -> naive Timestamps (epoch ms / s, YYYYMMDD, ISO strings)."""
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


def _norm(x):
    return "".join(ch for ch in str(x).lower() if ch.isalnum())


def _num(v):
    v = pd.to_numeric(pd.Series([v]), errors="coerce").iloc[0]
    return float(v) if pd.notna(v) else None


def clean(x):
    """JSON-safe: NaN/inf -> None, numpy scalars -> python, Timestamps -> YYYY-MM-DD."""
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, pd.Timestamp):
        return x.strftime("%Y-%m-%d")
    if isinstance(x, (np.floating, float)):
        return None if not math.isfinite(float(x)) else float(x)
    if isinstance(x, np.integer):
        return int(x)
    return x


TABLES = {"profile": "company_profile", "foreign": "foreign_trading", "stock": "stock_summary",
          "ratio": "financial_ratio", "dividend": "company_dividend"}


class IDXAPI:
    """Read-only, schema-aware adapter. Every query opens its own short-lived connection (safe across threads)."""

    def __init__(self, db_path):
        self.db_path = str(db_path)
        self.tables, self._schemas, self.notes = [], {}, []
        try:
            with closing(connect(self.db_path)) as c:
                self.tables = [r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'").fetchall()]
                for t in self.tables:
                    self._schemas[t] = [r[1] for r in c.execute(f'PRAGMA table_info("{t}")').fetchall()]
        except Exception as e:
            self.notes.append(f"open failed: {type(e).__name__}: {e}")
            self.tables, self._schemas = [], {}

    @property
    def available(self):
        return bool(self.tables)

    def _query(self, sql, params=()):
        try:
            with closing(connect(self.db_path)) as c:
                return pd.read_sql_query(sql, c, params=params)
        except Exception as e:
            self.notes = (self.notes + [f"query failed: {type(e).__name__}: {e}"])[-20:]
            return pd.DataFrame()

    def _table(self, key):
        n = _norm(TABLES[key])
        for t in self._schemas:
            if _norm(t) == n:
                return t
        for t in self._schemas:
            if n in _norm(t):
                return t
        return None

    def _col(self, table, *names):
        norm = {_norm(c): c for c in self._schemas.get(table, [])}
        for nm in names:
            if _norm(nm) in norm:
                return norm[_norm(nm)]
        return None

    @staticmethod
    def _tk(ticker):
        return str(ticker).upper().replace(".JK", "").replace("IDX:", "").strip()

    @staticmethod
    def _recent(df, days):
        df = df.sort_values("Date")
        latest = df["Date"].max()
        win = df[df["Date"] >= latest - pd.Timedelta(days=int(days * 7 / 5) + 3)].tail(days)
        return win, latest

    def company_name(self, ticker):
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
        """[{d, amt}] sorted by ex-date (cash dividend per share)."""
        t = self._table("dividend")
        if not t:
            return []
        cc, xc, vc = self._col(t, "code", "ticker", "symbol"), self._col(t, "ex_dividend"), self._col(t, "cash_dividend")
        if not (cc and xc and vc):
            return []
        d = self._query(f'SELECT "{xc}", "{vc}" FROM "{t}" WHERE UPPER("{cc}") = ?', (self._tk(ticker),))
        if d.empty:
            return []
        s = pd.Series(pd.to_numeric(d[vc], errors="coerce").values, index=pd.DatetimeIndex(to_datetime(d[xc]).values))
        s = s[s.index.notna() & s.notna()].groupby(level=0).sum().sort_index()
        return [{"d": ix.strftime("%Y-%m-%d"), "amt": float(v)} for ix, v in s.items()]

    def foreign_flow(self, ticker, days=20, max_age_days=10, market=True):
        """(stock_bn, market_bn, info, problems): Rp billions, net over the last `days` sessions."""
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
                    d["Date"] = to_datetime(d[dc]).values
                    win, latest = self._recent(d.dropna(subset=["Date"]), days)
                    if (today - latest).days > max_age_days:
                        problems.append(f"stock flow STALE: last {tk} row is {latest:%Y-%m-%d} ({(today - latest).days} days old)")
                    elif len(win) < need:
                        problems.append(f"stock flow: only {len(win)} of {days} sessions for {tk}")
                    else:
                        vol, val, close = (pd.to_numeric(win[c], errors="coerce") for c in (vo, va, cl))
                        vwap = (val / vol.where(vol > 0)).fillna(close)
                        fs = float((pd.to_numeric(win[fn], errors="coerce") * vwap).sum() / 1e9)
                        info.append(f"{tk} stock flow Rp {fs:+,.0f} bn over {len(win)} sessions to {latest:%Y-%m-%d}")
            else:
                problems.append("stock flow: stock_summary lacks foreign_net/volume/value/close")
        else:
            problems.append("stock flow: stock_summary table not found")
        if not market:
            return fs, None, "; ".join(info), "; ".join(problems)

        t = self._table("foreign")
        if t:
            dc, bv, sv = self._col(t, "date"), self._col(t, "buy_value"), self._col(t, "sell_value")
            if all((dc, bv, sv)):
                d = self._query(f'SELECT "{dc}", "{bv}", "{sv}" FROM "{t}" ORDER BY "{dc}" DESC LIMIT ?', (int(days * 3),))
                if len(d):
                    d["Date"] = to_datetime(d[dc]).values
                    win, latest = self._recent(d.dropna(subset=["Date"]), days)
                    if (today - latest).days > max_age_days:
                        problems.append(f"market flow STALE: foreign_trading ends {latest:%Y-%m-%d}")
                    elif len(win) < need:
                        problems.append(f"market flow: only {len(win)} of {days} sessions in foreign_trading")
                    else:
                        net = pd.to_numeric(win[bv], errors="coerce") - pd.to_numeric(win[sv], errors="coerce")
                        fm = float(net.sum() / 1e9)
                        info.append(f"market flow Rp {fm:+,.0f} bn over {len(win)} sessions to {latest:%Y-%m-%d} (foreign_trading)")
        if fm is None and self._table("stock"):   # fallback: add up every stock's net foreign
            t = self._table("stock")
            cc, dc = self._col(t, "code"), self._col(t, "date")
            fn, vo, va, cl = self._col(t, "foreign_net"), self._col(t, "volume"), self._col(t, "value"), self._col(t, "close")
            if all((cc, dc, fn, vo, va, cl)):
                cutoff = int((today - pd.Timedelta(days=int(days * 7 / 5) + 10)).timestamp() * 1000)
                d = self._query(f'SELECT "{dc}", "{fn}", "{vo}", "{va}", "{cl}" FROM "{t}" WHERE "{dc}" >= ?', (cutoff,))
                if len(d):
                    d["Date"] = to_datetime(d[dc]).values
                    vol, val, close = (pd.to_numeric(d[c], errors="coerce") for c in (vo, va, cl))
                    d["rp"] = pd.to_numeric(d[fn], errors="coerce") * (val / vol.where(vol > 0)).fillna(close)
                    g = d.dropna(subset=["Date"]).groupby("Date").agg(rp=("rp", "sum"), n=("rp", "size")).reset_index()
                    win, latest = self._recent(g, days)
                    if len(win) >= need and (today - latest).days <= max_age_days and win["n"].median() >= 300:
                        fm = float(win["rp"].sum() / 1e9)
                        info.append(f"market flow Rp {fm:+,.0f} bn (sum of {int(win['n'].median())} stocks)")
                    elif len(win):
                        problems.append(f"market flow fallback not used: {int(win['n'].median())} stocks/day (needs >= 300)")
        return fs, fm, "; ".join(info), "; ".join(problems)

    def financial_ratio(self, ticker, max_age_months=15):
        t = self._table("ratio")
        if not t:
            return {}
        cc, pc = self._col(t, "code", "ticker", "symbol"), self._col(t, "period")
        if not (cc and pc):
            return {}
        d = self._query(f'SELECT * FROM "{t}" WHERE UPPER("{cc}") = ?', (self._tk(ticker),))
        if d.empty:
            return {}
        d["__p"] = to_datetime(d[pc]).values
        d = d.dropna(subset=["__p"]).sort_values("__p")
        if d.empty:
            return {}
        last, period = d.iloc[-1], d["__p"].iloc[-1]
        age_m = (pd.Timestamp.now().normalize() - period).days / 30.4
        if age_m > max_age_months:
            return {"stale": f"latest period {period:%Y-%m-%d} is {age_m:.0f} months old"}
        out = {}
        for key in ("roe", "roa", "pbv", "per", "eps"):
            c = self._col(t, key)
            if c:
                v = _num(last[c])
                if v is not None:
                    out[key] = v
        ec = self._col(t, "eps")
        if ec and out.get("eps") is not None:   # YoY on the same cumulative basis
            prev = d[(d["__p"] >= period - pd.Timedelta(days=380)) & (d["__p"] <= period - pd.Timedelta(days=350))]
            pe = _num(prev[ec].iloc[-1]) if len(prev) else None
            if pe and pe > 0 and out["eps"] > 0:
                out["eps_growth"] = (out["eps"] / pe - 1) * 100
        out["period"] = period.strftime("%Y-%m-%d")
        out["basis"] = "full-year" if period.month == 12 else f"{period.month}-month cumulative"
        return out


def latest_stock_date(db_path):
    """Newest date in stock_summary (epoch ms + 7 h = WIB day), or None."""
    try:
        with closing(connect(db_path)) as c:
            mx = c.execute('SELECT MAX("date") FROM "stock_summary"').fetchone()[0]
        return None if mx is None else (pd.Timestamp(int(mx), unit="ms") + IDX_WIB).normalize()
    except Exception:
        return None


def snapshot(db_path, codes, days=20):
    """Everything the UI shows from IDX-API for `codes`: {info: {CODE: {...}}, market: {...}, db: {...}}."""
    a = IDXAPI(db_path)
    latest = latest_stock_date(db_path)
    db = {"path": str(db_path), "latest": latest.strftime("%Y-%m-%d") if latest is not None else None, "tables": len(a.tables)}
    if not a.available:
        return {"info": {}, "market": {}, "db": db}
    info, market = {}, {}
    for code in codes:
        code = a._tk(code)
        first = not market
        fs, fm, msg, prob = a.foreign_flow(code, days, market=first)
        if first:
            market = {"flow_bn": fm, "info": msg, "problems": prob, "days": days}
        item = {"name": a.company_name(code), "dividends": a.dividends(code), "ratio": a.financial_ratio(code),
                "flow_bn": fs, "flow_info": msg if fs is not None else "", "flow_problem": prob if fs is None else ""}
        info[code] = item
    return clean({"info": info, "market": market, "db": db})


# ---------------------------------------------------------------- background sync (Cron.ts)
# The sync must live and die with this Python process:
#  - normal exit / Ctrl+C / SIGTERM / console close -> stop_all() (atexit + the server's shutdown path) kills the Deno process tree
#  - Python killed hard (Task Manager, taskkill /F)  -> on Windows the Deno process is inside a Job Object with
#    KILL_ON_JOB_CLOSE, so the OS kills it the moment this process disappears
_JOB = None


def _kill_on_close_job():
    """A Windows Job Object that kills every process assigned to it when the last handle (ours) closes. None elsewhere."""
    global _JOB
    if os.name != "nt":
        return None
    if _JOB is not None:
        return _JOB
    try:
        import ctypes
        from ctypes import wintypes

        class IO_COUNTERS(ctypes.Structure):
            _fields_ = [(n, ctypes.c_ulonglong) for n in ("ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                                                           "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

        class BASIC(ctypes.Structure):
            _fields_ = [("PerProcessUserTimeLimit", ctypes.c_longlong), ("PerJobUserTimeLimit", ctypes.c_longlong),
                        ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                        ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                        ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD), ("SchedulingClass", wintypes.DWORD)]

        class EXT(ctypes.Structure):
            _fields_ = [("BasicLimitInformation", BASIC), ("IoInfo", IO_COUNTERS), ("ProcessMemoryLimit", ctypes.c_size_t),
                        ("JobMemoryLimit", ctypes.c_size_t), ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.CreateJobObjectW.restype = wintypes.HANDLE
        k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        job = k32.CreateJobObjectW(None, None)
        info = EXT()
        info.BasicLimitInformation.LimitFlags = 0x2000          # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not k32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info)):   # 9 = JobObjectExtendedLimitInformation
            return None
        _JOB = (job, k32)
    except Exception:
        _JOB = None
    return _JOB


def _bind_to_this_process(proc):
    job = _kill_on_close_job()
    if job and getattr(proc, "_handle", None):
        try:
            job[1].AssignProcessToJobObject(job[0], int(proc._handle))
        except Exception:
            pass


def stop_all():
    """Stop every sync this process started (whole process tree) and mark it stopped, so the next start does not think it still runs."""
    for folder, proc in list(REGISTRY.items()):
        try:
            if proc.poll() is None:
                if os.name == "nt":
                    subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15)
                else:
                    proc.terminate()
                try:
                    proc.wait(timeout=10)
                except Exception:
                    proc.kill()
                data_dir = Path(folder) / "data"
                state_file = data_dir / "autosync_state.json"
                try:
                    st_ = json.loads(state_file.read_text(encoding="utf-8"))
                except Exception:
                    st_ = {}
                st_.update(stopped=True, stopped_at=pd.Timestamp.now(tz="Asia/Jakarta").isoformat())
                state_file.write_text(json.dumps(st_, ensure_ascii=False), encoding="utf-8")
                with open(data_dir / "autosync.log", "a", encoding="utf-8") as fh:
                    fh.write("\n[autosync] stopped because the IDX Trend Studio server was stopped\n")
        except Exception:
            pass
    REGISTRY.clear()


atexit.register(stop_all)
def expected_last_session(now_wib=None, close_hour=17):
    """The last trading date the database should already hold (today once the session closed, else the previous weekday)."""
    now = now_wib if now_wib is not None else pd.Timestamp.now(tz="Asia/Jakarta")
    d = pd.Timestamp(now.year, now.month, now.day)
    if not (now.weekday() < 5 and now.hour >= close_hour):
        d -= pd.Timedelta(days=1)
    while d.weekday() >= 5:
        d -= pd.Timedelta(days=1)
    return d


class AutoSync:
    """Runs `deno run -A Cron.ts` in the background and reports its progress from the log file."""

    def __init__(self, db_path):
        self.db = Path(db_path)
        self.data_dir = self.db.parent
        self.folder = self.data_dir.parent
        self.cron = self.folder / "Cron.ts"
        self.state_file = self.data_dir / "autosync_state.json"
        self.log_file = self.data_dir / "autosync.log"
        self.registry = REGISTRY

    def deno(self):
        found = shutil.which("deno")
        if found:
            return found
        for c in (Path.home() / ".deno" / "bin" / "deno.exe", Path.home() / ".deno" / "bin" / "deno"):
            if c.is_file():
                return str(c)
        return None

    def readiness(self):
        if self.db.name != "database.sqlite" or self.data_dir.name != "data":
            return False, f"sync writes to <IDX-API>/data/database.sqlite, not {self.db}"
        if not self.cron.is_file():
            return False, f"Cron.ts not found in {self.folder}"
        try:
            head = self.cron.read_text(encoding="utf-8", errors="ignore")[:4000]
        except OSError as e:
            return False, f"cannot read Cron.ts: {e}"
        if CRON_MARKER not in head:
            return False, "Cron.ts is the original (downloads all ~960 companies): replace it with the modified Cron.ts"
        if not self.deno():
            return False, "Deno not found (install from deno.com or add it to PATH)"
        return True, "ready"

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
        st_ = self._read_state()
        if not st_:
            return {"state": "idle"}
        if st_.get("stopped") and self.registry.get(str(self.folder)) is None:
            return {"state": "stopped", "done": 0, "total": None, "last_line": "stopped with the server", "prices_to": None,
                    "started_at": st_.get("started_at"), "days": st_.get("days"), "exit_code": None}
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
            state = "failed"
        elif age_min < 3:
            state = "running"      # started by another process (e.g. the Streamlit app) and still writing the log
        else:
            state = "stalled"
        return {"state": state, "done": done, "total": total, "last_line": lines[-1][:200] if lines else "",
                "prices_to": ok_m.group(1) if ok_m else None, "started_at": st_.get("started_at"), "days": st_.get("days"),
                "exit_code": proc.poll() if proc is not None else None}

    def should_auto_update(self, min_hours=0.5, now_wib=None):
        """(start?, reason, days): only when the database is behind the last session, nothing runs, and the last attempt is old enough."""
        ok, msg = self.readiness()
        if not ok:
            return False, msg, 0
        if self.status()["state"] == "running":
            return False, "a sync is already running", 0
        now = now_wib if now_wib is not None else pd.Timestamp.now(tz="Asia/Jakarta")
        latest, expected = latest_stock_date(self.db), expected_last_session(now)
        if latest is not None and latest >= expected:
            return False, f"database current (prices to {latest:%Y-%m-%d})", 0
        prev = self._read_state()
        if prev and prev.get("started_at") and not prev.get("stopped"):   # an attempt cut short by stopping the server does not count
            try:
                hrs = (now - pd.Timestamp(prev["started_at"])).total_seconds() / 3600
                if hrs < min_hours:
                    return False, f"last attempt {hrs:.1f} h ago (minimum {min_hours} h)", 0
            except Exception:
                pass
        if latest is None:
            return True, "database has no prices yet", 365
        gap = (expected - latest).days
        return True, f"prices to {latest:%Y-%m-%d}, last session {expected:%Y-%m-%d}", int(min(max(gap + 10, 14), 420))

    def start(self, days, per_company=False, trigger="auto"):
        ok, msg = self.readiness()
        if not ok:
            return False, msg
        if self.status()["state"] == "running":
            return False, "a sync is already running"
        cmd = [self.deno(), "run", "-A", "Cron.ts", f"--days={int(days)}"] + ([] if per_company else ["--no-per-company"])
        self._write_state(started_at=pd.Timestamp.now(tz="Asia/Jakarta").isoformat(), days=int(days), trigger=trigger, cmd=cmd)
        kw = dict(cwd=str(self.folder), stdin=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        if os.name == "nt":
            kw["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)   # same console group: it dies with us, no detaching
        try:
            with open(self.log_file, "w", encoding="utf-8") as fh:
                proc = subprocess.Popen(cmd, stdout=fh, **kw)
        except OSError as e:
            return False, f"could not start Deno: {e}"
        self.registry[str(self.folder)] = proc
        _bind_to_this_process(proc)
        return True, f"started ({' '.join(cmd[1:])})"
