#!/bin/bash
# Collects everything needed to see why the auto start does not work. Run it and paste the whole output.
#   ./diagnose.sh
LABEL="com.idxtrendstudio"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKDIR="$(cd "$HERE/.." && pwd)"
PROJECT="$(cd "$WORKDIR/.." && pwd)"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
PY="$PROJECT/.venv/bin/python"
sec() { printf '\n===== %s =====\n' "$1"; }

sec "system"; sw_vers 2>/dev/null; uname -m; echo "user: $(whoami)  uid: $(id -u)"
sec "folders"; echo "workdir: $WORKDIR"; ls "$WORKDIR" | head -n 20; echo "project: $PROJECT"; ls "$PROJECT" | head -n 30
sec "python"; ls -l "$PY" 2>&1; "$PY" --version 2>&1
sec "libraries"; "$PY" -c "
import importlib
for m in ('pandas','numpy','yfinance','sklearn','xgboost','statsmodels','streamlit','plotly','scipy'):
    try:
        importlib.import_module(m); print('ok     ', m)
    except Exception as e:
        print('FAILED ', m, '->', str(e).splitlines()[0][:150])
" 2>&1
sec "plist"; ls -l "$PLIST" 2>&1; plutil -lint "$PLIST" 2>&1; cat "$PLIST" 2>&1
sec "launchctl"; launchctl print "gui/$(id -u)/$LABEL" 2>&1 | head -n 40
sec "port 8000"; (lsof -nP -iTCP:8000 -sTCP:LISTEN 2>&1 || true); curl -s -m 5 -o /dev/null -w "HTTP %{http_code}\n" http://localhost:8000/ 2>&1
sec "server.log (last 40 lines)"; tail -n 40 "$WORKDIR/server.log" 2>&1
sec "run the server by hand for 12 seconds (shows start-up errors)"
cd "$WORKDIR" || exit 1
( "$PY" server.py --app off --no-browser --port 8799 2>&1 & PID=$!; sleep 12; curl -s -m 5 -o /dev/null -w "HTTP %{http_code} on :8799\n" http://localhost:8799/ 2>&1; kill $PID 2>/dev/null; wait $PID 2>/dev/null ) | tail -n 40
echo; echo "===== done: paste everything above ====="
