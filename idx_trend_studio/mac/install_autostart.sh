#!/bin/bash
# Install IDX Trend Studio as a macOS login item (launchd user agent).
#
#   ./install_autostart.sh                 HTML UI only (default), no browser tab at login
#   ./install_autostart.sh --app latest    also start the newest stock_forecast_app_*.py with Streamlit (or a version: 1.5.0 | 1.4)
#   ./install_autostart.sh --browser       open the UI in the browser at every login
#   ./install_autostart.sh --db PATH       path of the IDX-API database.sqlite (when it is not in ~/Documents/IDX-API)
#   ./install_autostart.sh --python PATH   use this Python instead of the project's .venv
#
# It (1) creates <project>/.venv and installs requirements.txt when no Python is given and there is no .venv yet,
# (2) writes ~/Library/LaunchAgents/com.idxtrendstudio.plist, (3) loads it, so the server also starts now.
# Remove it again with ./uninstall_autostart.sh

set -euo pipefail

LABEL="com.idxtrendstudio"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"          # .../idx_trend_studio/mac
WORKDIR="$(cd "$HERE/.." && pwd)"                               # .../idx_trend_studio
PROJECT="$(cd "$WORKDIR/.." && pwd)"                            # .../stock_forecast (stock_forecast_app_*.py, deep_forecast.py, requirements.txt)
TEMPLATE="$HERE/$LABEL.plist"
AGENTS="$HOME/Library/LaunchAgents"
PLIST="$AGENTS/$LABEL.plist"
LOG="$WORKDIR/server.log"

APP="off"; OPEN_BROWSER=0; DB=""; PYTHON=""
while [ $# -gt 0 ]; do
  case "$1" in
    --app)     APP="${2:?--app needs a value (off | latest | 1.5.0)}"; shift 2 ;;
    --browser) OPEN_BROWSER=1; shift ;;
    --db)      DB="${2:?--db needs a path}"; shift 2 ;;
    --python)  PYTHON="${2:?--python needs a path}"; shift 2 ;;
    -h|--help) sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "Unknown option: $1 (try --help)"; exit 1 ;;
  esac
done
[ "$APP" = "ask" ] && { echo "--app ask needs a terminal and cannot be used at login. Use off | latest | a version."; exit 1; }

[ -f "$TEMPLATE" ]            || { echo "Template not found: $TEMPLATE"; exit 1; }
[ -f "$WORKDIR/server.py" ]   || { echo "server.py not found in $WORKDIR"; exit 1; }

# ---------------------------------------------------------------- Python
if [ -z "$PYTHON" ]; then
  PYTHON="$PROJECT/.venv/bin/python"
  if [ ! -x "$PYTHON" ]; then
    BASE_PY="$(command -v python3 || true)"
    [ -n "$BASE_PY" ] || { echo "python3 not found. Install it first, e.g.: brew install python"; exit 1; }
    echo "Creating the virtual environment $PROJECT/.venv ..."
    "$BASE_PY" -m venv "$PROJECT/.venv"
    "$PYTHON" -m pip install --upgrade pip
    if [ -f "$PROJECT/requirements.txt" ]; then
      "$PYTHON" -m pip install -r "$PROJECT/requirements.txt"
    else
      "$PYTHON" -m pip install streamlit yfinance pandas numpy scipy plotly scikit-learn xgboost statsmodels
    fi
  fi
fi
[ -x "$PYTHON" ] || { echo "Python not executable: $PYTHON"; exit 1; }

# the ML libraries must import, otherwise Deep Forecast is unavailable (XGBoost on a Mac needs OpenMP: brew install libomp)
if ! "$PYTHON" -c "import sklearn, xgboost, statsmodels, yfinance, pandas" 2>/tmp/idx_import_check.txt; then
  echo "WARNING: a library does not import with $PYTHON:"; tail -n 3 /tmp/idx_import_check.txt
  echo "         if it mentions libomp:  brew install libomp"
fi
command -v deno >/dev/null 2>&1 || [ -x "$HOME/.deno/bin/deno" ] || echo "Note: Deno not found. The IDX sync is disabled until you install it (brew install deno)."

# ---------------------------------------------------------------- the plist
xe() { printf '%s' "$1" | sed -e 's/&/\&amp;/g' -e 's/</\&lt;/g' -e 's/>/\&gt;/g'; }      # XML-escape a value (a path may contain & < >)
ARGS="    <string>--app</string><string>$(xe "$APP")</string>"
[ "$OPEN_BROWSER" -eq 0 ] && ARGS="$ARGS"$'\n'"    <string>--no-browser</string>"
[ -n "$DB" ] && ARGS="$ARGS"$'\n'"    <string>--db</string><string>$(xe "$DB")</string>"
AGENT_PATH="/opt/homebrew/bin:/opt/homebrew/sbin:/usr/local/bin:$HOME/.deno/bin:/usr/bin:/bin:/usr/sbin:/sbin"

mkdir -p "$AGENTS"
# the values are put into the file with a literal replace (no regex / '&' / '\' surprises with special characters in a path)
awk -v v_py="$(xe "$PYTHON")" -v v_wd="$(xe "$WORKDIR")" -v v_log="$(xe "$LOG")" -v v_path="$(xe "$AGENT_PATH")" -v v_args="$ARGS" '
  function lit(s, key, val,   out, i) {
    out = ""
    while ((i = index(s, key)) > 0) { out = out substr(s, 1, i - 1) val; s = substr(s, i + length(key)) }
    return out s
  }
  /^__ARGS__$/ { print v_args; next }
  { $0 = lit($0, "__PYTHON__", v_py); $0 = lit($0, "__WORKDIR__", v_wd); $0 = lit($0, "__LOG__", v_log); $0 = lit($0, "__PATH__", v_path); print }
' "$TEMPLATE" > "$PLIST"

if command -v plutil >/dev/null 2>&1; then plutil -lint "$PLIST" >/dev/null || { echo "The generated plist is not valid: $PLIST"; exit 1; }; fi

# ---------------------------------------------------------------- load it (also starts the server now)
if command -v launchctl >/dev/null 2>&1; then
  DOMAIN="gui/$(id -u)"
  launchctl bootout "$DOMAIN/$LABEL" >/dev/null 2>&1 || true     # replace an older version
  launchctl bootstrap "$DOMAIN" "$PLIST"
  launchctl enable "$DOMAIN/$LABEL" >/dev/null 2>&1 || true
  launchctl kickstart -k "$DOMAIN/$LABEL" >/dev/null 2>&1 || true
  echo "Installed and started."
else
  echo "launchctl not found (this is not a Mac?). The plist was only written."
fi

cat <<EOF

  Label        $LABEL
  Plist        $PLIST
  Python       $PYTHON
  Working dir  $WORKDIR
  Streamlit    $APP
  Log          $LOG
  UI           http://localhost:8000

  Status:      launchctl print gui/$(id -u)/$LABEL | head -n 20
  Follow log:  tail -f "$LOG"
  Restart:     launchctl kickstart -k gui/$(id -u)/$LABEL
  Remove:      "$HERE/uninstall_autostart.sh"
EOF
