#!/bin/bash
# Stop IDX Trend Studio and remove it from the macOS login items.
# The project files, the .venv and the saved Deep Forecast results (idx_trend_studio/cache) are not touched.
set -uo pipefail

LABEL="com.idxtrendstudio"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"

if command -v launchctl >/dev/null 2>&1; then
  # bootout sends SIGTERM: server.py then stops the IDX sync and Streamlit that it started
  launchctl bootout "gui/$(id -u)/$LABEL" >/dev/null 2>&1 && echo "Stopped." || echo "It was not running."
fi
if [ -f "$PLIST" ]; then
  rm -f "$PLIST"
  echo "Removed $PLIST"
else
  echo "Nothing installed ($PLIST does not exist)."
fi
