@echo off
cd /d C:\Users\Thinkpad\Documents\Python\stock_forecast\idx_trend_studio
rem HTML service only (no Streamlit). To start Streamlit as well, use: python server.py --app latest
python server.py --app off >> server.log 2>&1
