@echo off
title Hybrid IR and Question Answering - 23MID0429
echo =====================================================================
echo CSI4001: Natural Language Processing and Computational Linguistics
echo Assessment: Case Study 1 (Question 29)
echo Student: Vinay Vishwanathan ^| Register Number: 23MID0429
echo Project: Hybrid Transformer-Based IR and Question Answering System
echo =====================================================================
echo.

cd /d "%~dp0"

IF NOT EXIST ".venv\Scripts\python.exe" (
    echo [INFO] Virtual environment not detected. Creating virtual environment...
    python -m venv .venv
    echo [INFO] Installing required dependencies...
    .\.venv\Scripts\pip install -r requirements.txt
)

echo [INFO] Launching Streamlit web application...
echo [INFO] Access the web UI at: http://localhost:8501
echo.
.\.venv\Scripts\streamlit.exe run app/streamlit_app.py

pause
