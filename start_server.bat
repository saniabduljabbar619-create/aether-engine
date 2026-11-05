@echo off
title Aether Edge AI Engine - API Server & Web Portal
echo ========================================================
echo   Starting Aether Edge AI Engine Server (FastAPI + Web)
echo ========================================================
echo.
echo Web Portal:  http://localhost:8000
echo API Docs:    http://localhost:8000/docs
echo Live Stream: http://localhost:8000/api/stream
echo.

if exist .venv\Scripts\python.exe (
    set PYTHON_CMD=.venv\Scripts\python.exe
) else (
    set PYTHON_CMD=python
)

%PYTHON_CMD% main.py --server --no-voice --port 8000
pause
