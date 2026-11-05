@echo off
title Aether Edge AI Engine - API Server & Web Portal
echo ========================================================
echo   Starting Aether Edge AI Engine Server (FastAPI + Web)
echo ========================================================
echo.

if exist .venv\Scripts\python.exe (
    set PYTHON_CMD=.venv\Scripts\python.exe
) else (
    set PYTHON_CMD=python
)

%PYTHON_CMD% main.py --server --port 8000
pause
