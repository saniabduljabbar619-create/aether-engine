@echo off
title Aether Edge AI Engine - Desktop Application
echo ========================================================
echo   Starting Aether Desktop UI (PySide6)
echo ========================================================
echo.

if exist .venv\Scripts\python.exe (
    set PYTHON_CMD=.venv\Scripts\python.exe
) else (
    set PYTHON_CMD=python
)

%PYTHON_CMD% main.py --gui
pause
