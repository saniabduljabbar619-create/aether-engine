#!/usr/bin/env bash
# Aether Edge AI Engine - Unix Launcher
echo "========================================================"
echo "  Starting Aether Edge AI Engine Server (FastAPI + Web)"
echo "========================================================"

if [ -f ".venv/bin/python" ]; then
    PYTHON_CMD=".venv/bin/python"
else
    PYTHON_CMD="python3"
fi

$PYTHON_CMD main.py --server --port 8000
