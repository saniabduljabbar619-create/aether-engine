#!/usr/bin/env bash
# Aether Edge AI Engine - Unix Launcher
echo "========================================================"
echo "  Starting Aether Edge AI Engine Server (FastAPI + Web)"
echo "========================================================"
echo ""
echo "Web Portal:  http://localhost:8000"
echo "API Docs:    http://localhost:8000/docs"
echo "Live Stream: http://localhost:8000/api/stream"
echo ""

if [ -f ".venv/bin/python" ]; then
    PYTHON_CMD=".venv/bin/python"
else
    PYTHON_CMD="python3"
fi

$PYTHON_CMD main.py --server --no-voice --port 8000
