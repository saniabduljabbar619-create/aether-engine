# Aether Edge AI Engine - PowerShell Server Launcher
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "  Starting Aether Edge AI Engine Server (FastAPI + Web)" -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host ""

if (Test-Path ".\.venv\Scripts\python.exe") {
    $pythonCmd = ".\.venv\Scripts\python.exe"
} else {
    $pythonCmd = "python"
}

& $pythonCmd main.py --server --port 8000
