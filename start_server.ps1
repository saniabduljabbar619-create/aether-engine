# Aether Edge AI Engine - PowerShell Server Launcher
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "  Starting Aether Edge AI Engine Server (FastAPI + Web)" -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Web Portal:  http://localhost:8000" -ForegroundColor Green
Write-Host "API Docs:    http://localhost:8000/docs" -ForegroundColor Green
Write-Host "Live Stream: http://localhost:8000/api/stream" -ForegroundColor Green
Write-Host ""

if (Test-Path ".\.venv\Scripts\python.exe") {
    $pythonCmd = ".\.venv\Scripts\python.exe"
} else {
    $pythonCmd = "python"
}

& $pythonCmd main.py --server --no-voice --port 8000
