# Aether Edge AI Engine - PowerShell GUI Launcher
Write-Host "========================================================" -ForegroundColor Green
Write-Host "  Starting Aether Desktop UI (PySide6)" -ForegroundColor Green
Write-Host "========================================================" -ForegroundColor Green
Write-Host ""

if (Test-Path ".\.venv\Scripts\python.exe") {
    $pythonCmd = ".\.venv\Scripts\python.exe"
} else {
    $pythonCmd = "python"
}

& $pythonCmd main.py --gui
