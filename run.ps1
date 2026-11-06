# ============================================================
# Expense Copilot — one-command launcher (Windows PowerShell)
#
# Usage:
#   .\run.ps1          # start backend + UI (assumes setup already done)
#   .\run.ps1 -Setup   # first-time: create venv and install dependencies
#
# Open http://localhost:8501 once both windows are up.
# ============================================================
param(
    [switch]$Setup
)

$ErrorActionPreference = "Stop"
$py = ".\.venv\Scripts\python.exe"

if ($Setup -or -not (Test-Path ".\.venv")) {
    Write-Host "==> Creating virtual environment..." -ForegroundColor Cyan
    python -m venv .venv

    Write-Host "==> Installing dependencies (this can take a few minutes)..." -ForegroundColor Cyan
    & $py -m pip install --upgrade pip
    & $py -m pip install -r requirements.txt

}

Write-Host "==> Starting FastAPI backend on http://localhost:8000 ..." -ForegroundColor Green
Start-Process powershell -ArgumentList @(
    "-NoExit", "-Command",
    "& '$py' -m uvicorn app.main:app --reload --port 8000"
)

Start-Sleep -Seconds 3

Write-Host "==> Starting Streamlit UI on http://localhost:8501 ..." -ForegroundColor Green
Start-Process powershell -ArgumentList @(
    "-NoExit", "-Command",
    "& '$py' -m streamlit run frontend/streamlit_app.py --server.port 8501"
)

Write-Host ""
Write-Host "Both services launching in separate windows." -ForegroundColor Yellow
Write-Host "Open the UI at: http://localhost:8501" -ForegroundColor Yellow
Write-Host "API docs at:    http://localhost:8000/docs" -ForegroundColor Yellow
