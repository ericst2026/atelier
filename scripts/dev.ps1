<#
.SYNOPSIS
  Run Atelier on Windows without Docker: SQLite, a local Redis, and the Vite dev server.

.EXAMPLE
  .\scripts\dev.ps1

  Needs Python 3.11+, Node 20+, and a Redis reachable at localhost:6379
  (docker run -d -p 6379:6379 redis:7-alpine).
#>
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
$root = (Get-Location).Path

$env:ATELIER_ROOT = Join-Path $root ".dev"
$env:ATELIER_EXPERIMENTS_DIR = Join-Path $root "experiments"
$env:ATELIER_MATERIALS_DIR = Join-Path $root ".dev\materials"
$env:ATELIER_DATA_DIR = Join-Path $root ".dev\data"
$env:ATELIER_REDIS_URL = if ($env:ATELIER_REDIS_URL) { $env:ATELIER_REDIS_URL } else { "redis://localhost:6379/0" }
$env:ATELIER_GPU_COUNT = "0"
$env:ATELIER_MAX_GPUS_PER_STUDENT_RUN = "0"
$env:PYTHONPATH = Join-Path $root "backend"
New-Item -ItemType Directory -Force -Path $env:ATELIER_DATA_DIR, $env:ATELIER_MATERIALS_DIR | Out-Null

if (-not (Test-Path (Join-Path $root ".venv"))) {
    Write-Host "creating .venv and installing the backend requirements…"
    python -m venv .venv
    & .\.venv\Scripts\pip.exe install -q -r backend\requirements.txt
}
$py = Join-Path $root ".venv\Scripts\python.exe"

if (-not (Test-Path (Join-Path $root "frontend\node_modules"))) {
    Write-Host "installing the frontend dependencies…"
    Push-Location frontend; npm install --no-audit --no-fund; Pop-Location
}

Write-Host "starting api, worker and the dev server…"
$api    = Start-Process -PassThru -NoNewWindow -WorkingDirectory (Join-Path $root "backend") -FilePath $py -ArgumentList "-m","uvicorn","atelier.main:app","--reload","--port","8000"
$worker = Start-Process -PassThru -NoNewWindow -WorkingDirectory (Join-Path $root "backend") -FilePath $py -ArgumentList "-m","atelier.worker"
$web    = Start-Process -PassThru -NoNewWindow -WorkingDirectory (Join-Path $root "frontend") -FilePath "npm" -ArgumentList "run","dev"

Write-Host ""
Write-Host "  frontend  http://localhost:5173"
Write-Host "  api docs  http://localhost:8000/api/docs"
Write-Host "  sign in   teacher / teacher"
Write-Host ""
Write-Host "Ctrl+C to stop everything."
try { Wait-Process -Id $api.Id } finally {
    foreach ($p in @($api, $worker, $web)) {
        if ($p -and -not $p.HasExited) { Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue }
    }
}
