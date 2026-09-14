<#
.SYNOPSIS
  Check a classroom node over SSH before installing: GPUs, Docker, the registry, disk.

.EXAMPLE
  .\scripts\offline\check-node.ps1 -Node you@gpu-node -Registry harbor.local/atelier
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Node,
    [string]$Registry
)
$ErrorActionPreference = "Continue"

function Test-Remote([string]$Label, [string]$Command) {
    Write-Host "==> $Label" -ForegroundColor Cyan
    ssh $Node $Command
    if ($LASTEXITCODE -ne 0) { Write-Host "    FAILED" -ForegroundColor Red } else { Write-Host "    ok" -ForegroundColor Green }
}

Test-Remote "GPUs visible to the driver" "nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader"
Test-Remote "Docker and Compose v2" "docker version --format '{{.Server.Version}}' && docker compose version --short"
Test-Remote "driver new enough for CUDA 12.8" "nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1"
Test-Remote "GPUs visible inside a container" "docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi -L"
Test-Remote "disk space under /srv" "df -h /srv | tail -1"
if ($Registry) {
    Test-Remote "registry reachable" "docker pull $Registry/redis:7-alpine >/dev/null && echo pulled"
}
Write-Host ""
Write-Host "Anything red has to be fixed before docker compose up." -ForegroundColor Yellow
Write-Host "CUDA 12.8 needs driver 525.60.13 or newer; 570+ is what NVIDIA ships alongside it." -ForegroundColor Yellow
