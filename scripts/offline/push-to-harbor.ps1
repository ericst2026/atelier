<#
.SYNOPSIS
  Build Atelier's images on Windows, pull the third-party ones, retag everything
  for your own registry, and push. Run this on a machine with internet; the
  classroom node then only needs to reach the registry.

.EXAMPLE
  .\scripts\offline\push-to-harbor.ps1 -Registry harbor.local -Project atelier

.EXAMPLE
  # only re-push the three Atelier images after a code change
  .\scripts\offline\push-to-harbor.ps1 -Registry harbor.local -Project atelier -Skip Third

.EXAMPLE
  # a newer PyTorch
  .\scripts\offline\push-to-harbor.ps1 -Registry harbor.local -Project atelier -TorchVersion 2.10.0

.EXAMPLE
  # a class that includes older cards (GTX 10-series, V100)
  .\scripts\offline\push-to-harbor.ps1 -Registry harbor.local -Project atelier `
      -CudaImage nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04 -CudaWheel cu126 -CudaVersion 12.6

.NOTES
  Needs Docker Desktop running and `docker login harbor.local` to have succeeded.
  If Harbor uses a self-signed certificate, add it under
  Docker Desktop → Settings → Docker Engine → "insecure-registries": ["harbor.local"]
  or install the CA on the machine. Harbor must have the project created first —
  it does not create projects on push.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Registry,          # harbor.local
    [string]$Project = "atelier",                             # the Harbor project
    [string]$Tag = "latest",
    [ValidateSet("None", "Atelier", "Third")][string]$Skip = "None",
    [switch]$NoBuild,                                         # retag and push what is already local
    # The runtime the worker image is built with. These four have to agree —
    # see docs/runtime.md. Defaults suit Ampere and newer.
    [string]$CudaImage = "nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04",
    [string]$CudaWheel = "cu128",
    [string]$CudaVersion = "12.8",
    [string]$TorchVersion = "2.8.0",
    [string]$PythonVersion = "3.12"
)
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..\..")
$prefix = "$Registry/$Project"

# name → how it is built, or $null when it is pulled
$atelier = [ordered]@{
    "atelier-api"    = @{ Dockerfile = "backend\Dockerfile";        Context = "backend" }
    "atelier-worker" = @{ Dockerfile = "backend\Dockerfile.worker"; Context = "." }
    "atelier-web"    = @{ Dockerfile = "frontend\Dockerfile";       Context = "frontend" }
}
$third = @(
    "postgres:16-alpine",
    "redis:7-alpine",
    "prom/prometheus:v2.54.1",
    "prom/node-exporter:v1.8.2",
    "grafana/grafana:11.2.0"
)

function Invoke-Step([string]$Message, [scriptblock]$Action) {
    Write-Host "==> $Message" -ForegroundColor Cyan
    & $Action
    if ($LASTEXITCODE -ne 0) { throw "$Message failed (exit $LASTEXITCODE)" }
}

Write-Host "registry prefix: $prefix" -ForegroundColor Yellow
docker info | Out-Null
if ($LASTEXITCODE -ne 0) { throw "Docker is not running. Start Docker Desktop and try again." }

if ($Skip -ne "Atelier") {
    foreach ($name in $atelier.Keys) {
        $spec = $atelier[$name]
        if (-not $NoBuild) {
            $buildArgs = @()
            if ($name -eq "atelier-worker") {
                Write-Host "    runtime: CUDA $CudaVersion ($CudaWheel) · torch $TorchVersion · python $PythonVersion" -ForegroundColor DarkGray
                $buildArgs = @(
                    "--build-arg", "CUDA_IMAGE=$CudaImage",
                    "--build-arg", "CUDA_WHEEL=$CudaWheel",
                    "--build-arg", "CUDA_MM=$CudaVersion",
                    "--build-arg", "TORCH_VERSION=$TorchVersion",
                    "--build-arg", "PYTHON_VERSION=$PythonVersion"
                )
            }
            Invoke-Step "building $name" { docker build -f $spec.Dockerfile @buildArgs -t "${name}:$Tag" $spec.Context }
        }
        Invoke-Step "tagging $name" { docker tag "${name}:$Tag" "$prefix/${name}:$Tag" }
        Invoke-Step "pushing $name" { docker push "$prefix/${name}:$Tag" }
    }
}

if ($Skip -ne "Third") {
    foreach ($img in $third) {
        # prom/prometheus:v2.54.1 -> harbor.local/atelier/prom/prometheus:v2.54.1
        # Harbor allows slashes inside a repository name, so the path is kept as is.
        Invoke-Step "pulling $img" { docker pull $img }
        Invoke-Step "tagging $img" { docker tag $img "$prefix/$img" }
        Invoke-Step "pushing $img" { docker push "$prefix/$img" }
    }
}

Write-Host ""
Write-Host "Done. On the classroom node put this in .env:" -ForegroundColor Green
Write-Host "  ATELIER_REGISTRY=$prefix/"
Write-Host "  ATELIER_TAG=$Tag"
Write-Host "then:  docker compose pull && docker compose up -d --no-build"
