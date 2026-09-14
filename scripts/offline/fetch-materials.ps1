<#
.SYNOPSIS
  Prepare the materials on Windows.

.DESCRIPTION
  The main course needs nothing: its corpus is generated from a seed on the node.
  This script is for the advanced track, which uses public models and datasets.

.EXAMPLE
  .\scripts\offline\fetch-materials.ps1 -Out D:\atelier-materials

.EXAMPLE
  # see the download plan and its size without fetching anything
  .\scripts\offline\fetch-materials.ps1 -Plan

.EXAMPLE
  # the usual class set, about 13 GB
  .\scripts\offline\fetch-materials.ps1 -Out D:\atelier-materials -Tracks core

.EXAMPLE
  # just the pieces one lesson needs
  .\scripts\offline\fetch-materials.ps1 -Out D:\atelier-materials -Names gsm8k,Qwen2.5-0.5B-Instruct
#>
[CmdletBinding()]
param(
    [string]$Out = ".\materials",
    [string[]]$Names,
    [string[]]$Tracks,                       # core, extended, large
    [string[]]$Groups,                       # rag, eval, sft, preference, ...
    [ValidateSet("all", "models", "datasets")][string]$Only = "all",
    [switch]$Plan,                           # list what would be fetched, download nothing
    [switch]$SkipOptional,
    [switch]$SkipInstall
)
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..\..")

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    throw "Python 3.11+ is needed. Install it from python.org or the Microsoft Store."
}
if (-not $SkipInstall) {
    Write-Host "==> installing huggingface_hub, datasets, pyyaml" -ForegroundColor Cyan
    python -m pip install --quiet --upgrade huggingface_hub datasets pyyaml
}

# Long model paths overrun the classic 260-character limit; symlinks need admin.
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$env:HF_HUB_ENABLE_HF_TRANSFER = "0"

$argv = @("scripts/offline/fetch-materials.py", "--out", $Out)
if ($Only -ne "all") { $argv += @("--only", $Only) }
if ($Names) { $argv += @("--names") + $Names }
if ($Tracks) { $argv += @("--tracks") + $Tracks }
if ($Groups) { $argv += @("--groups") + $Groups }
if ($Plan) { $argv += "--plan" }
if ($SkipOptional) { $argv += "--skip-optional" }

Write-Host "==> fetching into $Out" -ForegroundColor Cyan
python @argv
if ($LASTEXITCODE -ne 0) { throw "fetch-materials.py failed" }

if ($Plan) { exit 0 }
$size = (Get-ChildItem -Recurse -File $Out | Measure-Object Length -Sum).Sum / 1GB
Write-Host ""
Write-Host ("materials ready: {0:N1} GB in {1}" -f $size, (Resolve-Path $Out)) -ForegroundColor Green
Write-Host "Copy the folder to /srv/atelier/materials on the node, for example:"
Write-Host "  scp -r $Out you@node:/srv/atelier/materials"
