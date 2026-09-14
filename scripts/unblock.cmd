@echo off
REM Windows marks files that came out of a downloaded zip, and PowerShell then
REM refuses to run them even when the execution policy allows local scripts.
REM This clears that mark on the project's own scripts. Run it once after unzipping.
powershell -NoProfile -ExecutionPolicy Bypass -Command "Get-ChildItem -Path '%~dp0..' -Recurse -Include *.ps1,*.psm1 | Unblock-File; Write-Host 'Unblocked.' -ForegroundColor Green"
