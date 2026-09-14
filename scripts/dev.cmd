@echo off
REM Runs scripts\dev.ps1 without touching the machine's execution policy.
REM Double-click it, or from a prompt:  scripts\dev.cmd
REM Every argument is passed straight through, e.g.  scripts\dev.cmd -Port 5174
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0dev.ps1" %*
