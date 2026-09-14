@echo off
REM Wrapper so this runs without changing the machine's execution policy.
REM Arguments pass straight through:  scripts\offline\fetch-materials.cmd -Registry harbor.local
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0fetch-materials.ps1" %*
