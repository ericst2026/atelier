@echo off
REM Wrapper so this runs without changing the machine's execution policy.
REM Arguments pass straight through:  scripts\offline\check-node.cmd -Registry harbor.local
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0check-node.ps1" %*
