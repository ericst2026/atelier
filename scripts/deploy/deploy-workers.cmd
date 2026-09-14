@echo off
REM Runs deploy-workers.sh in Git Bash. PowerShell 5.1 corrupts the binary streams
REM that script pipes over ssh, so it is not a PowerShell script.
REM   scripts\deploy\deploy-workers.cmd up
REM Set CONTROL_HOST first:  set CONTROL_HOST=192.168.1.20
setlocal
set "BASH=%ProgramFiles%\Git\bin\bash.exe"
if not exist "%BASH%" set "BASH=%ProgramFiles(x86)%\Git\bin\bash.exe"
if not exist "%BASH%" set "BASH=%LocalAppData%\Programs\Git\bin\bash.exe"
if not exist "%BASH%" (
  echo Git Bash was not found. Install Git for Windows, or run deploy-workers.sh from a Linux shell.
  exit /b 1
)
"%BASH%" "%~dp0deploy-workers.sh" %*
