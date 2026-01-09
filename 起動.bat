@echo off
setlocal enabledelayedexpansion

set SCRIPT_DIR=%~dp0
set PS_SCRIPT=%SCRIPT_DIR%scripts\launch_streamlit.ps1

if not exist "%PS_SCRIPT%" (
	echo launch_streamlit.ps1 was not found under the scripts directory.
	echo Please pull the latest repository or reinstall.
	pause
	exit /b 1
)

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%PS_SCRIPT%" %*

if %errorlevel% neq 0 (
	echo.
	echo Streamlit launcher exited with an error. Review the above log for details.
	pause
)
