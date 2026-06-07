@echo off
chcp 65001 >nul 2>&1
title Deep-Dream Server

:: Kill existing Python server on port 16200 (skip non-Python processes like VS Code)
echo [1/3] Checking port 16200...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr "LISTENING" ^| findstr ":16200 "') do (
    for /f "tokens=1" %%b in ('tasklist /FI "PID eq %%a" /NH 2^>nul ^| findstr /I "python"') do (
        echo       Killing Python PID %%a...
        taskkill /PID %%a /F >nul 2>&1
    )
)
timeout /t 2 /nobreak >nul

:: Start server
echo [2/3] Starting Deep-Dream...
cd /d "%~dp0"
python -c "import sys; sys.argv=['deep-dream','--config','service_config.json','--port','16200']; from core.server.api import main; main()"

:: If server exits
echo.
echo [3/3] Server stopped.
pause
