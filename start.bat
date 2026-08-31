@echo off
chcp 65001 >nul 2>&1
title Deep-Dream Server

:: Never kill an unrelated process just because it owns the configured port.
echo [1/3] Checking port 16200...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr "LISTENING" ^| findstr ":16200 "') do (
    echo       Port 16200 is already in use (PID %%a). Stop that service explicitly, then retry.
    exit /b 2
)

:: Start server
echo [2/3] Starting Deep-Dream...
cd /d "%~dp0"
python -m core.server.api --config service_config.json
set "EXIT_CODE=%ERRORLEVEL%"

:: If server exits
echo.
echo [3/3] Server stopped.
pause
exit /b %EXIT_CODE%
