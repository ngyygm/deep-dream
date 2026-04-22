@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1

set PORT=5000

:: Kill existing process on port
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":%PORT% " ^| findstr "LISTENING"') do (
    echo Killing process %%a on port %PORT%...
    taskkill /F /PID %%a >nul 2>&1
)

timeout /t 2 /nobreak >nul

:: Start DeepDream Web visualization
echo Starting DeepDream Web on http://localhost:%PORT%...
cd /d "%~dp0"
.venv\Scripts\python.exe -m server.web --config service_config.json --port %PORT% --host 127.0.0.1 --graph-id default
pause
