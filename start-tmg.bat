@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1

set PORT=16200

:: Kill existing process on port
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":%PORT% " ^| findstr "LISTENING"') do (
    echo Killing process %%a on port %PORT%...
    taskkill /F /PID %%a >nul 2>&1
)

timeout /t 2 /nobreak >nul

:: Start TMG API service in monitor mode
echo Starting TMG API in monitor mode on http://127.0.0.1:%PORT%...
echo 提示: Embedding 权重加载完成后，会默认做 LLM 连通性检查（可能较慢）；若需跳过可加参数 --skip-llm-check
cd /d "%~dp0"
.venv\Scripts\python.exe service_api.py --config service_config.json --host 127.0.0.1 --port %PORT% --log-mode monitor
pause
