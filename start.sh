#!/usr/bin/env bash
# Deep-Dream 服务器启动脚本（Linux/macOS），相当于 Windows 版 start.bat
# 使用文档化的入口点 `deep-dream server`（console_script 在 pyproject.toml 中声明），
# 而非脆弱的 `python -c "sys.argv=..."` 包装。
set -euo pipefail

PORT="${DEEP_DREAM_PORT:-16200}"
CONFIG="${DEEP_DREAM_CONFIG:-service_config.json}"

# 切换到脚本所在目录（通常为仓库根），确保配置与相对路径可用
cd "$(dirname "$0")"

# 如果端口已被占用，尝试释放（仅终止本用户可访问的进程）
echo "[1/3] Checking port ${PORT}..."
if command -v lsof >/dev/null 2>&1; then
    pid="$(lsof -ti tcp:"${PORT}" || true)"
    if [ -n "${pid}" ]; then
        echo "      Killing process ${pid} on port ${PORT}..."
        kill "${pid}" 2>/dev/null || true
        sleep 2
    fi
fi

# 启动服务器：优先使用 console_script，回退到 python -m core.server.api
echo "[2/3] Starting Deep-Dream..."
if command -v deep-dream >/dev/null 2>&1; then
    exec deep-dream server --config "${CONFIG}" --port "${PORT}"
else
    exec python -m core.server.api --config "${CONFIG}" --port "${PORT}"
fi

# 若服务器退出
echo
echo "[3/3] Server stopped."
