"""管道调试日志：写入文件，便于事后分析。

使用方式：
    from core.debug_log import log as dbg, log_section as dbg_section

    dbg("步骤2完成: 抽取到 5 个实体")
    dbg_section("步骤9: 实体对齐")

日志文件位置：{storage_path}/debug_pipeline.log
如果未设置 storage_path，默认写入 /tmp/tmg_debug_pipeline.log

启用方式：设置环境变量 DEEP_DREAM_DEBUG_LOG=1
"""
from __future__ import annotations

import datetime
import os
import threading
import json as _json

_LOG_PATH: str | None = None
_LOCK = threading.Lock()
# Guard: set to True to enable debug logging (default False for production perf)
_ENABLED: bool = os.environ.get("DEEP_DREAM_DEBUG_LOG", "").lower() in ("1", "true", "yes")


def _get_path() -> str:
    if _LOG_PATH:
        return _LOG_PATH
    return "/tmp/tmg_debug_pipeline.log"


def _write_raw(text: str):
    try:
        with open(_get_path(), "a", encoding="utf-8") as f:
            f.write(text + "\n")
    except Exception:
        pass  # 日志写入失败不应影响主流程


def log(msg: str):
    """写入一行调试日志（带时间戳）。"""
    if not _ENABLED:
        return
    ts = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
    line = f"[{ts}] {msg}"
    with _LOCK:
        _write_raw(line)


def log_section(title: str):
    """写入分隔标题，便于在日志中定位。"""
    if not _ENABLED:
        return
    with _LOCK:
        _write_raw(f"\n{'─'*50}")
        _write_raw(f"  {title}")
        _write_raw(f"{'─'*50}")


def log_struct(event: str, **fields):
    """写入结构化日志行，便于机器解析和事后溯源。

    格式：[HH:MM:SS.mmm] [ALIGN] event | key1=val1 | key2=val2 | ...
    多行值会被转义为单行（换行符→⏎）。
    """
    if not _ENABLED:
        return
    ts = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
    parts = [f"[{ts}] [ALIGN] {event}"]
    for k, v in fields.items():
        if isinstance(v, str):
            v = v.replace("\n", "⏎")
        elif isinstance(v, (list, dict)):
            try:
                v = _json.dumps(v, ensure_ascii=False)[:500]
            except Exception:
                v = str(v)[:500]
        parts.append(f"{k}={v}")
    line = " | ".join(parts)
    with _LOCK:
        _write_raw(line)
