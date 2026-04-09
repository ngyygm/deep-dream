"""管道调试日志：写入文件，便于事后分析。

使用方式：
    from processor.debug_log import debug_log
    debug_log("步骤2完成: 抽取到 5 个实体")

日志文件位置：{storage_path}/debug_pipeline.log
如果未设置 storage_path，默认写入 /tmp/tmg_debug_pipeline.log
"""
from __future__ import annotations

import datetime
import threading

_LOG_PATH: str | None = None
_LOCK = threading.Lock()


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
    ts = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
    line = f"[{ts}] {msg}"
    with _LOCK:
        _write_raw(line)


def log_section(title: str):
    """写入分隔标题，便于在日志中定位。"""
    with _LOCK:
        _write_raw(f"\n{'─'*50}")
        _write_raw(f"  {title}")
        _write_raw(f"{'─'*50}")
