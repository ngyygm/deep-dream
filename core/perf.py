"""
轻量性能计时工具。

Usage:
    from core.perf import _perf_timer

    with _perf_timer("method_name | step_desc"):
        ... do work ...
    # outputs: [PERF] method_name | step_desc | 42.3ms
"""

import contextlib
import logging
import time

_perf_log = logging.getLogger("tmg.perf")


@contextlib.contextmanager
def _perf_timer(operation: str, logger=None):
    """计时上下文管理器，输出到 tmg.perf logger。"""
    log = logger or _perf_log
    t0 = time.perf_counter()
    yield
    ms = (time.perf_counter() - t0) * 1000
    log.debug("[PERF] %s | %.1fms", operation, ms)
