"""
Pipeline shared-state helpers for remember_text.

Extracted from orchestrator.py to keep that file focused on the main flow.
These functions operate on the SimpleNamespace state object created by
``init_remember_shared_state`` and used by the step9 / step10 workers.
"""
import sys
import time
import threading
import types
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Optional

from core.utils import (
    clear_parallel_log_context,
    set_pipeline_role,
    set_window_label,
    wprint_info,
)


# ------------------------------------------------------------------
# State initialisation
# ------------------------------------------------------------------

def init_remember_shared_state(N: int) -> types.SimpleNamespace:
    """Pre-allocate arrays, events, and error collectors for *N* windows."""
    s = types.SimpleNamespace()
    s.N = N
    s.episodes = [None] * N
    s.input_texts = [None] * N
    s.extract_results = [None] * N
    s.early_entity_results = [None] * N
    s.entity_content_done = [threading.Event() for _ in range(N)]
    s.align_results = [None] * N
    s.step10_results = [None] * N
    s.aligned_entity_counts = [0] * N
    s.window_timings = [{} for _ in range(N)]
    s.extract_done = [threading.Event() for _ in range(N)]
    s.step9_done_ev = [threading.Event() for _ in range(N)]
    s.step10_done_ev = [threading.Event() for _ in range(N)]
    s.errors: list = []
    s.errors_lock = threading.Lock()
    s.window_failures = [None] * N
    s.control_lock = threading.Lock()
    s.control_state = {"action": None}
    s.prefetch_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tmg-chain-prefetch")
    return s


# ------------------------------------------------------------------
# Error recording
# ------------------------------------------------------------------

def record_window_error(state: types.SimpleNamespace, stage: str, idx: int, exc: Exception) -> bool:
    with state.errors_lock:
        if state.window_failures[idx] is None:
            state.window_failures[idx] = (stage, exc)
            state.errors.append((stage, idx, exc))
            return True
    return False


# ------------------------------------------------------------------
# Control-flow signalling
# ------------------------------------------------------------------

def signal_control_stop(state: types.SimpleNamespace, action: str, from_index: int, *,
                        set_extract: bool = True, set_step9: bool = True, set_step10: bool = True):
    with state.control_lock:
        if state.control_state["action"] is None:
            state.control_state["action"] = action
        _from = max(0, min(from_index, state.N))
        for j in range(_from, state.N):
            if set_extract:
                state.extract_done[j].set()
            if set_step9:
                state.step9_done_ev[j].set()
            if set_step10:
                state.step10_done_ev[j].set()


def poll_control(state: types.SimpleNamespace, control_callback: Optional[Callable]) -> Optional[str]:
    action = state.control_state["action"]
    if action:
        return action
    if control_callback is None:
        return None
    action = control_callback()
    if action in ("pause", "cancel"):
        with state.control_lock:
            if state.control_state["action"] is None:
                state.control_state["action"] = action
            return state.control_state["action"]
    return None


# ------------------------------------------------------------------
# Progress helpers
# ------------------------------------------------------------------

def safe_progress(progress_callback, progress: float, label: str, message: str, chain_id: str = "step9"):
    if not progress_callback:
        return
    progress_callback(progress, label, message, chain_id)


def run_with_progress_heartbeat(
    run_fn: Callable[[], Any],
    *,
    chain_id: str,
    base_progress: float,
    phase_label: str,
    message: str,
    window_label: str,
    pipeline_role: str,
    progress_callback=None,
    heartbeat_seconds: float = 5.0,
    log_interval_seconds: float = 30.0,
) -> Any:
    """Run *run_fn* while emitting periodic heartbeats so the UI does not appear stuck."""
    stop_ev = threading.Event()
    started = time.time()

    def _heartbeat() -> None:
        last_log_elapsed = 0.0
        set_window_label(window_label)
        set_pipeline_role(pipeline_role)
        try:
            while not stop_ev.wait(heartbeat_seconds):
                elapsed = max(1, int(time.time() - started))
                hb_label = f"{phase_label} · 已等待 {elapsed}s"
                hb_message = f"{message}（已等待 {elapsed}s）"
                safe_progress(progress_callback, base_progress, hb_label, hb_message, chain_id)
                if elapsed - last_log_elapsed >= log_interval_seconds:
                    wprint_info(f"{phase_label} · 长调用进行中（已等待 {elapsed}s）")
                    last_log_elapsed = float(elapsed)
        finally:
            clear_parallel_log_context()

    hb = threading.Thread(
        target=_heartbeat,
        name=f"tmg-heartbeat-{chain_id}",
        daemon=True,
    )
    hb.start()
    try:
        return run_fn()
    finally:
        stop_ev.set()
        hb.join(timeout=0.2)


def safe_prefetch_submit(state: types.SimpleNamespace, fn, *args, **kwargs):
    """Submit to prefetch executor; returns ``None`` if interpreter is finalising or executor shut down."""
    try:
        if sys.is_finalizing():
            return None
    except Exception:
        pass
    try:
        return state.prefetch_executor.submit(fn, *args, **kwargs)
    except RuntimeError:
        return None
