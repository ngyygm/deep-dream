"""
Deep-Dream API helpers — validation, response formatting, and server utilities.

Extracted from api.py to keep route definitions separate from infrastructure.
"""
from __future__ import annotations

import errno
import logging
import os
import re as _re
import signal
import socket
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from flask import jsonify

from core.server.config import merge_llm_alignment, resolve_embedding_model
from core.server.registry import GraphRegistry

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Input validation helpers
# ----------------------------------------------------------------------

def validate_graph_id(graph_id: Any) -> Tuple[bool, Optional[str], int]:
    """Validate graph_id parameter. Returns (is_valid, error_message, status_code)."""
    if not graph_id:
        return False, "graph_id is required", 400
    if not isinstance(graph_id, str):
        return False, "graph_id must be a string", 400
    try:
        GraphRegistry.validate_graph_id(graph_id)
        return True, None, 200
    except ValueError as e:
        return False, str(e), 400


def validate_text_input(text: Any, field_name: str = "text", min_length: int = 1,
                        max_length: int = 10_000_000) -> Tuple[bool, Optional[str], int]:
    """Validate text input parameters. Returns (is_valid, error_message, status_code)."""
    if text is None:
        return False, f"{field_name} is required", 400
    if not isinstance(text, str):
        return False, f"{field_name} must be a string", 400
    text = text.strip()
    if len(text) < min_length:
        return False, f"{field_name} must be at least {min_length} character(s)", 400
    if len(text) > max_length:
        return False, f"{field_name} exceeds maximum length of {max_length} characters", 400
    return True, None, 200


def validate_positive_int(value: Any, field_name: str = "value",
                          min_val: int = 1, max_val: int = 10000) -> Tuple[bool, Optional[str], int]:
    """Validate positive integer parameters. Returns (is_valid, error_message, status_code)."""
    if value is None:
        return True, None, 200  # Optional parameter
    try:
        int_val = int(value)
        if int_val < min_val:
            return False, f"{field_name} must be at least {min_val}", 400
        if int_val > max_val:
            return False, f"{field_name} must not exceed {max_val}", 400
        return True, None, 200
    except (ValueError, TypeError):
        return False, f"{field_name} must be a valid integer", 400


def validate_float_range(value: Any, field_name: str = "value",
                         min_val: float = 0.0, max_val: float = 1.0) -> Tuple[bool, Optional[str], int]:
    """Validate float parameters in a range. Returns (is_valid, error_message, status_code)."""
    if value is None:
        return True, None, 200  # Optional parameter
    try:
        float_val = float(value)
        if float_val < min_val or float_val > max_val:
            return False, f"{field_name} must be between {min_val} and {max_val}", 400
        return True, None, 200
    except (ValueError, TypeError):
        return False, f"{field_name} must be a valid number", 400


def make_validation_error(message: str) -> tuple:
    """Return a standardized validation error response."""
    return jsonify({"success": False, "error": message}), 400


# ----------------------------------------------------------------------
# Processor builder
# ----------------------------------------------------------------------

def build_processor(config: Dict[str, Any]):
    """Build a TemporalMemoryGraphProcessor from a config dict."""
    from core import TemporalMemoryGraphProcessor

    storage_path = config.get("storage_path", "./graph/tmg_storage")
    chunking = config.get("chunking") or {}
    window_size = chunking.get("window_size", 1000)
    overlap = chunking.get("overlap", 200)
    llm = config.get("llm") or {}
    embedding = config.get("embedding") or {}
    pipeline = config.get("pipeline") or {}
    runtime = config.get("runtime") or {}
    runtime_concurrency = runtime.get("concurrency") or {}
    runtime_task = runtime.get("task") or {}
    pipeline_search = pipeline.get("search") or {}
    pipeline_alignment = pipeline.get("alignment") or {}
    pipeline_extraction = pipeline.get("extraction") or {}
    pipeline_remember = pipeline.get("remember") or {}
    pipeline_debug = pipeline.get("debug") or {}
    max_concurrency = llm.get("max_concurrency")
    model_path, model_name, use_local = resolve_embedding_model(embedding)
    kwargs: Dict[str, Any] = {
        "storage_path": storage_path,
        "window_size": window_size,
        "overlap": overlap,
        "llm_api_key": llm.get("api_key"),
        "llm_model": llm.get("model", "gpt-4"),
        "llm_base_url": llm.get("base_url"),
        "alignment_llm": merge_llm_alignment(llm),
        "llm_think_mode": bool(llm.get("think", llm.get("think_mode", False))),
        "llm_max_tokens": llm.get("max_tokens") if llm.get("max_tokens") else None,
        "llm_context_window_tokens": llm.get("context_window_tokens"),
        "max_llm_concurrency": max_concurrency,
        "llm_timeout_seconds": llm.get("timeout_seconds", 300),
        "llm_connect_timeout_seconds": llm.get("connect_timeout_seconds", 30),
        "embedding_model_path": model_path,
        "embedding_model_name": model_name,
        "embedding_device": embedding.get("device", "cpu"),
        "embedding_use_local": use_local,
        "embedding_cache_max_size": embedding.get("cache_max_size"),
        "embedding_cache_ttl": embedding.get("cache_ttl"),
        "load_cache_memory": runtime_task.get("load_cache_memory", pipeline.get("load_cache_memory")),
        "max_concurrent_windows": runtime_concurrency.get("window_workers", pipeline.get("max_concurrent_windows")),
    }
    for key in (
        "similarity_threshold", "max_similar_entities", "content_snippet_length",
        "relation_content_snippet_length", "relation_endpoint_jaccard_threshold",
        "relation_endpoint_embedding_threshold",
        "jaccard_search_threshold",
        "embedding_name_search_threshold", "embedding_full_search_threshold",
    ):
        if key in pipeline_search:
            kwargs[key] = pipeline_search[key]
    if "max_alignment_candidates" in pipeline_alignment:
        kwargs["max_alignment_candidates"] = pipeline_alignment["max_alignment_candidates"]
    for key in (
        "extraction_rounds", "entity_extraction_rounds", "relation_extraction_rounds",
        "entity_post_enhancement", "prompt_episode_max_chars",
        "compress_multi_round_extraction",
    ):
        if key in pipeline_extraction:
            kwargs[key] = pipeline_extraction[key]
    if pipeline_remember:
        kwargs["remember_config"] = pipeline_remember
    if "distill_data_dir" in pipeline_debug:
        kwargs["distill_data_dir"] = pipeline_debug["distill_data_dir"]
    return TemporalMemoryGraphProcessor(**kwargs)


# ----------------------------------------------------------------------
# Server startup utilities
# ----------------------------------------------------------------------

def tcp_bind_probe(host: str, port: int) -> Tuple[bool, Optional[str]]:
    """Try to exclusively bind host:port, used to check port availability before start."""
    bind_addr = host if host not in ("", "0.0.0.0") else "0.0.0.0"
    sock: Optional[socket.socket] = None
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((bind_addr, int(port)))
        return True, None
    except OSError as e:
        if e.errno == errno.EADDRINUSE:
            return False, "端口已被占用 (EADDRINUSE)"
        return False, str(e)
    finally:
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass


def get_port_pids(port: int) -> List[int]:
    """Get PIDs occupying the given port (excluding self)."""
    my_pid = os.getpid()
    pids: List[int] = []

    # Prefer ss (faster, more common)
    try:
        result = subprocess.run(
            ["ss", "-tlnp", f"sport = :{port}"],
            capture_output=True, text=True, timeout=5,
        )
        for m in _re.finditer(r"pid=(\d+)", result.stdout):
            pid = int(m.group(1))
            if pid != my_pid:
                pids.append(pid)
        if pids:
            return pids
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Fallback to lsof
    try:
        result = subprocess.run(
            ["lsof", "-t", "-i", f":{port}"],
            capture_output=True, text=True, timeout=5,
        )
        for line in result.stdout.strip().splitlines():
            pid = int(line.strip())
            if pid != my_pid:
                pids.append(pid)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return pids


def kill_port_occupants(port: int) -> bool:
    """Kill processes occupying the given port. Returns True if all killed."""
    pids = get_port_pids(port)
    if not pids:
        return True

    all_killed = True
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
            logging.info("已发送 SIGTERM 到进程 %d (占用端口 %d)", pid, port)
        except ProcessLookupError:
            pass
        except PermissionError:
            logging.warning("无权限终止进程 %d", pid)
            all_killed = False

    # Poll for process exit (max 3 seconds)
    for _ in range(15):
        remaining = get_port_pids(port)
        if not remaining:
            return True
        time.sleep(0.2)

    # SIGTERM didn't work, escalate to SIGKILL
    remaining = get_port_pids(port)
    for pid in remaining:
        try:
            os.kill(pid, signal.SIGKILL)
            logging.warning("SIGTERM 无效，已发送 SIGKILL 到进程 %d (占用端口 %d)", pid, port)
        except ProcessLookupError:
            pass
        except PermissionError:
            logging.warning("无权限终止进程 %d", pid)
            all_killed = False

    # Wait 1 more second to confirm
    time.sleep(1)
    return not get_port_pids(port)


def resolve_listen_port(
    host: str,
    preferred_port: int,
    auto_fallback: bool,
    max_extra: int = 10,
) -> Tuple[int, bool]:
    """
    If preferred_port is bindable, use it; otherwise try +1..+max_extra when auto_fallback.
    Returns (actual_port, whether port was switched).
    """
    ok, _ = tcp_bind_probe(host, preferred_port)
    if ok:
        return preferred_port, False
    if not auto_fallback:
        return preferred_port, False
    for delta in range(1, max_extra + 1):
        p = preferred_port + delta
        ok2, _ = tcp_bind_probe(host, p)
        if ok2:
            return p, True
    return preferred_port, False


def check_storage_writable(storage_root: Path) -> Optional[str]:
    """Try to create/delete a test file under storage_path; return error message if not writable."""
    probe = storage_root / ".tmg_write_probe"
    try:
        storage_root.mkdir(parents=True, exist_ok=True)
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return None
    except OSError as e:
        return f"存储路径不可写或无法创建: {storage_root} ({e})"
