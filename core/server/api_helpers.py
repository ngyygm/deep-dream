"""
Deep-Dream API helpers — server startup utilities.

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
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


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

    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
            logging.info("已发送 SIGTERM 到进程 %d (占用端口 %d)", pid, port)
        except ProcessLookupError:
            pass
        except PermissionError:
            logging.warning("无权限终止进程 %d", pid)

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
