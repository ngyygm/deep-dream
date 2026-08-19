"""``deep-dream server`` -- server lifecycle management.

Subcommands:
  start   Start the Deep-Dream Flask server (foreground or detached).
  stop    Stop a running detached server.
  status  Check if a server is running and show its details.
  logs    Show recent server log output.

All heavy imports are deferred to keep ``--help`` fast.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import click

from ._exit_codes import ERROR


# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

_PID_FILENAME = "server.pid"
_LOG_FILENAME = "server.log"
_DEFAULT_HOST = "0.0.0.0"
_DEFAULT_PORT = 16200


# ------------------------------------------------------------------
# Helpers (no heavy imports)
# ------------------------------------------------------------------


def _resolve_library_dir(ctx: click.Context) -> Path:
    """Return the storage root / library directory from the CLI context."""
    from core.cli._ctx import CliContext

    cli_ctx: CliContext = ctx.obj
    return cli_ctx.storage_root


def _pid_file_path(ctx: click.Context) -> Path:
    """Return the path to the server PID file inside the library directory."""
    return _resolve_library_dir(ctx) / _PID_FILENAME


def _log_file_path(ctx: click.Context) -> Path:
    """Return the path to the server log file inside the library directory."""
    return _resolve_library_dir(ctx) / _LOG_FILENAME


def _resolve_config_path(ctx: click.Context) -> str:
    """Return the config file path stored on the root Click context."""
    root_params = ctx.find_root().params
    return root_params.get("config", "service_config.json")


def _read_pid(pid_path: Path) -> Optional[int]:
    """Read a PID from *pid_path*. Returns None if missing or invalid."""
    if not pid_path.is_file():
        return None
    try:
        text = pid_path.read_text(encoding="utf-8").strip()
        return int(text)
    except (ValueError, OSError):
        return None


def _is_process_running(pid: int) -> bool:
    """Check whether a process with *pid* is alive (cross-platform)."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Process exists but we lack permission to signal it.
        return True
    except OSError:
        return False
    return True


def _read_pid_metadata(pid_path: Path) -> Dict[str, Any]:
    """Read the JSON metadata blob written alongside the PID.

    The PID file is a single line containing the integer PID. Metadata is
    stored in ``server.pid.json`` next to it so we can recover host, port,
    and start time without talking to the server.
    """
    meta_path = pid_path.with_suffix(".pid.json")
    if not meta_path.is_file():
        return {}
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _write_pid_file(
    pid_path: Path,
    pid: int,
    host: str,
    port: int,
    log_path: Path,
) -> None:
    """Write the PID file and a companion JSON metadata file."""
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path.write_text(str(pid), encoding="utf-8")
    meta = {
        "pid": pid,
        "host": host,
        "port": port,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "log_file": str(log_path),
    }
    meta_path = pid_path.with_suffix(".pid.json")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def _remove_pid_file(pid_path: Path) -> None:
    """Remove the PID file and its companion metadata file."""
    meta_path = pid_path.with_suffix(".pid.json")
    for p in (pid_path, meta_path):
        try:
            p.unlink(missing_ok=True)
        except OSError:
            pass


def _format_uptime(started_at: Optional[str]) -> str:
    """Return a human-readable uptime string from an ISO timestamp."""
    if not started_at:
        return "unknown"
    try:
        dt = datetime.fromisoformat(started_at)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        delta = datetime.now(timezone.utc) - dt
        total_seconds = int(delta.total_seconds())
        if total_seconds < 0:
            return "unknown"
        days, remainder = divmod(total_seconds, 86400)
        hours, remainder = divmod(remainder, 3600)
        minutes, seconds = divmod(remainder, 60)
        parts = []
        if days:
            parts.append(f"{days}d")
        if hours:
            parts.append(f"{hours}h")
        if minutes:
            parts.append(f"{minutes}m")
        parts.append(f"{seconds}s")
        return " ".join(parts)
    except (ValueError, TypeError):
        return "unknown"


def _kill_process(pid: int, timeout: float = 5.0) -> bool:
    """Terminate a process gracefully, then force-kill if needed.

    On Windows, ``SIGTERM`` is not available so we use ``taskkill /PID``.
    On POSIX, we send SIGTERM first and fall back to SIGKILL.
    """
    if sys.platform == "win32":
        try:
            subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=timeout,
            )
            return True
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            return False
    else:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            return True
        except PermissionError:
            return False
        except OSError:
            return False
        # Wait for the process to exit.
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not _is_process_running(pid):
                return True
            time.sleep(0.1)
        # Force kill.
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass
        return not _is_process_running(pid)


# ------------------------------------------------------------------
# Click group
# ------------------------------------------------------------------


@click.group()
def server() -> None:
    """Server lifecycle management (start, stop, status, logs)."""


# ------------------------------------------------------------------
# server start
# ------------------------------------------------------------------


@server.command()
@click.option(
    "--host",
    default=None,
    help="Bind address (overrides config).",
)
@click.option(
    "--port",
    default=None,
    type=int,
    help="Bind port (overrides config).",
)
@click.option(
    "--detach",
    is_flag=True,
    default=False,
    help="Run the server in the background.",
)
@click.option(
    "--verbose",
    is_flag=True,
    default=False,
    help="Enable verbose server logging.",
)
@click.pass_context
def start(ctx: click.Context, host: Optional[str], port: Optional[int], detach: bool, verbose: bool) -> None:
    """Start the Deep-Dream server."""
    from core.cli._output import OutputManager

    out = OutputManager(ctx)

    # Resolve host / port from config if not supplied on the CLI.
    config_path = _resolve_config_path(ctx)
    try:
        from core.server.config import load_config

        config = load_config(config_path)
    except Exception:
        config = {}

    host = host or config.get("host", _DEFAULT_HOST)
    port = port or config.get("port", _DEFAULT_PORT)

    pid_path = _pid_file_path(ctx)
    log_path = _log_file_path(ctx)

    # Check if a server is already running.
    existing_pid = _read_pid(pid_path)
    if existing_pid is not None and _is_process_running(existing_pid):
        meta = _read_pid_metadata(pid_path)
        existing_host = meta.get("host", host)
        existing_port = meta.get("port", port)
        out.error(
            f"Server already running (PID {existing_pid})",
            hint=f"URL: http://{existing_host}:{existing_port}  |  Use 'deep-dream server stop' first.",
            code=ERROR,
        )
        return  # unreachable; error() raises SystemExit

    # Clean stale PID file if the process is dead.
    if existing_pid is not None and not _is_process_running(existing_pid):
        _remove_pid_file(pid_path)

    # Build the command line for the server process.
    server_args = [
        sys.executable,
        "-m", "core.server.api",
        "--config", config_path,
        "--host", str(host),
        "--port", str(port),
        "--skip-llm-check",
    ]
    if verbose:
        server_args.append("--debug")

    if detach:
        # Ensure the library directory exists.
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # Open log file for stdout+stderr capture.
        log_file = open(log_path, "a", encoding="utf-8")

        try:
            # On Windows, creationflags=CREATE_NEW_PROCESS_GROUP + DETACHED_PROCESS
            # is the equivalent of start_new_session=True on POSIX.
            kwargs: Dict[str, Any] = {}
            if sys.platform == "win32":
                kwargs["creationflags"] = (
                    subprocess.CREATE_NEW_PROCESS_GROUP
                    | subprocess.DETACHED_PROCESS
                )
            else:
                kwargs["start_new_session"] = True

            proc = subprocess.Popen(
                server_args,
                stdout=log_file,
                stderr=log_file,
                close_fds=True,
                **kwargs,
            )
        except OSError as exc:
            log_file.close()
            out.error(
                f"Failed to start server: {exc}",
                hint="Check that Python is on PATH and the library directory is writable.",
                code=ERROR,
            )
            return

        # Give the process a moment to see if it dies immediately.
        time.sleep(0.5)
        if proc.poll() is not None:
            log_file.close()
            out.error(
                f"Server process exited immediately with code {proc.returncode}",
                hint=f"Check the log file for details: {log_path}",
                code=ERROR,
            )
            return

        # Write PID file with metadata.
        _write_pid_file(pid_path, proc.pid, host, port, log_path)

        # Show success.
        display_host = host if host != "0.0.0.0" else "127.0.0.1"
        url = f"http://{display_host}:{port}"
        if out.is_json:
            from core.cli._output import json_result

            payload = json_result(
                "server start",
                {
                    "pid": proc.pid,
                    "host": host,
                    "port": port,
                    "url": url,
                    "log_file": str(log_path),
                    "detached": True,
                },
            )
            click.echo(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            out.success(f"Server started in background (PID {proc.pid})")
            click.echo(f"  URL:  {url}", err=True)
            click.echo(f"  PID:  {proc.pid}", err=True)
            click.echo(f"  Log:  {log_path}", err=True)
    else:
        # Foreground mode: we exec into the server process so the user
        # sees output directly and Ctrl+C works naturally.
        if out.is_json:
            out.error(
                "Foreground server mode cannot produce JSON output (stdout is captured by the server).",
                hint="Use --detach for JSON output, or remove --json for interactive use.",
                code=ERROR,
            )
            return

        click.echo(f"Starting Deep-Dream server on {host}:{port} ...", err=True)
        click.echo("Press Ctrl+C to stop.", err=True)
        click.echo(err=True)

        try:
            os.execv(sys.executable, server_args)
        except OSError as exc:
            out.error(
                f"Failed to exec server: {exc}",
                code=ERROR,
            )


# ------------------------------------------------------------------
# server stop
# ------------------------------------------------------------------


@server.command()
@click.option(
    "--yes",
    "-y",
    is_flag=True,
    default=False,
    help="Skip confirmation prompt.",
)
@click.pass_context
def stop(ctx: click.Context, yes: bool) -> None:
    """Stop a running server."""
    from core.cli._output import OutputManager

    out = OutputManager(ctx)

    pid_path = _pid_file_path(ctx)
    pid = _read_pid(pid_path)

    if pid is None:
        out.error(
            "No running server found (PID file missing).",
            hint=f"Expected PID file at: {pid_path}",
            code=ERROR,
        )
        return

    meta = _read_pid_metadata(pid_path)
    display_host = meta.get("host", "unknown")
    display_port = meta.get("port", "unknown")

    if not _is_process_running(pid):
        # Stale PID file -- clean it up.
        _remove_pid_file(pid_path)
        out.error(
            f"Server process (PID {pid}) is not running.",
            hint="Stale PID file removed.",
            code=ERROR,
        )
        return

    # Confirmation.
    if not yes:
        if out.is_json:
            out.error(
                "Confirmation required. Use --yes flag with --json mode.",
                code=ERROR,
            )
            return
        click.echo(
            f"About to stop server (PID {pid}, {display_host}:{display_port}). "
            "Continue? [y/N] ",
            nl=False,
            err=True,
        )
        answer = click.getchar(echo=True)
        click.echo(err=True)
        if answer.lower() != "y":
            click.echo("Aborted.", err=True)
            return

    killed = _kill_process(pid)

    if killed:
        _remove_pid_file(pid_path)
        if out.is_json:
            from core.cli._output import json_result

            payload = json_result("server stop", {"pid": pid, "killed": True})
            click.echo(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            out.success(f"Server stopped (PID {pid})")
    else:
        out.error(
            f"Failed to stop server (PID {pid}).",
            hint="The process may require elevated permissions. Try manually: "
            + (
                f"taskkill /PID {pid} /F"
                if sys.platform == "win32"
                else f"kill -9 {pid}"
            ),
            code=ERROR,
        )


# ------------------------------------------------------------------
# server status
# ------------------------------------------------------------------


@server.command()
@click.pass_context
def status(ctx: click.Context) -> None:
    """Check if the server is running."""
    from core.cli._output import OutputManager

    out = OutputManager(ctx)

    pid_path = _pid_file_path(ctx)
    pid = _read_pid(pid_path)
    meta = _read_pid_metadata(pid_path)

    host = meta.get("host", _DEFAULT_HOST)
    port = meta.get("port", _DEFAULT_PORT)
    started_at = meta.get("started_at")
    log_file = meta.get("log_file", str(_log_file_path(ctx)))

    if pid is None:
        data = {
            "running": False,
            "pid": None,
            "host": host,
            "port": port,
            "uptime": None,
            "url": None,
            "log_file": log_file,
        }
        if out.is_json:
            click.echo(json.dumps(data, ensure_ascii=False, indent=2))
        else:
            click.echo("Server is not running.", err=True)
            click.echo(f"  PID file: {pid_path} (not found)", err=True)
        return

    running = _is_process_running(pid)
    uptime = _format_uptime(started_at) if running else None
    display_host = host if host != "0.0.0.0" else "127.0.0.1"
    url = f"http://{display_host}:{port}" if running else None

    data = {
        "running": running,
        "pid": pid,
        "host": host,
        "port": port,
        "uptime": uptime,
        "url": url,
        "started_at": started_at,
        "log_file": log_file,
    }

    if out.is_json:
        from core.cli._output import json_result

        payload = json_result("server status", data)
        click.echo(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    if running:
        click.echo("Server is running.", err=True)
        click.echo(f"  PID:    {pid}", err=True)
        click.echo(f"  URL:    {url}", err=True)
        click.echo(f"  Uptime: {uptime}", err=True)
        click.echo(f"  Log:    {log_file}", err=True)
    else:
        click.echo(f"Server is NOT running (stale PID file, PID {pid}).", err=True)
        click.echo(f"  PID file: {pid_path}", err=True)
        click.echo("  Hint: Run 'deep-dream server start' to launch.", err=True)


# ------------------------------------------------------------------
# server logs
# ------------------------------------------------------------------


@server.command()
@click.option(
    "--lines",
    "-n",
    default=50,
    show_default=True,
    type=int,
    help="Number of recent lines to show.",
)
@click.pass_context
def logs(ctx: click.Context, lines: int) -> None:
    """Show recent server log output."""
    from core.cli._output import OutputManager

    out = OutputManager(ctx)

    log_path = _log_file_path(ctx)

    if not log_path.is_file():
        if out.is_json:
            click.echo(json.dumps({"success": True, "lines": [], "log_file": str(log_path)}))
        else:
            click.echo(f"No log file found at: {log_path}", err=True)
            click.echo("The server may not have been started with --detach.", err=True)
        return

    try:
        all_lines = log_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        out.error(f"Cannot read log file: {exc}", code=ERROR)
        return

    tail = all_lines[-lines:] if lines < len(all_lines) else all_lines

    if out.is_json:
        click.echo(json.dumps({
            "success": True,
            "log_file": str(log_path),
            "total_lines": len(all_lines),
            "showing": len(tail),
            "lines": tail,
        }, ensure_ascii=False))
    else:
        if not tail:
            click.echo("Log file is empty.", err=True)
            return
        for line in tail:
            click.echo(line)
