"""``deep-dream task`` -- task queue management via the server API.

All operations make HTTP requests to the running Deep-Dream server.
Requires ``deep-dream server start`` to be running first.
"""
from __future__ import annotations

import json as _json
import urllib.error
import urllib.request
import urllib.parse
from typing import Any, Dict, List, Optional

import click

from ._exit_codes import ERROR, NETWORK, NOT_FOUND
from ._output import OutputManager, format_timestamp


# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

_API_BASE = "http://127.0.0.1:16200/api/v1"
_SERVER_HINT = "Start with 'deep-dream server start'."


# ------------------------------------------------------------------
# HTTP helpers
# ------------------------------------------------------------------

def _api_url(path: str, params: Optional[Dict[str, str]] = None) -> str:
    """Build a full API URL from a relative path."""
    url = f"{_API_BASE}{path}"
    if params:
        filtered = {k: v for k, v in params.items() if v is not None}
        if filtered:
            url += "?" + urllib.parse.urlencode(filtered)
    return url


def _request(
    method: str,
    path: str,
    params: Optional[Dict[str, str]] = None,
    body: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Make an HTTP request to the server API and return parsed JSON.

    Parameters
    ----------
    method:
        HTTP method (GET, POST, DELETE).
    path:
        Relative API path, e.g. ``/remember/tasks``.
    params:
        Query string parameters.
    body:
        JSON body for POST requests.

    Returns
    -------
    Parsed JSON response body as a dict.
    """
    url = _api_url(path, params)
    data = None
    if body is not None:
        data = _json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Accept", "application/json")
    if data is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            raw = resp.read().decode("utf-8")
            return _json.loads(raw)
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            parsed = _json.loads(raw)
        except _json.JSONDecodeError:
            parsed = {"error": raw}
        # Attach HTTP status to help callers distinguish 404 from 409 etc.
        parsed["_http_status"] = exc.code
        return parsed
    except (OSError, urllib.error.URLError, TimeoutError) as exc:
        return {"_network_error": str(exc)}


def _is_error(resp: Dict[str, Any]) -> bool:
    """Check if a response dict represents an error."""
    if "_network_error" in resp:
        return True
    if resp.get("_http_status") and resp["_http_status"] >= 400:
        return True
    if resp.get("success") is False and resp.get("error"):
        return True
    if resp.get("error") and not resp.get("task_id") and not resp.get("tasks"):
        return True
    return False


def _error_message(resp: Dict[str, Any]) -> str:
    """Extract a human-readable error message from a response."""
    if "_network_error" in resp:
        return f"Cannot reach server at http://127.0.0.1:16200 ({resp['_network_error']})"
    return resp.get("error") or resp.get("message") or _json.dumps(resp, ensure_ascii=False)


def _ensure_connected(out: OutputManager) -> None:
    """Check server connectivity; exit with a helpful message if unreachable."""
    try:
        req = urllib.request.Request(f"{_API_BASE}/../health", method="GET")
        with urllib.request.urlopen(req, timeout=3) as resp:
            if resp.status >= 400:
                raise OSError(f"HTTP {resp.status}")
    except (OSError, urllib.error.URLError, TimeoutError):
        out.error(
            f"Cannot reach server at http://127.0.0.1:16200",
            hint=_SERVER_HINT,
            code=NETWORK,
        )


# ------------------------------------------------------------------
# Formatting helpers
# ------------------------------------------------------------------

def _short_id(task_id: str) -> str:
    """Truncate a task ID for display."""
    if len(task_id) > 12:
        return task_id[:12] + ".."
    return task_id


def _format_progress(task: Dict[str, Any]) -> str:
    """Format progress as a percentage string."""
    p = task.get("progress")
    if p is None:
        return ""
    try:
        return f"{float(p) * 100:.1f}%"
    except (ValueError, TypeError):
        return str(p)


def _format_status(status: str) -> str:
    """Colourise a status string for Rich output."""
    colours = {
        "queued": "cyan",
        "running": "bold green",
        "pausing": "yellow",
        "paused": "yellow",
        "completed": "green",
        "failed": "bold red",
        "cancelled": "dim",
        "cancelling": "yellow",
    }
    colour = colours.get(status, "")
    if colour:
        return f"[{colour}]{status}[/{colour}]"
    return status


def _format_elapsed(seconds: Optional[float]) -> str:
    """Format elapsed seconds as a human-readable duration."""
    if seconds is None:
        return ""
    try:
        s = float(seconds)
        if s < 60:
            return f"{s:.0f}s"
        if s < 3600:
            return f"{s / 60:.1f}m"
        return f"{s / 3600:.1f}h"
    except (ValueError, TypeError):
        return str(seconds)


def _format_timestamp_epoch(epoch: Optional[float]) -> str:
    """Format a unix epoch timestamp for display."""
    if epoch is None:
        return ""
    try:
        from datetime import datetime, timezone
        dt = datetime.fromtimestamp(float(epoch), tz=timezone.utc)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError, OSError):
        return str(epoch)


def _format_size(bytes_size: Optional[int]) -> str:
    """Format a byte count as a human-readable size."""
    if bytes_size is None:
        return ""
    try:
        n = int(bytes_size)
        for unit in ("B", "KB", "MB", "GB"):
            if abs(n) < 1024:
                return f"{n:.0f} {unit}" if unit != "B" else f"{n} B"
            n /= 1024  # type: ignore[assignment]
        return f"{n:.0f} TB"
    except (ValueError, TypeError):
        return str(bytes_size)


# ------------------------------------------------------------------
# Subcommands
# ------------------------------------------------------------------

@click.group()
@click.pass_context
def task(ctx: click.Context) -> None:
    """Manage the task queue (requires running server).

    Inspect, pause, resume, cancel, and retry tasks in the
    remember-pipeline task queue.  All commands communicate with the
    Deep-Dream server over HTTP.

    \b
    Examples:
      deep-dream task list
      deep-dream task status 1
      deep-dream task cancel 3 --yes
      deep-dream task pause 2
      deep-dream task resume 2
      deep-dream task retry 2
      deep-dream task resume-all
    """


@task.command("list")
@click.option(
    "--status",
    "status_filter",
    type=click.Choice(
        ["queued", "running", "pausing", "paused", "completed", "failed", "cancelled", "cancelling"],
        case_sensitive=False,
    ),
    default=None,
    help="Filter by status (queued, running, pausing, paused, completed, failed, cancelled, cancelling).",
)
@click.option(
    "--limit",
    type=int,
    default=50,
    show_default=True,
    help="Maximum number of tasks to show.",
)
@click.pass_context
def task_list(ctx: click.Context, status_filter: Optional[str], limit: int) -> None:
    """List tasks in the queue.

    Shows a table with Seq, ID, Source, Status, Progress, Phase, Size, and Created.
    """
    out = OutputManager(ctx)
    _ensure_connected(out)

    # Pass the status filter to the server so it is applied BEFORE the
    # priority-sort + limit slice; filtering client-side on a truncated list
    # silently returned empty results for completed/cancelled tasks.
    params: Dict[str, str] = {"limit": str(limit)}
    if status_filter:
        params["status"] = status_filter
    resp = _request("GET", "/remember/tasks", params=params)

    if _is_error(resp):
        if "_network_error" in resp:
            out.error(
                f"Cannot reach server at http://127.0.0.1:16200",
                hint=_SERVER_HINT,
                code=NETWORK,
            )
        out.error(_error_message(resp), code=ERROR)

    data = resp.get("data", resp)
    tasks: List[Dict[str, Any]] = data.get("tasks", []) if isinstance(data, dict) else []

    if out.is_json:
        from ._output import json_result
        payload = json_result("task-list", {"tasks": tasks, "count": len(tasks)})
        click.echo(_json.dumps(payload, ensure_ascii=False, indent=2))
        return

    if out.is_quiet:
        return

    # Rich table
    from rich.table import Table
    from rich.markup import escape as _rich_escape

    table = Table(
        title=f"Task Queue ({len(tasks)} tasks)",
        show_header=True,
        header_style="bold",
    )
    table.add_column("Seq", justify="right", style="dim", min_width=4)
    table.add_column("ID", style="cyan", min_width=14)
    table.add_column("Source", min_width=20, max_width=50)
    table.add_column("Status", min_width=10)
    table.add_column("Progress", justify="right", min_width=8)
    table.add_column("Phase", min_width=16, max_width=30)
    table.add_column("Size", justify="right", min_width=8)
    table.add_column("Created", min_width=18)

    for t in tasks:
        task_seq = str(t.get("task_seq", ""))
        tid = _short_id(t.get("task_id", ""))
        source = t.get("source_name", "") or ""
        if len(source) > 50:
            source = source[:47] + "..."
        # Source / phase_label come from user content and may contain
        # [brackets] that Rich would interpret as markup — escape them.
        source = _rich_escape(source)
        status = _format_status(t.get("status", ""))
        progress = _format_progress(t)
        phase_label = t.get("phase_label", "") or t.get("phase", "")
        if len(phase_label) > 30:
            phase_label = phase_label[:27] + "..."
        phase_label = _rich_escape(phase_label)
        size = _format_size(t.get("document_size_bytes"))
        created = _format_timestamp_epoch(t.get("created_at"))
        table.add_row(task_seq, tid, source, status, progress, phase_label, size, created)

    out.console.print(table)

    if not tasks:
        out.console.print("[dim]No tasks in queue.[/dim]")


@task.command("status")
@click.argument("task_id")
@click.pass_context
def task_status(ctx: click.Context, task_id: str) -> None:
    """Show detailed status of a single task.

    TASK_ID can be a sequence number (e.g. 1, 2) or a full task ID.
    """
    out = OutputManager(ctx)
    _ensure_connected(out)

    resp = _request("GET", f"/remember/tasks/{task_id}")

    if _is_error(resp):
        if "_network_error" in resp:
            out.error(
                f"Cannot reach server at http://127.0.0.1:16200",
                hint=_SERVER_HINT,
                code=NETWORK,
            )
        http_status = resp.get("_http_status")
        if http_status == 404:
            out.error(f"Task not found: {task_id}", code=NOT_FOUND)
        out.error(_error_message(resp), code=ERROR)

    data = resp.get("data", resp)

    if out.is_json:
        from ._output import json_result
        payload = json_result("task-status", data)
        click.echo(_json.dumps(payload, ensure_ascii=False, indent=2))
        return

    if out.is_quiet:
        return

    # Rich panel with full task info
    _render_task_detail(out, data)


def _render_task_detail(out: OutputManager, data: Dict[str, Any]) -> None:
    """Render a full task detail panel."""
    from rich.panel import Panel
    from rich.markup import escape as _rich_esc
    from rich.table import Table

    # -- Main info panel ---------------------------------------------------
    status = data.get("status", "unknown")
    lines: List[str] = []

    lines.append(f"[bold]Task ID:[/bold]       {data.get('task_id', '?')}")
    lines.append(f"[bold]Sequence:[/bold]      {data.get('task_seq', '?')}")
    lines.append(f"[bold]Source:[/bold]        {data.get('source_name', '?')}")
    lines.append(f"[bold]Status:[/bold]        {_format_status(status)}")
    lines.append(f"[bold]Phase:[/bold]         {_rich_esc(data.get('phase_label', '') or data.get('phase', ''))}")
    if data.get("message"):
        lines.append(f"[bold]Message:[/bold]      {_rich_esc(data['message'])}")
    if data.get("error"):
        lines.append(f"[bold]Error:[/bold]        [bold red]{_rich_esc(data['error'])}[/bold red]")

    lines.append("")
    lines.append(f"[bold]Progress:[/bold]      {_format_progress(data)}")
    lines.append(f"[bold]Document size:[/bold]  {_format_size(data.get('document_size_bytes'))}")
    lines.append(f"[bold]Chunks:[/bold]        {data.get('processed_chunks', 0)} / {data.get('total_chunks', '?')}")

    lines.append("")
    lines.append(f"[bold]Created:[/bold]       {_format_timestamp_epoch(data.get('created_at'))}")
    lines.append(f"[bold]Started:[/bold]       {_format_timestamp_epoch(data.get('started_at'))}")
    lines.append(f"[bold]Finished:[/bold]      {_format_timestamp_epoch(data.get('finished_at'))}")
    lines.append(f"[bold]Elapsed:[/bold]       {_format_elapsed(data.get('elapsed_seconds'))}")

    eta = data.get("eta_seconds")
    if eta is not None:
        lines.append(f"[bold]ETA:[/bold]           {_format_elapsed(eta)}")

    out.console.print(Panel("\n".join(lines), title=f"Task {data.get('task_seq', '?')}: {_rich_esc(data.get('source_name', '?'))}", border_style="cyan"))

    # -- Pipeline step progress -------------------------------------------
    progress_detail = data.get("progress_detail") or {}
    step_lines: List[str] = []

    main_progress = data.get("main_progress")
    main_label = data.get("main_label") or "Main pipeline"
    if main_progress is not None:
        try:
            pct = float(main_progress) * 100
            done = data.get("main_done_chunks", 0)
            total = data.get("total_chunks", "?")
            step_lines.append(f"  [bold]Main:[/bold]     {pct:.1f}%  ({done}/{total} chunks)  {main_label}")
        except (ValueError, TypeError):
            step_lines.append(f"  [bold]Main:[/bold]     {main_progress}  {main_label}")

    step9_progress = data.get("step9_progress")
    step9_label = data.get("step9_label") or "Step 9 (entity alignment)"
    if step9_progress is not None:
        try:
            pct = float(step9_progress) * 100
            step_lines.append(f"  [bold]Step 9:[/bold]    {pct:.1f}%  {step9_label}")
        except (ValueError, TypeError):
            step_lines.append(f"  [bold]Step 9:[/bold]    {step9_progress}  {step9_label}")

    step10_progress = data.get("step10_progress")
    step10_label = data.get("step10_label") or "Step 10 (entity merge)"
    if step10_progress is not None:
        try:
            pct = float(step10_progress) * 100
            step_lines.append(f"  [bold]Step 10:[/bold]   {pct:.1f}%  {step10_label}")
        except (ValueError, TypeError):
            step_lines.append(f"  [bold]Step 10:[/bold]   {step10_progress}  {step10_label}")

    if step_lines:
        out.console.print(Panel("\n".join(step_lines), title="Pipeline Steps", border_style="blue"))

    # -- Failed/repair windows --------------------------------------------
    failed = data.get("failed_window_indices") or []
    if failed:
        fail_errors = data.get("failed_window_errors") or []
        err_lines = [f"  [bold]{len(failed)} windows failed or missing:[/bold]"]
        for fe in fail_errors[:8]:
            wi = fe.get("window_index", "?")
            phase = fe.get("phase", "?")
            err_msg = fe.get("error", "")
            err_lines.append(f"    Window {wi} ({phase}): {err_msg}")
        if len(fail_errors) > 8:
            err_lines.append(f"    ... and {len(fail_errors) - 8} more")
        out.console.print(Panel("\n".join(err_lines), title="Failed Windows", border_style="red"))


@task.command("cancel")
@click.argument("task_id")
@click.option(
    "--yes", "-y",
    is_flag=True,
    help="Skip confirmation prompt.",
)
@click.pass_context
def task_cancel(ctx: click.Context, task_id: str, yes: bool) -> None:
    """Cancel and delete a task.

    TASK_ID can be a sequence number or a full task ID.
    """
    out = OutputManager(ctx)
    _ensure_connected(out)

    if not yes:
        if not out.is_json:
            click.confirm(f"Cancel task {task_id}?", abort=True)

    resp = _request("DELETE", f"/remember/tasks/{task_id}")

    if _is_error(resp):
        if "_network_error" in resp:
            out.error(
                f"Cannot reach server at http://127.0.0.1:16200",
                hint=_SERVER_HINT,
                code=NETWORK,
            )
        http_status = resp.get("_http_status")
        if http_status == 404:
            out.error(f"Task not found: {task_id}", code=NOT_FOUND)
        out.error(_error_message(resp), code=ERROR)

    data = resp.get("data", resp)
    message = data.get("message", "Task cancelled")

    if out.is_json:
        from ._output import json_result
        payload = json_result("task-cancel", data)
        click.echo(_json.dumps(payload, ensure_ascii=False, indent=2))
        return

    out.success(f"Task {task_id}: {message}")


@task.command("pause")
@click.argument("task_id")
@click.pass_context
def task_pause(ctx: click.Context, task_id: str) -> None:
    """Pause a running or queued task.

    TASK_ID can be a sequence number or a full task ID.
    """
    out = OutputManager(ctx)
    _ensure_connected(out)

    resp = _request("POST", f"/remember/tasks/{task_id}/pause", body={})

    if _is_error(resp):
        if "_network_error" in resp:
            out.error(
                f"Cannot reach server at http://127.0.0.1:16200",
                hint=_SERVER_HINT,
                code=NETWORK,
            )
        http_status = resp.get("_http_status")
        if http_status == 404:
            out.error(f"Task not found: {task_id}", code=NOT_FOUND)
        out.error(_error_message(resp), code=ERROR)

    data = resp.get("data", resp)
    message = data.get("message", "Pause requested")

    if out.is_json:
        from ._output import json_result
        payload = json_result("task-pause", data)
        click.echo(_json.dumps(payload, ensure_ascii=False, indent=2))
        return

    out.success(f"Task {task_id}: {message}")


@task.command("resume")
@click.argument("task_id")
@click.pass_context
def task_resume(ctx: click.Context, task_id: str) -> None:
    """Resume a paused task.

    TASK_ID can be a sequence number or a full task ID.
    """
    out = OutputManager(ctx)
    _ensure_connected(out)

    resp = _request("POST", f"/remember/tasks/{task_id}/resume", body={})

    if _is_error(resp):
        if "_network_error" in resp:
            out.error(
                f"Cannot reach server at http://127.0.0.1:16200",
                hint=_SERVER_HINT,
                code=NETWORK,
            )
        http_status = resp.get("_http_status")
        if http_status == 404:
            out.error(f"Task not found: {task_id}", code=NOT_FOUND)
        out.error(_error_message(resp), code=ERROR)

    data = resp.get("data", resp)
    message = data.get("message", "Task resumed")

    if out.is_json:
        from ._output import json_result
        payload = json_result("task-resume", data)
        click.echo(_json.dumps(payload, ensure_ascii=False, indent=2))
        return

    out.success(f"Task {task_id}: {message}")


@task.command("retry")
@click.argument("task_id")
@click.pass_context
def task_retry(ctx: click.Context, task_id: str) -> None:
    """Retry a failed or paused task.

    Re-queues only the failed or missing windows; already-processed
    windows are not re-run.

    TASK_ID can be a sequence number or a full task ID.
    """
    out = OutputManager(ctx)
    _ensure_connected(out)

    resp = _request("POST", f"/remember/tasks/{task_id}/retry", body={})

    if _is_error(resp):
        if "_network_error" in resp:
            out.error(
                f"Cannot reach server at http://127.0.0.1:16200",
                hint=_SERVER_HINT,
                code=NETWORK,
            )
        http_status = resp.get("_http_status")
        if http_status == 404:
            out.error(f"Task not found: {task_id}", code=NOT_FOUND)
        out.error(_error_message(resp), code=ERROR)

    data = resp.get("data", resp)
    message = data.get("message", "Retry requested")
    retry_windows = data.get("retry_windows", [])

    if out.is_json:
        from ._output import json_result
        payload = json_result("task-retry", data)
        click.echo(_json.dumps(payload, ensure_ascii=False, indent=2))
        return

    out.success(f"Task {task_id}: {message}")
    if retry_windows:
        out.console.print(f"  Retrying {len(retry_windows)} windows: {retry_windows}")


@task.command("resume-all")
@click.pass_context
def task_resume_all(ctx: click.Context) -> None:
    """Resume all paused tasks.

    Resumes every paused task in the queue, preserving original order.
    """
    out = OutputManager(ctx)
    _ensure_connected(out)

    resp = _request("POST", "/remember/tasks/resume-all", body={})

    if _is_error(resp):
        if "_network_error" in resp:
            out.error(
                f"Cannot reach server at http://127.0.0.1:16200",
                hint=_SERVER_HINT,
                code=NETWORK,
            )
        out.error(_error_message(resp), code=ERROR)

    data = resp.get("data", resp)
    resumed = data.get("resumed", [])
    skipped = data.get("skipped", [])
    count = data.get("count", len(resumed))

    if out.is_json:
        from ._output import json_result
        payload = json_result("task-resume-all", data)
        click.echo(_json.dumps(payload, ensure_ascii=False, indent=2))
        return

    out.success(f"Resumed {count} task(s)")
    if skipped:
        out.console.print(f"  [dim]Skipped {len(skipped)} task(s):[/dim]")
        for s in skipped:
            out.console.print(f"    [dim]- {s.get('task_id', '?')}: {s.get('message', '')} (status: {s.get('status', '?')})[/dim]")
