"""Dual Rich/JSON output layer for the Deep-Dream CLI.

* ``--json``  mode writes structured JSON envelopes to **stdout**.
* Normal mode writes Rich-formatted output to **stderr** so that
  piping ``stdout`` still works for data capture.
"""
from __future__ import annotations

import json
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence

import click

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.status import Status

    _HAS_RICH = True
except ImportError:  # pragma: no cover
    _HAS_RICH = False


# ------------------------------------------------------------------
# Output mode enum
# ------------------------------------------------------------------

class OutputMode(Enum):
    RICH = "rich"
    JSON = "json"
    QUIET = "quiet"


# ------------------------------------------------------------------
# Formatting helpers
# ------------------------------------------------------------------

def format_timestamp(ts: Optional[str]) -> str:
    """Format an ISO timestamp for human-readable display."""
    if not ts:
        return ""
    try:
        dt = datetime.fromisoformat(ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.strftime("%Y-%m-%d %H:%M:%S %Z")
    except (ValueError, TypeError):
        return str(ts)


def format_confidence(conf: Any) -> str:
    """Format a confidence value (0-1 float) as a percentage string."""
    if conf is None:
        return ""
    try:
        return f"{float(conf) * 100:.1f}%"
    except (ValueError, TypeError):
        return str(conf)


# ------------------------------------------------------------------
# JSON envelope builder
# ------------------------------------------------------------------

def _command_name(ctx: Optional[click.Context]) -> str:
    """Derive a stable command name (``group subcommand``) from the context chain.

    Walks from the leaf command up to (but excluding) the root group, whose
    ``info_name`` is the program invocation rather than a real command. Returns
    e.g. ``"episode from-file"``, ``"db init-v15"``, ``"find"``.
    """
    if ctx is None:
        return ""
    try:
        root = ctx.find_root()
    except (RuntimeError, AttributeError):
        root = None
    parts: list[str] = []
    cur: Optional[click.Context] = ctx
    while cur is not None:
        if cur is root:
            # Skip the program-invocation level (e.g. "deep-dream").
            break
        name = cur.info_name
        if name:
            parts.append(name)
        cur = cur.parent
    parts.reverse()
    return " ".join(parts)


def json_result(
    command: str,
    data: Any,
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a standard JSON result envelope (spec 5.3/14.2).

    ``meta`` is nested under the ``meta`` key — never flattened to the top
    level — so callers can pass ``{"graph_id": ..., "count": ...}`` and get
    the canonical ``{success, command, data, meta}`` shape.
    """
    payload: Dict[str, Any] = {
        "success": True,
        "command": command,
        "data": data,
    }
    if meta:
        payload["meta"] = meta
    return payload


# ------------------------------------------------------------------
# OutputManager
# ------------------------------------------------------------------

class OutputManager:
    """Unified output handler that reads flags from the Click context.

    Parameters
    ----------
    ctx:
        A ``click.Context`` whose params may include ``json``, ``quiet``,
        and ``no_color``.
    """

    def __init__(self, ctx: click.Context) -> None:
        self._ctx: Optional[click.Context] = ctx
        # Resolve the stable command name once (e.g. "episode from-file").
        self._command: str = _command_name(ctx)
        # Walk up the context chain to find root group flags (--json, --quiet, --no-color).
        # These are declared on the root `cli` group but subcommands only see their own params.
        root_params: Dict[str, Any] = {}
        cur = ctx
        while cur is not None:
            for k, v in (cur.params or {}).items():
                root_params.setdefault(k, v)
            # Also check _click_params stored on CliContext by root group
            obj = getattr(cur, "obj", None)
            if obj is not None:
                cp = getattr(obj, "_click_params", None)
                if cp:
                    for k, v in cp.items():
                        root_params.setdefault(k, v)
            cur = cur.parent

        json_flag = root_params.get("json") or root_params.get("json_output")
        quiet_flag = root_params.get("quiet")
        self._mode = OutputMode.JSON if json_flag else (
            OutputMode.QUIET if quiet_flag else OutputMode.RICH
        )
        self._no_color: bool = bool(root_params.get("no_color"))
        self._console: Optional[Console] = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def command(self) -> str:
        """The resolved command name for this invocation (``group subcommand``)."""
        return self._command

    @property
    def graph_id(self) -> Optional[str]:
        """Active library / graph id, if the context carries one.

        Resolved lazily via ``CliContext.get_active_graph()`` when available;
        returns ``None`` for commands that never touch storage.
        """
        obj = getattr(self._ctx, "obj", None) if self._ctx is not None else None
        if obj is None:
            return None
        get_active = getattr(obj, "get_active_graph", None)
        if get_active is None:
            return None
        try:
            return get_active()
        except Exception:
            return None

    @property
    def is_json(self) -> bool:
        return self._mode == OutputMode.JSON

    @property
    def is_quiet(self) -> bool:
        return self._mode == OutputMode.QUIET

    @property
    def console(self) -> Console:
        """Rich Console writing to **stderr**.

        When stderr is not a TTY (e.g. output is piped/redirected) Rich
        auto-detects a 1-column width and mangles table titles into one
        character per line. We pass an explicit width in that case so
        titles/tables stay readable while piped. Interactive TTY behaviour
        is left untouched.
        """
        if self._console is None:
            if not _HAS_RICH:
                raise RuntimeError("rich is not installed")
            import os as _os

            kwargs: Dict[str, Any] = dict(
                stderr=True,
                no_color=self._no_color,
                highlight=False,
            )
            try:
                is_tty = sys.stderr.isatty()
            except (AttributeError, ValueError):
                is_tty = False
            if not is_tty:
                width = 0
                try:
                    width = int(_os.environ.get("COLUMNS", "0") or 0)
                except (TypeError, ValueError):
                    width = 0
                kwargs["width"] = width if width > 0 else 120
            self._console = Console(**kwargs)
        return self._console

    # ------------------------------------------------------------------
    # Core output methods
    # ------------------------------------------------------------------

    def result(self, data: Any, meta: Optional[Dict[str, Any]] = None) -> None:
        """Emit a result — JSON to stdout, Rich formatting to stderr.

        The JSON form follows the canonical envelope (spec 5.3/14.2):
        ``{success, command, data, meta}``. ``meta`` defaults to
        ``{"graph_id": <active library>}`` and is extended with anything the
        caller passes (e.g. ``{"count": N}``).
        """
        if self.is_json:
            resolved_meta = self._build_meta(meta)
            payload = json_result(self._command, data, meta=resolved_meta)
            click.echo(json.dumps(payload, ensure_ascii=False, indent=2))
            return
        if self.is_quiet:
            return
        # Rich mode: pretty-print to stderr
        self.console.print_json(json.dumps(data, ensure_ascii=False, indent=2))

    def table(
        self,
        title: str,
        columns: Sequence[str],
        rows: Sequence[Sequence[Any]],
    ) -> None:
        """Render a table — Rich Table to stderr or JSON array to stdout."""
        if self.is_json:
            data = [dict(zip(columns, row)) for row in rows]
            click.echo(json.dumps(data, ensure_ascii=False, indent=2))
            return
        if self.is_quiet:
            return
        from rich.markup import escape as _escape
        tbl = Table(title=title, show_header=True, header_style="bold")
        for col in columns:
            tbl.add_column(col)
        for row in rows:
            tbl.add_row(*(_escape(str(v)) for v in row))
        self.console.print(tbl)

    def panel(self, title: str, content: str) -> None:
        """Render a panel — Rich Panel to stderr or JSON to stdout.

        User-provided content is escaped so that square brackets, e.g.
        ``[文档元数据]``, are not misinterpreted as Rich markup.
        """
        if self.is_json:
            click.echo(json.dumps({"title": title, "content": content}, ensure_ascii=False, indent=2))
            return
        if self.is_quiet:
            return
        from rich.markup import escape
        self.console.print(Panel(escape(content), title=title))

    def error(
        self,
        message: str,
        hint: Optional[str] = None,
        code: int = 1,
    ) -> None:
        """Render an error and raise ``SystemExit(code)``.

        The JSON form follows the canonical error envelope (spec 5.3/14.2):
        ``{success: false, command, error: {code, message, hint}}`` where
        ``code`` is the numeric exit code.
        """
        if self.is_json:
            error_obj: Dict[str, Any] = {
                "code": int(code),
                "message": message,
            }
            if hint:
                error_obj["hint"] = hint
            payload = {
                "success": False,
                "command": self._command,
                "error": error_obj,
            }
            click.echo(json.dumps(payload, ensure_ascii=False, indent=2))
        elif not self.is_quiet:
            from rich.markup import escape as _esc
            self.console.print(f"[bold red]Error:[/bold red] {_esc(message)}")
            if hint:
                self.console.print(f"[dim]  Hint: {_esc(hint)}[/dim]")
        raise SystemExit(code)

    def _build_meta(self, meta: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Assemble the ``meta`` block, seeding ``graph_id`` from the active library.

        Caller-provided keys take precedence and may add fields such as
        ``count`` or ``elapsed_ms``. An empty meta is still returned as a dict
        so the envelope shape stays consistent.
        """
        resolved: Dict[str, Any] = {}
        gid = self.graph_id
        if gid:
            resolved["graph_id"] = gid
        if meta:
            resolved.update(meta)
        return resolved

    def success(self, message: str) -> None:
        """Render a success message."""
        if self.is_json:
            click.echo(json.dumps({"success": True, "message": message}, ensure_ascii=False, indent=2))
            return
        if self.is_quiet:
            return
        from rich.markup import escape as _esc
        self.console.print(f"[bold green]✓[/bold green] {_esc(message)}")

    @contextmanager
    def spinner(self, text: str):
        """Context manager that shows a Rich status spinner on stderr."""
        if self.is_json or self.is_quiet or not _HAS_RICH:
            yield
            return
        status = Status(text, console=self.console)
        status.start()
        try:
            yield
        finally:
            status.stop()
