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

def json_result(
    command: str,
    data: Any,
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a standard JSON result envelope."""
    payload: Dict[str, Any] = {
        "success": True,
        "command": command,
        "data": data,
    }
    if meta:
        payload.update(meta)
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
    def is_json(self) -> bool:
        return self._mode == OutputMode.JSON

    @property
    def is_quiet(self) -> bool:
        return self._mode == OutputMode.QUIET

    @property
    def console(self) -> Console:
        """Rich Console writing to **stderr**."""
        if self._console is None:
            if not _HAS_RICH:
                raise RuntimeError("rich is not installed")
            self._console = Console(
                stderr=True,
                no_color=self._no_color,
                highlight=False,
            )
        return self._console

    # ------------------------------------------------------------------
    # Core output methods
    # ------------------------------------------------------------------

    def result(self, data: Any, meta: Optional[Dict[str, Any]] = None) -> None:
        """Emit a result — JSON to stdout, Rich formatting to stderr."""
        if self.is_json:
            payload = json_result("", data, meta=meta)
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
        """Render an error and raise ``SystemExit(code)``."""
        if self.is_json:
            payload = {"success": False, "error": message}
            if hint:
                payload["hint"] = hint
            click.echo(json.dumps(payload, ensure_ascii=False, indent=2))
        elif not self.is_quiet:
            from rich.markup import escape as _esc
            self.console.print(f"[bold red]Error:[/bold red] {_esc(message)}")
            if hint:
                self.console.print(f"[dim]  Hint: {_esc(hint)}[/dim]")
        raise SystemExit(code)

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
