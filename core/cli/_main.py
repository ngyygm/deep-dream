"""Root Click group for the Deep-Dream CLI.

All heavy imports (core.remember, core.storage, etc.) are deferred so
that ``--help`` and ``--version`` return in under 200 ms.

The ``core`` package's ``__init__.py`` eagerly imports the entire
pipeline, so we must avoid importing anything under ``core.*`` at
module level here.  ``CliContext`` and command modules are imported
lazily inside callbacks.
"""
from __future__ import annotations

import importlib
import sys
from typing import Any

import click

from ._exit_codes import ARGS, OK


# ------------------------------------------------------------------
# Version — no core.* imports needed
# ------------------------------------------------------------------

# Fallback only when the package is not installed (no metadata available).
# The canonical version lives in pyproject.toml and is read at call time
# via importlib.metadata.version('deep-dream').
_VERSION_FALLBACK = "0.2.0"


def _version_from_pyproject() -> str | None:
    """Read the ``[project] version`` field from pyproject.toml.

    Second-tier source used when the package is not installed (no metadata).
    Returns ``None`` if pyproject.toml cannot be found, parsed, or lacks the
    version key, so the caller can fall back to the hard-coded constant.
    """
    try:
        import tomllib
        from pathlib import Path

        # Walk up from this file to locate the project root pyproject.toml.
        here = Path(__file__).resolve()
        for candidate in [here.parent.parent.parent, *here.parent.parent.parent.parents]:
            pp = candidate / "pyproject.toml"
            if pp.is_file():
                with pp.open("rb") as fh:
                    data = tomllib.load(fh)
                ver = data.get("project", {}).get("version")
                if isinstance(ver, str) and ver.strip():
                    return ver.strip()
                return None
    except Exception:
        return None
    return None


def get_version() -> str:
    """Resolve the installed package version, falling back honestly.

    Resolution order (so ``pyproject.toml`` remains the single source of truth):

    1. ``importlib.metadata.version('deep-dream')`` — the installed metadata.
    2. Parse ``[project] version`` from ``pyproject.toml`` when the package
       is not installed (e.g. running from a checkout).
    3. The hard-coded ``_VERSION_FALLBACK`` constant, qualified with
       ``(uninstalled; from fallback)`` so the provenance is honest.
    """
    try:
        from importlib.metadata import version as _pkg_version

        return _pkg_version("deep-dream")
    except Exception:  # pragma: no cover - metadata missing in dev checkouts
        pyproject_ver = _version_from_pyproject()
        if pyproject_ver:
            return pyproject_ver
        return f"{_VERSION_FALLBACK} (uninstalled; from fallback)"


# Backwards-compatible alias. Older code imports ``_VERSION`` as a constant;
# keep it as the resolved value so both name usages stay in sync.
_VERSION = get_version()


def _print_version(ctx: click.Context, param: click.Parameter, value: Any) -> None:
    """Print version and exit immediately (no heavy imports)."""
    if not value or ctx.resilient_parsing:
        return
    click.echo(f"deep-dream {get_version()}")
    ctx.exit()


# ------------------------------------------------------------------
# Lazy command loader
# ------------------------------------------------------------------

_LAZY_COMMANDS: dict[str, str] = {
    # meta / info
    "version":    "core.cli.cmd_version:version",
    "doctor":     "core.cli.cmd_doctor:doctor",
    "config":     "core.cli.cmd_config:config",
    "completion": "core.cli.cmd_completion:completion",
    # core retrieval
    "find":       "core.cli.cmd_find:find",
    "remember":   "core.cli.cmd_remember:remember",
    "explore":    "core.cli.cmd_explore:explore",
    # data
    "docs":       "core.cli.cmd_docs:docs",
    "concept":    "core.cli.cmd_concept:concept",
    "episode":    "core.cli.cmd_episode:episode",
    "relation":   "core.cli.cmd_relation:relation",
    # management
    "graph":      "core.cli.cmd_graph:graph",
    "vault":      "core.cli.cmd_vault:vault",
    "library":    "core.cli.cmd_library:library",
    # server
    "server":     "core.cli.cmd_server:server",
    "task":       "core.cli.cmd_task:task",
    # maintenance
    "db":         "core.cli.cmd_db:db",
    "sql":        "core.cli.cmd_sql:sql",
}


def _lazy_import(dotted_path: str):
    """Import an object from a dotted path like 'pkg.mod:attr'."""
    module_path, _, attr = dotted_path.partition(":")
    mod = importlib.import_module(module_path)
    return getattr(mod, attr)


class _LazyGroup(click.Group):
    """Click group that resolves commands from _LAZY_COMMANDS on demand."""

    def list_commands(self, ctx: click.Context) -> list[str]:
        return list(_LAZY_COMMANDS)

    def get_command(self, ctx: click.Context, cmd_name: str) -> click.Command | None:
        dotted = _LAZY_COMMANDS.get(cmd_name)
        if dotted is None:
            return None
        try:
            return _lazy_import(dotted)
        except ImportError:
            return None


# ------------------------------------------------------------------
# Root group
# ------------------------------------------------------------------

_HELP_TEXT = """\
Deep-Dream — document-first concept graph knowledge server.

\b
Quick start:
  deep-dream remember --file notes.md      Index a document
  deep-dream explore "machine learning"     Semantic exploration
  deep-dream find "neural network"          Search concepts
  deep-dream docs list                       List indexed documents
  deep-dream doctor                          Health check

Use 'deep-dream <command> --help' for command-specific options.
"""


@click.group(
    cls=_LazyGroup,
    invoke_without_command=False,
    context_settings=dict(
        help_option_names=["-h", "--help"],
        auto_envvar_prefix="DEEPDREAM",
        max_content_width=100,
    ),
    help=_HELP_TEXT,
)
@click.option(
    "--config",
    default="service_config.json",
    envvar="DEEPDREAM_CONFIG",
    show_default=True,
    help="Path to service_config.json.",
)
@click.option(
    "--json",
    "json_output",
    is_flag=True,
    default=False,
    help="Output structured JSON to stdout.",
)
@click.option(
    "--no-color",
    is_flag=True,
    default=False,
    help="Disable coloured output.",
)
@click.option(
    "-q", "--quiet",
    is_flag=True,
    default=False,
    help="Suppress non-essential output.",
)
@click.option(
    "--version",
    is_flag=True,
    callback=_print_version,
    expose_value=False,
    is_eager=True,
    help="Print version and exit.",
)
@click.pass_context
def cli(ctx: click.Context, **kwargs: Any) -> None:
    """Deep-Dream document-first concept graph CLI."""
    from ._ctx import CliContext  # deferred to keep --version/--help fast

    ctx.ensure_object(dict)
    ctx.obj = CliContext()
    # Store raw Click params so OutputManager can read --json/--quiet/--no-color
    ctx.obj._click_params = kwargs  # type: ignore[attr-defined]

    # If the user asked for JSON or quiet output, switch the process-wide
    # logging mode BEFORE any command runs, so that pipeline/embedding log
    # banners (core.utils._emit_log_line) are routed to stderr instead of
    # polluting stdout. core.utils._json_mode gates this routing.
    json_flag = kwargs.get("json_output") or kwargs.get("json")
    quiet_flag = kwargs.get("quiet")
    if json_flag or quiet_flag:
        import os

        os.environ["DEEPDREAM_JSON_OUTPUT"] = "1"
        try:
            import core.utils as _utils

            _utils._json_mode = True
        except Exception:
            # core.utils is optional here — the env var above already
            # makes _emit_log_line fall back to stderr routing.
            pass


def _emit_uncaught_error(exc: BaseException, *, json_mode: bool) -> int:
    """Render an uncaught exception through OutputManager and return its exit code.

    Used by ``main()`` so that catastrophic failures (e.g. ``sqlite3.Error``
    from a corrupt library) present a clean one-line error instead of a raw
    traceback. In JSON mode the canonical error envelope is emitted on stdout.
    """
    from ._exit_codes import ERROR
    from ._output import OutputManager, OutputMode

    code: int = ERROR
    try:
        import sqlite3

        if isinstance(exc, sqlite3.Error):
            message = f"Database error: {exc}"
        else:
            message = f"{type(exc).__name__}: {exc}"
    except Exception:
        message = f"{type(exc).__name__}: {exc}"

    try:
        out = OutputManager(ctx=None)
        out._mode = OutputMode.JSON if json_mode else OutputMode.RICH
        out.error(message, code=code)
    except SystemExit as sexit:
        return int(sexit.code) if sexit.code is not None else code
    except Exception:
        # Last-resort fallback: never let the error handler itself crash.
        import sys as _sys

        _sys.stderr.write(f"Error: {message}\n")
        return code
    return code


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    """Programmatic entry point used by ``python -m core.cli``."""
    import sqlite3

    # Determine JSON mode from argv so the error envelope matches the
    # requested output mode even when the failure happens before/around
    # command parsing.
    json_mode = bool(argv and ("--json" in argv))

    try:
        cli(standalone_mode=False, args=argv)
    except SystemExit as exc:
        return int(exc.code) if exc.code is not None else OK
    except click.UsageError as exc:
        click.echo(f"Error: {exc.format_message()}", err=True)
        return ARGS
    except sqlite3.Error as exc:
        return _emit_uncaught_error(exc, json_mode=json_mode)
    except Exception as exc:  # noqa: BLE001 — top-level safety net
        return _emit_uncaught_error(exc, json_mode=json_mode)
    return OK
