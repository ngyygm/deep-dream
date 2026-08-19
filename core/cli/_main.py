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

_VERSION = "2.0.0"


def _print_version(ctx: click.Context, param: click.Parameter, value: Any) -> None:
    """Print version and exit immediately (no heavy imports)."""
    if not value or ctx.resilient_parsing:
        return
    click.echo(f"deep-dream {_VERSION}")
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
    "benchmark":  "core.cli.cmd_benchmark:benchmark",
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
    "-v", "--verbose",
    is_flag=True,
    default=False,
    help="Enable verbose logging.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Show what would be done without executing.",
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


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    """Programmatic entry point used by ``python -m core.cli``."""
    try:
        cli(standalone_mode=False, args=argv)
    except SystemExit as exc:
        return int(exc.code) if exc.code is not None else OK
    except click.UsageError as exc:
        click.echo(f"Error: {exc.format_message()}", err=True)
        return ARGS
    return OK
