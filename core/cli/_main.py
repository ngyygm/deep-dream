"""Root Click group for the Deep-Dream CLI.

``core/__init__.py`` is PEP 562 lazy, so the command modules can be
imported and registered here directly (the ``_LazyGroup`` indirection
that used to break the import cycle is gone).  Each ``cmd_*`` module
itself only pulls in stdlib + click/rich + ``core.cli`` internals at
module level; heavy imports (``core.remember``, ``core.storage``, …)
stay deferred inside command callbacks, keeping ``--help`` and
``--version`` fast.
"""
from __future__ import annotations

import importlib.metadata
from typing import Any

import click

from ._exit_codes import ARGS, OK
from .cmd_completion import completion
from .cmd_concept import concept
from .cmd_config import config
from .cmd_db import db
from .cmd_doctor import doctor
from .cmd_docs import docs
from .cmd_episode import episode
from .cmd_explore import explore
from .cmd_find import find
from .cmd_graph import graph
from .cmd_remember import remember
from .cmd_relation import relation
from .cmd_server import server
from .cmd_sql import sql
from .cmd_task import task
from .cmd_vault import vault
from .cmd_version import version


# ------------------------------------------------------------------
# Version — single source: installed package metadata, constant fallback
# ------------------------------------------------------------------

_VERSION = "0.2.0"  # 与 pyproject.toml 同步；仅在包未安装/元数据缺失时兜底


def _resolve_version() -> str:
    """单一版本来源：优先读安装包元数据（源自 pyproject.toml），异常回退常量。"""
    try:
        return importlib.metadata.version("deep-dream")
    except Exception:  # 未安装或元数据损坏
        return _VERSION


def _print_version(ctx: click.Context, param: click.Parameter, value: Any) -> None:
    """Print version and exit immediately (no heavy imports)."""
    if not value or ctx.resilient_parsing:
        return
    click.echo(f"deep-dream {_resolve_version()}")
    ctx.exit()


# ------------------------------------------------------------------
# Root group
# ------------------------------------------------------------------

class _OrderedCommandGroup(click.Group):
    """List subcommands in registration order.

    click 默认按字母序列出子命令，会改变既有的 ``--help`` 输出顺序；
    这里按注册顺序列出以保持输出不变。
    """

    def list_commands(self, ctx: click.Context) -> list[str]:
        return list(self.commands)


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
    cls=_OrderedCommandGroup,
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


# -- 命令注册（顺序即 --help 列表顺序）----------------------------------

cli.add_command(version)      # meta / info
cli.add_command(doctor)
cli.add_command(config)
cli.add_command(completion)
cli.add_command(find)         # core retrieval
cli.add_command(remember)
cli.add_command(explore)
cli.add_command(docs)         # data
cli.add_command(concept)
cli.add_command(episode)
cli.add_command(relation)
cli.add_command(graph)        # management
cli.add_command(vault)
cli.add_command(server)       # server
cli.add_command(task)
cli.add_command(db)           # maintenance
cli.add_command(sql)


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
