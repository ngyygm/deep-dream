"""``deep-dream version`` — show version information.

Designed to be **fast**: no heavy imports at module level. Everything
except ``click`` and ``sys`` is imported lazily inside the callback so
that ``--help`` and bare invocations return in well under 200 ms.
"""
from __future__ import annotations

import sys

import click


@click.command()
@click.pass_context
def version(ctx: click.Context) -> None:
    """Show version information."""
    from core.cli._output import OutputManager

    out = OutputManager(ctx)

    # --- Deep-Dream version ---------------------------------------------------
    # 单一版本来源：core.cli._main._resolve_version（包元数据优先，常量兜底），
    # 与 ``deep-dream --version`` 输出保持一致。
    import importlib.metadata

    from core.cli._main import _resolve_version

    dd_version = _resolve_version()

    # --- Python version -------------------------------------------------------
    py_version = (
        f"{sys.version_info.major}"
        f".{sys.version_info.minor}"
        f".{sys.version_info.micro}"
    )

    # --- Storage path ---------------------------------------------------------
    storage_path = str(ctx.obj.storage_root.resolve())

    # --- Assemble payload -----------------------------------------------------
    data = {
        "deep-dream": dd_version,
        "python": py_version,
        "storage_path": storage_path,
    }

    if out.is_json:
        out.result(data)
        return

    # Rich / plain-text output — import the heaviest deps only here.
    import sqlite3

    data["sqlite"] = sqlite3.sqlite_version

    click.echo(f"Deep-Dream v{data['deep-dream']}")
    click.echo(f"  Python:   {data['python']}")
    click.echo(f"  Storage:  {data['storage_path']}")
    click.echo(f"  SQLite:   {data['sqlite']}")

    # Optional dependency versions — displayed only in human-readable mode.
    for pkg in ("click", "rich"):
        try:
            data[pkg] = importlib.metadata.version(pkg)
            click.echo(f"  {pkg.capitalize():10s}{data[pkg]}")
        except Exception:
            pass
