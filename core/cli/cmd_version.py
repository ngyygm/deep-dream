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
    # The canonical version lives in pyproject.toml and is read via
    # importlib.metadata at call time. The single shared fallback constant
    # lives in _main.py; get_version() resolves both paths.
    import importlib.metadata

    try:
        from core.cli._main import get_version

        dd_version = get_version()
    except Exception:
        dd_version = "unknown"

    # --- Python version -------------------------------------------------------
    py_version = (
        f"{sys.version_info.major}"
        f".{sys.version_info.minor}"
        f".{sys.version_info.micro}"
    )

    # --- Storage path ---------------------------------------------------------
    storage_path = str(ctx.obj.storage_root.resolve())

    # --- Dependency versions --------------------------------------------------
    # Gathered before the JSON branch so that ``--json`` exposes the same
    # fields as the human-readable output (sqlite / click / rich versions).
    import sqlite3

    data = {
        "deep-dream": dd_version,
        "python": py_version,
        "storage_path": storage_path,
        "sqlite": sqlite3.sqlite_version,
    }

    for pkg in ("click", "rich"):
        try:
            data[pkg] = importlib.metadata.version(pkg)
        except Exception:
            pass

    if out.is_json:
        out.result(data)
        return

    # Rich / plain-text output.
    click.echo(f"Deep-Dream v{data['deep-dream']}")
    click.echo(f"  Python:   {data['python']}")
    click.echo(f"  Storage:  {data['storage_path']}")
    click.echo(f"  SQLite:   {data['sqlite']}")

    for pkg in ("click", "rich"):
        if pkg in data:
            click.echo(f"  {pkg.capitalize():10s}{data[pkg]}")
