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
    import importlib.metadata

    try:
        dd_version = importlib.metadata.version("deep-dream")
    except Exception:
        # Package not installed via pip / metadata missing — fall back to
        # the constant defined in _main.py so we stay in sync.
        try:
            from core.cli._main import _VERSION

            dd_version = _VERSION
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
