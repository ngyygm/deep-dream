"""Deep-Dream CLI package — Click-based command line interface.

Usage::

    from core.cli import cli, main

    # Programmatic invocation
    sys.exit(main(["--version"]))
"""
from __future__ import annotations


def __getattr__(name: str):
    """Lazy re-export so that ``from core.cli import cli`` does not
    trigger ``core.__init__`` at module load time.

    This keeps ``--version`` and ``--help`` fast by deferring the heavy
    ``core`` package initialisation until a real command is executed.
    """
    if name in ("cli", "main"):
        from ._main import cli, main
        globals()["cli"] = cli
        globals()["main"] = main
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
