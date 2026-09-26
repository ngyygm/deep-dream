"""Exit code constants for the Deep-Dream CLI."""
from __future__ import annotations

OK: int = 0
"""Successful execution."""

ERROR: int = 1
"""General error."""

ARGS: int = 2
"""Invalid arguments or usage error."""

NETWORK: int = 4
"""Network connectivity error."""

NOT_FOUND: int = 5
"""Requested resource not found."""
