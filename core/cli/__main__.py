"""Allow ``python -m core.cli`` invocation."""
from __future__ import annotations

import sys

from ._main import main

sys.exit(main())
