"""
Entities blueprint package — Entity CRUD, search, relations, timeline, intelligence.

Split from the original monolithic entities.py into focused sub-modules:
  _search   — listing, SSE streaming, name lookup, profiles, recent activity
  _crud     — create, update, delete, batch ops, merge, isolated entities
  _versions — version queries, timeline, diff, patches, confidence, contradictions
"""
from __future__ import annotations

import logging
import re as _re
from concurrent.futures import ThreadPoolExecutor

from flask import Blueprint

entities_bp = Blueprint("entities", __name__)

# Shared pool for parallel queries (avoids per-request thread creation)
_shared_pool = ThreadPoolExecutor(max_workers=3, thread_name_prefix="ent-req")

logger = logging.getLogger(__name__)

# Pre-compiled regex for stripping parenthetical annotations from entity names
_CORE_NAME_RE = _re.compile(r'\s*[\(（].*?[\)）]\s*')

# ---------------------------------------------------------------------------
# Import sub-modules so their @entities_bp.route decorators execute at import
# time.  Each sub-module imports ``entities_bp`` from this package.
# ---------------------------------------------------------------------------
from core.server.blueprints.entities import _search   # noqa: E402, F401
from core.server.blueprints.entities import _crud      # noqa: E402, F401
from core.server.blueprints.entities import _versions   # noqa: E402, F401

__all__ = ["entities_bp", "_shared_pool"]
