"""
Relations blueprint package — Relation CRUD, search, path finding, and domain ops.

Split from the original monolithic relations.py into focused sub-modules:
  _search  — unified find, candidate search, relation search, path finding
  _crud    — create, update, delete, version lookup, batch ops
  _domain  — redirect, confidence, contradiction, invalidation, graph stats, traversal
"""
from __future__ import annotations

import logging
import re as _re
from concurrent.futures import ThreadPoolExecutor

from flask import Blueprint

relations_bp = Blueprint('relations', __name__)

# Shared pool for find_unified — avoids thread creation/destruction per request
_shared_pool = ThreadPoolExecutor(max_workers=3, thread_name_prefix="find-unified")

logger = logging.getLogger(__name__)

# Pre-compiled regex for stripping parenthetical annotations from entity names
_PAREN_ANNOTATION_RE = _re.compile(r'[（(][^）)]+[）)]')  # keep in sync with core/remember/helpers.py

# ---------------------------------------------------------------------------
# Import sub-modules so their @relations_bp.route decorators execute at import
# time.  Each sub-module imports ``relations_bp`` from this package.
# ---------------------------------------------------------------------------
from core.server.blueprints.relations import _search   # noqa: E402, F401

__all__ = ["relations_bp", "_shared_pool"]
