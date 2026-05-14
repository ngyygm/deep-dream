#!/usr/bin/env python3
"""
Dispatch helpers for Deep Dream MCP Server.

Shared argument extraction, validation, and graph ID resolution utilities
used by handler modules.
"""

from .transport import _resolve_graph_id


def _arg(args, key, default=None):
    return args.get(key, default)


def _req(args, key):
    """Get required argument with clear error if missing."""
    val = args.get(key)
    if val is None:
        raise ValueError(f"Missing required parameter: {key}")
    if isinstance(val, str) and not val.strip():
        raise ValueError(f"Parameter '{key}' must not be empty")
    return val


def _gid(args):
    """Extract per-call graph_id for HTTP routing. Removes it from args to avoid passing it to API body."""
    return _resolve_graph_id(args)


def _validate_absolute_id(value, param_name="absolute_id"):
    """Check that a value looks like an absolute_id (version ID), not a family_id."""
    if not value or not isinstance(value, str):
        raise ValueError(f"{param_name} is required (use get_entity to find the current absolute_id for an entity)")
    # family_ids start with "ent_" or "rel_" while absolute_ids are UUIDs or longer
    for prefix in ("ent_", "rel_"):
        if value.startswith(prefix) and "-" not in value:
            which = "entity" if prefix == "ent_" else "relation"
            raise ValueError(
                f"'{value}' looks like a family_id, but {param_name} requires an absolute_id (version ID). "
                f"Use get_entity(family_id='{value}') to find the current absolute_id." if prefix == "ent_" else
                f"'{value}' looks like a relation family_id, but {param_name} requires an absolute_id (version ID). "
                f"Use get_entity(family_id=...) for entity absolute_ids."
            )


def _validate_family_id(value, param_name="family_id"):
    """Check that a value looks like a family_id, not an absolute_id (UUID)."""
    if not value or not isinstance(value, str):
        raise ValueError(f"{param_name} is required")
    # UUIDs contain hyphens in 8-4-4-4-12 pattern — family_ids don't
    if len(value) == 36 and value.count("-") == 4:
        raise ValueError(
            f"'{value[:8]}...' looks like an absolute_id (UUID), but {param_name} requires a family_id (e.g. 'ent_abc123' or 'rel_abc123'). "
            f"Use get_entity_by_absolute_id(absolute_id='{value}') if you need to access by version ID."
        )
