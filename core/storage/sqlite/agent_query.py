"""Agent read-only SQL sandbox for V1.5 schema."""
from __future__ import annotations

import re
import sqlite3
import time
from typing import Any, Optional


_SQL_COMMENT_RE = re.compile(r"(--[^\r\n]*|/\*.*?\*/)", re.DOTALL)
_SQL_SINGLE_QUOTE_RE = re.compile(r"'(?:''|[^'])*'")
_SQL_DOUBLE_QUOTE_RE = re.compile(r'"(?:""|[^"])*"')
_SQL_BLOCKED_TOKENS = {
    "insert", "update", "delete", "replace", "create", "drop", "alter",
    "attach", "detach", "vacuum", "pragma", "reindex", "analyze",
    "begin", "commit", "rollback", "savepoint", "release",
}


def _sql_for_validation(sql: str) -> str:
    cleaned = _SQL_COMMENT_RE.sub(" ", sql or "")
    cleaned = _SQL_SINGLE_QUOTE_RE.sub("''", cleaned)
    cleaned = _SQL_DOUBLE_QUOTE_RE.sub('""', cleaned)
    return cleaned.strip()


def validate_readonly_sql(sql: str) -> str:
    raw = (sql or "").strip()
    if not raw:
        raise ValueError("SQL cannot be empty")
    cleaned = _sql_for_validation(raw)
    if ";" in cleaned.rstrip(";"):
        raise ValueError("Only single read-only SQL statement allowed")
    cleaned_no_tail = cleaned.rstrip().rstrip(";").strip()
    lowered = cleaned_no_tail.lower()
    if lowered.startswith("explain"):
        if not re.match(r"^explain\s+query\s+plan\s+(select|with)\b", lowered, re.DOTALL):
            raise ValueError("EXPLAIN only allows EXPLAIN QUERY PLAN SELECT/WITH")
    elif not re.match(r"^(select|with)\b", lowered, re.DOTALL):
        raise ValueError("Only SELECT / WITH / EXPLAIN QUERY PLAN allowed")
    tokens = set(re.findall(r"\b[a-z_][a-z0-9_]*\b", lowered))
    blocked = sorted(tokens.intersection(_SQL_BLOCKED_TOKENS))
    if blocked:
        raise ValueError(f"Read-only SQL blocked tokens: {', '.join(blocked)}")
    return raw.rstrip().rstrip(";").strip()


def execute_readonly_query(conn: sqlite3.Connection, sql: str,
                           params: Any = None, *, limit: int = 200,
                           timeout_seconds: float = 5.0,
                           include_query_plan: bool = False) -> dict:
    validated = validate_readonly_sql(sql)
    start = time.time()

    # Use a separate read-only connection
    db_path = conn.execute("PRAGMA database_list").fetchone()[2]
    ro_conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5)
    ro_conn.row_factory = sqlite3.Row
    ro_conn.set_progress_handler(lambda: (_ := time.time()) - start > timeout_seconds and (
        __import__('sqlite3').OperationalError("timeout") or None
    ), 100)

    try:
        cursor = ro_conn.execute(validated, params or ())
        rows = cursor.fetchmany(limit + 1)
        truncated = len(rows) > limit
        rows = rows[:limit]
        columns = [d[0] for d in cursor.description] if cursor.description else []
        result_rows = []
        for row in rows:
            result_rows.append({col: _json_safe(row[col]) for col in columns})
    except Exception as exc:
        return {"error": str(exc), "columns": [], "rows": [], "row_count": 0,
                "truncated": False, "elapsed_ms": (time.time() - start) * 1000}
    finally:
        ro_conn.close()

    query_plan = None
    if include_query_plan:
        try:
            plan_rows = conn.execute(f"EXPLAIN QUERY PLAN {validated}", params or ()).fetchall()
            query_plan = [dict(r) for r in plan_rows]
        except Exception:
            pass

    return {
        "columns": columns,
        "rows": result_rows,
        "row_count": len(result_rows),
        "truncated": truncated,
        "elapsed_ms": round((time.time() - start) * 1000, 1),
        "query_plan": query_plan,
    }


def _json_safe(value):
    if isinstance(value, bytes):
        return f"<BLOB {len(value)} bytes>"
    return value
