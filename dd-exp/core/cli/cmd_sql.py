"""``deep-dream sql`` -- execute read-only SQL against the graph database.

Only ``SELECT`` and ``WITH`` (CTE) queries are allowed.  The storage layer
performs a second validation pass internally via
:func:`core.storage.sqlite.agent_query.validate_readonly_sql`.

All commands respect the global ``--json`` / ``--quiet`` / ``--no-color``
flags handled by :class:`OutputManager`.
"""
from __future__ import annotations

from typing import Optional

import click

from ._ctx import CliContext
from ._exit_codes import ARGS, ERROR
from ._output import OutputManager


# ------------------------------------------------------------------
# Click command
# ------------------------------------------------------------------

@click.command()
@click.argument("query", required=False)
@click.option(
    "--query",
    "query_opt",
    help="SQL query (alternative to passing query as argument).",
)
@click.option(
    "--limit",
    default=200,
    type=int,
    show_default=True,
    help="Maximum number of rows to return.",
)
@click.option(
    "--explain",
    is_flag=True,
    default=False,
    help="Include EXPLAIN QUERY PLAN output.",
)
@click.option(
    "--graph",
    default=None,
    help="Graph ID (defaults to the active library).",
)
@click.pass_context
def sql(
    ctx: click.Context,
    query: Optional[str],
    query_opt: Optional[str],
    limit: int,
    explain: bool,
    graph: Optional[str],
) -> None:
    """Run read-only SQL against the graph database.

    Only SELECT and WITH queries are allowed.  Use 'deep-dream db'
    commands for write operations and maintenance.

    \b
    Examples:
      deep-dream sql "SELECT * FROM entities LIMIT 5"
      deep-dream sql --query "SELECT count(*) FROM episodes" --explain
      deep-dream sql "WITH RECURSIVE ... SELECT ..." --limit 50
    """
    out = OutputManager(ctx)
    obj: CliContext = ctx.obj

    q = query or query_opt
    if not q:
        out.error(
            "Provide a SQL query.",
            hint='Example: deep-dream sql "SELECT count(*) FROM entities"',
            code=ARGS,
        )

    # Client-side validation: only SELECT / WITH at the top level.
    # The storage layer also validates via validate_readonly_sql().
    normalized = q.strip().upper()
    if not normalized.startswith("SELECT") and not normalized.startswith("WITH"):
        out.error(
            "Only SELECT queries are allowed.",
            hint="Use 'deep-dream db' commands for maintenance.",
            code=ARGS,
        )

    graph_id = obj.get_active_graph(graph)

    with obj.get_storage(graph_id) as storage:
        result = storage.read_sql(
            q,
            limit=limit,
            include_query_plan=explain,
        )

    # Check for storage-layer errors.
    if result.get("error"):
        out.error(
            f"SQL error: {result['error']}",
            hint="Check your query syntax.",
            code=ERROR,
        )

    # ---- JSON output ----
    if out.is_json:
        out.result(result, meta={"graph_id": graph_id})
        return

    # ---- Rich human-readable output ----
    columns = result.get("columns", [])
    rows = result.get("rows", [])
    row_count = result.get("row_count", len(rows))
    truncated = result.get("truncated", False)
    elapsed_ms = result.get("elapsed_ms", 0)
    query_plan = result.get("query_plan")

    if not columns and not rows:
        # Query returned no result set (e.g. PRAGMA that was blocked).
        out.success("Query executed (no rows returned).")
        return

    # Render the result table.
    table_rows = []
    for row in rows:
        table_rows.append([row.get(col, "") for col in columns])

    title = f"SQL Result ({row_count} row{'s' if row_count != 1 else ''}"
    if truncated:
        title += f", truncated to {limit}"
    title += f", {elapsed_ms:.1f} ms)"

    out.table(title, columns, table_rows)

    # Render EXPLAIN QUERY PLAN if requested.
    if query_plan:
        plan_columns = list(query_plan[0].keys()) if query_plan else []
        plan_rows = [[r.get(c, "") for c in plan_columns] for r in query_plan]
        out.table("Query Plan", plan_columns, plan_rows)
