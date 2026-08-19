"""``deep-dream relation`` — relation evidence helpers.

Subcommands
-----------
evidence   Find evidence linking two concepts via relation edges.

All commands respect the global ``--json`` / ``--quiet`` / ``--no-color``
flags handled by :class:`OutputManager`.
"""
from __future__ import annotations

import json
from typing import Any

import click

from ._ctx import CliContext
from ._helpers import relation_evidence
from ._output import OutputManager


# ------------------------------------------------------------------
# Rich rendering helpers
# ------------------------------------------------------------------

def _render_evidence_rich(out: OutputManager, data: dict[str, Any]) -> None:
    """Render relation evidence as a Rich table."""
    evidence = data.get("evidence", [])
    if not evidence:
        out.console.print("[dim]No relation evidence found.[/dim]")
        return

    columns = ["Source", "Relation", "Entities", "Lines", "Evidence"]
    rows: list[list[str]] = []

    for row in evidence:
        source = row.get("title") or row.get("read_path") or ""
        relation_name = row.get("relation_name") or ""
        e1 = row.get("entity1_name") or ""
        e2 = row.get("entity2_name") or ""
        entities = f"{e1} <-> {e2}" if e1 and e2 else (e1 or e2)

        line_start = row.get("line_start")
        line_end = row.get("line_end")
        if line_start is not None and line_end is not None:
            lines = f"{line_start}-{line_end}"
        elif line_start is not None:
            lines = str(line_start)
        else:
            lines = ""

        source_text = row.get("source_text") or ""
        if len(source_text) > 120:
            source_text = source_text[:117] + "..."

        rows.append([source, relation_name, entities, lines, source_text])

    out.table("Relation Evidence", columns, rows)

    total = data.get("total", len(evidence))
    out.console.print(f"[dim]{total} evidence row(s)[/dim]")


# ------------------------------------------------------------------
# Click group
# ------------------------------------------------------------------

@click.group()
def relation() -> None:
    """Relation evidence helpers."""
    pass


# ------------------------------------------------------------------
# relation evidence
# ------------------------------------------------------------------

@relation.command()
@click.argument("concept_a")
@click.argument("concept_b")
@click.option(
    "--limit",
    default=50,
    show_default=True,
    type=int,
    help="Maximum number of evidence rows to return.",
)
@click.option(
    "--graph",
    default=None,
    help="Graph ID (defaults to the active library).",
)
@click.pass_context
def evidence(ctx: click.Context, concept_a: str, concept_b: str, limit: int, graph: str | None) -> None:
    """Find evidence linking two concepts via relation edges.

    Resolves CONCEPT_A and CONCEPT_B by name or family ID, then queries
    the relation-edge view for rows that mention both concepts.

    In Rich mode the output is a human-readable table.  Pass ``--json``
    on the root group to get structured JSON instead.
    """
    obj: CliContext = ctx.obj
    out = OutputManager(ctx)

    # Resolve the active graph ID from the root context params.
    root_params = ctx.find_root().params
    config_path = root_params.get("config", "service_config.json")
    obj.load_config(config_path)
    graph_id = obj.get_active_graph(graph)

    # Acquire storage and run the query.
    with obj.get_storage(graph_id) as storage:
        rows = relation_evidence(storage, concept_a, concept_b, limit=limit)

    data = {
        "evidence": rows,
        "total": len(rows),
    }

    # ---- JSON output ----
    if out.is_json:
        payload = {
            "success": True,
            "command": "relation evidence",
            "graph_id": graph_id,
            "data": data,
            "used": {
                "raw_files": False,
                "sqlite": True,
                "semantic": False,
                "graph_traversal": True,
                "api": False,
            },
        }
        click.echo(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    # ---- Quiet mode ----
    if out.is_quiet:
        return

    # ---- Rich mode ----
    _render_evidence_rich(out, data)
