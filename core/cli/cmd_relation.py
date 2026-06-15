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
from ._exit_codes import NOT_FOUND, OK
from ._helpers import compact_text, relation_evidence, resolve_concept_id
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
        # A 0-0 sentinel is not a real range — render it as empty.
        if line_start == 0 and line_end == 0:
            lines = ""
        elif line_start is not None and line_end is not None:
            lines = f"{line_start}-{line_end}"
        elif line_start is not None:
            lines = str(line_start)
        else:
            lines = ""

        # Collapse whitespace/newlines before truncating so multi-line
        # source_text does not blow up the table row height.
        source_text = compact_text(row.get("source_text") or "", max_chars=120)

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
    type=click.IntRange(min=1),
    help="Maximum number of evidence rows to return (minimum 1).",
)
@click.pass_context
def evidence(ctx: click.Context, concept_a: str, concept_b: str, limit: int) -> None:
    """Find evidence linking two concepts via relation edges.

    Resolves CONCEPT_A and CONCEPT_B by name or family ID, then queries
    the relation-edge view for rows that mention both concepts.

    In Rich mode the output is a human-readable table.  Pass ``--json``
    on the root group to get structured JSON instead.
    """
    obj: CliContext = ctx.obj
    out = OutputManager(ctx)

    # Resolve the active graph ID from the root context params.
    root_params = ctx.parent.params if ctx.parent else {}
    config_path = root_params.get("config", "service_config.json")
    config = obj.load_config(config_path)
    graph_id = obj.get_active_graph()

    # Acquire storage and resolve both concepts by name / family ID.
    with obj.get_storage(graph_id) as storage:
        a_id = resolve_concept_id(storage, concept_a)
        if a_id is None:
            out.error(
                f"Concept not found: {concept_a}",
                hint="Use 'deep-dream concept search <query>' to find concepts.",
                code=NOT_FOUND,
            )
            return  # unreachable
        b_id = resolve_concept_id(storage, concept_b)
        if b_id is None:
            out.error(
                f"Concept not found: {concept_b}",
                hint="Use 'deep-dream concept search <query>' to find concepts.",
                code=NOT_FOUND,
            )
            return  # unreachable

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
