"""``deep-dream remember`` -- ingest text or a file into the concept graph.

This is the primary data-ingestion command. It reads text (either from a
file via ``--file`` or inline via ``--text``) and runs the full extraction
pipeline: chunking, entity extraction, relation extraction, and alignment.

All heavy imports (pipeline, registry, storage) are deferred to the command
body so that ``--help`` returns instantly.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import click

from ._ctx import CliContext
from ._exit_codes import ARGS
from ._output import OutputManager


# ------------------------------------------------------------------
# Summary helper
# ------------------------------------------------------------------

def _extract_summary(result: Dict[str, Any]) -> Dict[str, Any]:
    """Pull human-readable counts from the pipeline result dict.

    The pipeline returns a dict with at least the following keys:

    * ``episode_id``          -- ID of the created episode (None on failure)
    * ``document_version_id`` -- version identifier
    * ``chunks_processed``    -- number of text windows processed
    * ``storage_path``        -- on-disk storage location
    * ``entities``            -- entity count created/updated
    * ``relations``           -- relation count created/updated

    Optional keys that are surfaced when present:

    * ``warnings``   -- list of warning dicts
    * ``errors``     -- list of error dicts (partial-failure mode)
    """
    summary: Dict[str, Any] = {}

    # Core identifiers
    if result.get("episode_id"):
        summary["Episode"] = result["episode_id"]
    if result.get("document_version_id"):
        summary["Document version"] = result["document_version_id"]

    # Processing counts
    chunks = result.get("chunks_processed", 0)
    if chunks:
        summary["Chunks processed"] = chunks

    entities = result.get("entities")
    if entities is not None:
        summary["Entities"] = entities

    relations = result.get("relations")
    if relations is not None:
        summary["Relations"] = relations

    # Warnings / errors -- compact representation
    warnings = result.get("warnings")
    if warnings:
        summary["Warnings"] = len(warnings)

    errors = result.get("errors")
    if errors:
        summary["Errors"] = len(errors)

    return summary


# ------------------------------------------------------------------
# Click command
# ------------------------------------------------------------------

@click.command()
@click.option(
    "-f",
    "--file",
    "file_path",
    type=click.Path(exists=True),
    help="File to remember.",
)
@click.option(
    "-t",
    "--text",
    help="Inline text to remember.",
)
@click.option(
    "-s",
    "--source",
    default=None,
    help="Source label [default: filename or cli:text].",
)
@click.option(
    "--encoding",
    default="utf-8",
    show_default=True,
    help="File encoding.",
)
@click.option(
    "--graph",
    default=None,
    help="Graph ID [default: library].",
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    help="Show processing details.",
)
@click.pass_context
def remember(
    ctx: click.Context,
    file_path: str | None,
    text: str | None,
    source: str | None,
    encoding: str,
    graph: str | None,
    verbose: bool,
) -> None:
    """Ingest text or a file into the concept graph.

    Provide exactly one of --file or --text.

    \b
    Examples:
      deep-dream remember --file notes.md
      deep-dream remember --text "Key insight about quantum computing"
      deep-dream remember --file doc.md --source "research-paper" -v
    """
    out = OutputManager(ctx)
    cli_ctx: CliContext = ctx.obj

    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    if not file_path and text is None:
        out.error(
            "Provide --file or --text.",
            hint="Example: deep-dream remember --file notes.md",
            code=ARGS,
        )
        return  # unreachable; error raises SystemExit

    # ------------------------------------------------------------------
    # Resolve graph
    # ------------------------------------------------------------------
    graph_id = cli_ctx.get_active_graph(graph)

    # ------------------------------------------------------------------
    # Load text
    # ------------------------------------------------------------------
    if text is not None:
        source_label = source or "cli:text"
        source_document = source_label
    else:
        fp = Path(file_path)
        text = fp.read_text(encoding=encoding)
        source_label = source or fp.name
        source_document = str(fp.resolve())

    # ------------------------------------------------------------------
    # Run pipeline
    # ------------------------------------------------------------------
    registry = cli_ctx.get_registry()
    processor = registry.get_processor(graph_id)

    if out.is_json:
        result = processor.remember_text(
            text,
            doc_name=source_label,
            verbose=verbose,
            source_document=source_document,
        )
        out.result(
            {"result": result},
            meta={"graph_id": graph_id},
        )
        return

    # Rich / plain-text mode
    with out.spinner(f"Remembering {source_label}..."):
        result = processor.remember_text(
            text,
            doc_name=source_label,
            verbose=verbose,
            source_document=source_document,
        )

    # ------------------------------------------------------------------
    # Display summary
    # ------------------------------------------------------------------
    summary = _extract_summary(result)

    click.echo("", err=True)
    click.echo("  Summary:", err=True)
    for key, val in summary.items():
        click.echo(f"    {key}: {val}", err=True)

    # Surface warnings
    warnings = result.get("warnings")
    if warnings:
        click.echo("", err=True)
        click.echo(f"  Warnings ({len(warnings)}):", err=True)
        for w in warnings[:5]:
            msg = w if isinstance(w, str) else w.get("error", str(w))
            click.echo(f"    - {msg}", err=True)
        if len(warnings) > 5:
            click.echo(f"    ... and {len(warnings) - 5} more", err=True)

    # Surface errors (partial failure)
    errors = result.get("errors")
    if errors:
        click.echo("", err=True)
        click.echo(f"  Errors ({len(errors)}):", err=True)
        for e in errors[:5]:
            click.echo(f"    - {e}", err=True)
        if len(errors) > 5:
            click.echo(f"    ... and {len(errors) - 5} more", err=True)

    click.echo("", err=True)

    if result.get("episode_id"):
        out.success(f"Remembered {source_label}")
    else:
        click.echo("  Completed with issues.", err=True)
