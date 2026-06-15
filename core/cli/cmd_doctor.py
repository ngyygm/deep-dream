"""``deep-dream doctor`` — inspect configuration, storage health, and API reachability.

Runs a series of diagnostic checks and reports results as a Rich table
(in human mode) or structured JSON (with ``--json``).
"""
from __future__ import annotations

import json as _json
import os
import sqlite3
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

import click

from ._exit_codes import ERROR, OK


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _human_bytes(n: int) -> str:
    """Return a human-friendly byte string."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} {unit}"
        n /= 1024  # type: ignore[assignment]
    return f"{n:.1f} PB"


def _check_icon(ok: bool) -> str:
    """Return a check/cross icon suitable for Rich markup."""
    return "[bold green]✓[/bold green]" if ok else "[bold red]✗[/bold red]"


def _plain_icon(ok: bool) -> str:
    """Return a plain-text check/cross (no Rich markup)."""
    return "✓" if ok else "✗"


# ------------------------------------------------------------------
# Individual checks
# ------------------------------------------------------------------

def _check_storage(storage_root: Path) -> Dict[str, Any]:
    """Check that the storage directory exists and report its size."""
    exists = storage_root.is_dir()
    total_size = 0
    file_count = 0
    if exists:
        try:
            for f in storage_root.rglob("*"):
                if f.is_file():
                    total_size += f.stat().st_size
                    file_count += 1
        except OSError:
            pass
    return {
        "path": str(storage_root.resolve()),
        "exists": exists,
        "size_bytes": total_size,
        "size_human": _human_bytes(total_size),
        "file_count": file_count,
    }


def _check_config(config_path: str) -> Dict[str, Any]:
    """Check whether the config file exists and is loadable."""
    p = Path(config_path)
    exists = p.is_file()
    loadable = False
    error: Optional[str] = None
    if exists:
        try:
            raw = p.read_text(encoding="utf-8")
            _json.loads(raw)
            loadable = True
        except Exception as exc:
            error = str(exc)
    return {
        "path": str(p.resolve()),
        "exists": exists,
        "loadable": loadable,
        "error": error,
    }


def _check_llm(config: Dict[str, Any]) -> Dict[str, Any]:
    """Try to instantiate the LLM client from config settings."""
    llm_cfg = config.get("llm") or {}
    api_key = llm_cfg.get("api_key")
    model = llm_cfg.get("model", "gpt-4")
    base_url = llm_cfg.get("base_url")
    try:
        from core.llm.client import LLMClient  # deferred
        client = LLMClient(
            api_key=api_key,
            model_name=model,
            base_url=base_url,
            context_window_tokens=llm_cfg.get("context_window_tokens"),
        )
        # If we got here, instantiation succeeded.
        configured = True
        error: Optional[str] = None
        # Attempt a lightweight connectivity test if base_url is set.
        reachable = False
        if base_url:
            import socket
            from urllib.parse import urlparse
            parsed = urlparse(base_url)
            host = parsed.hostname or "localhost"
            port = parsed.port or (443 if parsed.scheme == "https" else 80)
            try:
                with socket.create_connection((host, port), timeout=3):
                    reachable = True
            except OSError as exc:
                error = f"Connection refused: {exc}"
    except Exception as exc:
        configured = False
        reachable = False
        error = str(exc)
        client = None  # type: ignore[assignment]
    return {
        "configured": configured,
        "model": model,
        "base_url": base_url,
        "reachable": reachable,
        "error": error,
    }


def _check_embedding(config: Dict[str, Any]) -> Dict[str, Any]:
    """Check whether an embedding model is available."""
    emb_cfg = config.get("embedding") or {}
    model = emb_cfg.get("model")
    device = emb_cfg.get("device", "cpu")
    available = False
    error: Optional[str] = None
    try:
        import contextlib
        import sys

        from core.storage.embedding import EmbeddingClient  # deferred
        # The EmbeddingClient constructor prints a model-load banner
        # ("加载HuggingFace embedding模型…") via the pipeline logger, which
        # routes to stdout in non-JSON mode. That would corrupt the JSON
        # envelope on stdout, so redirect stdout -> stderr for the duration
        # of construction + the smoke-test encode. The banner is still
        # visible to humans (on stderr) but never hits stdout.
        with contextlib.redirect_stdout(sys.stderr):
            # Attempt to create the client (may download model on first run).
            client = EmbeddingClient(model_name=model, device=device)
            # A quick smoke test: encode a short string.
            vec = client.encode("health check")
        available = vec is not None and len(vec) > 0
    except Exception as exc:
        error = str(exc)
    return {
        "model": model,
        "device": device,
        "available": available,
        "error": error,
    }


def _check_api(api_base: str) -> Dict[str, Any]:
    """HTTP GET /health on the API server."""
    url = f"{api_base.rstrip('/')}/health"
    result: Dict[str, Any] = {
        "url": url,
        "available": False,
        "response": None,
        "error": None,
    }
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=3) as resp:
            body = resp.read().decode("utf-8")
            result["response"] = _json.loads(body)
            result["available"] = True
    except (OSError, urllib.error.URLError, TimeoutError, _json.JSONDecodeError) as exc:
        result["error"] = str(exc)
    return result


def _check_graphs(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """List all graphs with their stats."""
    try:
        from core.server.registry import GraphRegistry  # deferred
        registry = GraphRegistry(
            config.get("storage_path", "./library"),
            config,
        )
        return registry.list_graphs_info()
    except Exception as exc:
        return [{"error": str(exc)}]


def _check_integrity(config: Dict[str, Any]) -> Dict[str, Any]:
    """Run V1.5 integrity validation on the library DB (no file checks).

    Mirrors how ``cmd_db`` resolves the DB path (``library.db`` first, then
    legacy ``graph.db``) and wires the validator. Degrades to ``'unknown'``
    when the DB or validator cannot be reached, so a corrupt DB never silently
    passes *and* a validator bug never crashes doctor.
    """
    storage_path = config.get("storage_path", "./library")
    result: Dict[str, Any] = {
        "db_path": None,
        "total_violations": None,
        "by_issue": {},
        "ok": None,
        "error": None,
    }

    # Resolve the DB file the same way cmd_db does.
    db_path: Optional[str] = None
    for name in ("library.db", "graph.db"):
        candidate = os.path.join(storage_path, name)
        if os.path.exists(candidate):
            db_path = candidate
            break
    if db_path is None:
        db_path = os.path.join(storage_path, "library.db")
    result["db_path"] = db_path

    if not os.path.exists(db_path):
        result["error"] = f"Database not found: {db_path}"
        return result

    try:
        from core.storage.sqlite.integrity import validate_all  # deferred
        conn = sqlite3.connect(db_path)
        try:
            violations = validate_all(
                conn,
                library_path=storage_path,
                include_file_checks=False,
            )
        finally:
            conn.close()

        total = len(violations)
        # Aggregate top categories by issue type.
        by_issue: Dict[str, int] = {}
        for v in violations:
            issue = v.get("issue") or "unknown"
            by_issue[issue] = by_issue.get(issue, 0) + 1
        # Keep the noisiest categories for the summary detail.
        top = dict(
            sorted(by_issue.items(), key=lambda kv: kv[1], reverse=True)[:5]
        )
        result["total_violations"] = total
        result["by_issue"] = top
        result["ok"] = total == 0
    except Exception as exc:
        # Validator failure degrades to 'unknown' rather than crashing doctor.
        result["error"] = str(exc)
    return result





# ------------------------------------------------------------------
# Click command
# ------------------------------------------------------------------

@click.command()
@click.option(
    "--api-base",
    default="http://127.0.0.1:16200/api/v1",
    show_default=True,
    help="API base URL for health check.",
)
@click.pass_context
def doctor(ctx: click.Context, api_base: str) -> None:
    """Inspect configuration, storage health, and API reachability."""
    from ._output import OutputManager
    from ._ctx import CliContext

    out = OutputManager(ctx)
    cli_ctx: CliContext = ctx.obj

    # Load config from the path stored on the Click root context.
    root_params = ctx.parent.params if ctx.parent else {}
    config_path = root_params.get("config", "service_config.json")
    config = cli_ctx.load_config(config_path)
    storage_root = cli_ctx.storage_root

    # ---- Run checks ----
    storage = _check_storage(storage_root)
    config_status = _check_config(config_path)
    llm = _check_llm(config)
    embedding = _check_embedding(config)
    api = _check_api(api_base)
    graphs = _check_graphs(config)
    integrity = _check_integrity(config)

    # ---- Overall verdict (computed before output) ----
    cfg_ok = config_status["exists"] and config_status["loadable"]
    # Integrity ok is None when the validator could not run (degraded to
    # 'unknown'); only a definitive False (violations found) should fail
    # the overall verdict so a corrupt DB can no longer report all-green.
    integrity_ok = integrity.get("ok")
    all_ok = (
        storage["exists"]
        and cfg_ok
        and llm["configured"]
        and embedding["available"]
        and api["available"]
        and integrity_ok is not False
    )

    data = {
        "storage": storage,
        "config": config_status,
        "llm": llm,
        "embedding": embedding,
        "api": api,
        "graphs": graphs,
        "integrity": integrity,
        "graph_count": len([g for g in graphs if "error" not in g]),
        "overall_ok": all_ok,
    }

    # ---- Output ----
    if out.is_json:
        payload = {
            "success": all_ok,
            "command": out.command or "doctor",
            "data": data,
        }
        click.echo(_json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        # Rich human-readable output
        _render_human(out, data)

    # Exit non-zero when any check failed so callers/scripts can detect it.
    if not all_ok:
        raise SystemExit(ERROR)


# ------------------------------------------------------------------
# Rich rendering
# ------------------------------------------------------------------

def _render_human(out: "OutputManager", data: Dict[str, Any]) -> None:  # noqa: F821
    """Render the doctor report as Rich panels and a table."""
    from rich.table import Table
    from rich.markup import escape as _rich_esc

    # -- Summary table ------------------------------------------------
    table = Table(title="Deep-Dream Doctor", show_header=True, header_style="bold")
    table.add_column("Check", style="cyan", min_width=20)
    table.add_column("Status", justify="center", min_width=6)
    table.add_column("Detail", min_width=30)

    # Storage
    s = data["storage"]
    table.add_row(
        "Storage directory",
        _check_icon(s["exists"]),
        f"{s['path']}  ({s['size_human']}, {s['file_count']} files)"
        if s["exists"] else f"{s['path']}  (not found)",
    )

    # Config
    c = data["config"]
    cfg_ok = c["exists"] and c["loadable"]
    cfg_detail = c["path"]
    if c["exists"] and not c["loadable"]:
        cfg_detail += f"  (parse error: {c['error']})"
    table.add_row(
        "Config file",
        _check_icon(cfg_ok),
        cfg_detail,
    )

    # LLM
    llm = data["llm"]
    llm_ok = llm["configured"]
    llm_detail_parts: list[str] = [f"model={llm['model']}"]
    if llm["base_url"]:
        llm_detail_parts.append(f"url={llm['base_url']}")
    if llm["reachable"]:
        llm_detail_parts.append("reachable")
    if llm["error"]:
        llm_detail_parts.append(f"error: {llm['error']}")
    table.add_row(
        "LLM client",
        _check_icon(llm_ok),
        "  ".join(llm_detail_parts),
    )

    # Embedding
    emb = data["embedding"]
    emb_ok = emb["available"]
    emb_detail_parts: list[str] = []
    if emb["model"]:
        emb_detail_parts.append(f"model={emb['model']}")
    emb_detail_parts.append(f"device={emb['device']}")
    if emb["error"]:
        emb_detail_parts.append(f"error: {emb['error']}")
    table.add_row(
        "Embedding",
        _check_icon(emb_ok),
        "  ".join(emb_detail_parts),
    )

    # API
    api = data["api"]
    table.add_row(
        "API server",
        _check_icon(api["available"]),
        api["url"] + ("" if api["available"] else f"  ({api['error']})"),
    )

    # Integrity
    integ = data["integrity"]
    integ_ok = integ.get("ok")
    if integ_ok is True:
        integ_icon = _check_icon(True)
    elif integ_ok is False:
        integ_icon = _check_icon(False)
    else:
        # 'unknown' — neutral icon rather than a hard fail.
        integ_icon = "[bold yellow]?[/bold yellow]"
    integ_parts: list[str] = []
    if integ.get("db_path"):
        integ_parts.append(f"db={integ['db_path']}")
    if integ.get("total_violations") is not None:
        integ_parts.append(f"violations={integ['total_violations']}")
        for issue, count in integ.get("by_issue", {}).items():
            integ_parts.append(f"{_rich_esc(str(issue))}={count}")
    if integ.get("error"):
        integ_parts.append(f"error: {_rich_esc(str(integ['error']))}")
    table.add_row(
        "Integrity",
        integ_icon,
        "  ".join(integ_parts) if integ_parts else "unknown",
    )

    out.console.print(table)

    # -- Graph list ---------------------------------------------------
    graphs = data["graphs"]
    if graphs and "error" not in graphs[0]:
        gtable = Table(title="Graphs", show_header=True, header_style="bold")
        gtable.add_column("Graph ID", style="cyan")
        gtable.add_column("Path", min_width=30)
        gtable.add_column("Concepts", justify="right")
        gtable.add_column("Episodes", justify="right")
        gtable.add_column("Relations", justify="right")
        for g in graphs:
            gtable.add_row(
                g.get("graph_id", "?"),
                g.get("path", ""),
                str(g.get("entity_count", "?")),
                str(g.get("episode_count", "?")),
                str(g.get("relation_count", "?")),
            )
        out.console.print(gtable)
    elif graphs and "error" in graphs[0]:
        out.console.print(f"[bold red]Graph list error:[/bold red] {_rich_esc(str(graphs[0]['error']))}")
    else:
        out.console.print("[dim]No graphs found.[/dim]")

    # -- Overall verdict ---------------------------------------------
    integ_ok = data["integrity"].get("ok")
    all_ok = (
        s["exists"]
        and cfg_ok
        and llm_ok
        and emb_ok
        and api["available"]
        and integ_ok is not False
    )
    if all_ok:
        out.console.print("\n[bold green]All checks passed.[/bold green]")
    else:
        out.console.print("\n[bold yellow]Some checks failed — see details above.[/bold yellow]")
