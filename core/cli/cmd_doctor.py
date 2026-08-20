"""``deep-dream doctor`` — inspect configuration, storage health, and API reachability.

Runs a series of diagnostic checks and reports results as a Rich table
(in human mode) or structured JSON (with ``--json``).
"""
from __future__ import annotations

import json as _json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

import click



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


def _check_schema(storage_root: Path) -> Dict[str, Any]:
    """Check V1.5 schema completeness (tables/indexes/views + user_version).

    只读检查（sqlite_master + PRAGMA），坏库上也能安全运行并报出缺失，
    不会触发建表修复——修复由 LibraryManager 打开库时的启动自愈负责。
    """
    db_path: Optional[Path] = None
    for name in ("library.db", "graph.db"):
        candidate = storage_root / name
        if candidate.is_file():
            db_path = candidate
            break
    result: Dict[str, Any] = {
        "db_path": str((db_path or storage_root / "library.db").resolve()),
        "exists": db_path is not None,
        "ok": False,
        "user_version": None,
        "missing_tables": [],
        "missing_indexes": [],
        "missing_views": [],
        "error": None,
    }
    if db_path is None:
        return result

    import sqlite3  # deferred
    from core.storage.sqlite.schema_v15 import schema_health  # deferred

    conn = sqlite3.connect(str(db_path))
    try:
        result.update(schema_health(conn))
    except sqlite3.Error as exc:
        # 库文件损坏/非 SQLite 格式：诊断命令必须报出错误而不是自己炸掉
        result["error"] = str(exc)
    finally:
        conn.close()
    return result


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
        LLMClient(
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
        from core.storage.embedding import EmbeddingClient  # deferred
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
    schema = _check_schema(storage_root)
    config_status = _check_config(config_path)
    llm = _check_llm(config)
    embedding = _check_embedding(config)
    api = _check_api(api_base)
    graphs = _check_graphs(config)

    data = {
        "storage": storage,
        "schema": schema,
        "config": config_status,
        "llm": llm,
        "embedding": embedding,
        "api": api,
        "graphs": graphs,
        "graph_count": len([g for g in graphs if "error" not in g]),
    }

    # ---- Output ----
    if out.is_json:
        from ._output import json_result
        payload = json_result("doctor", data)
        click.echo(_json.dumps(payload, ensure_ascii=False, indent=2))
        return

    # Rich human-readable output
    _render_human(out, data)


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

    # DB schema（表齐全性 + user_version）
    sc = data["schema"]
    if not sc["exists"]:
        schema_detail = f"{sc['db_path']}  (database file not found)"
    elif sc.get("error"):
        # 库文件存在但读不出来（损坏/非 SQLite 格式）
        schema_detail = f"unreadable database: {sc['error']}"
    elif sc["ok"]:
        exp = sc["expected"]
        schema_detail = (
            f"tables {exp['tables']}/{exp['tables']}, "
            f"indexes {exp['indexes']}/{exp['indexes']}, "
            f"views {exp['views']}/{exp['views']}  "
            f"(user_version={sc['user_version']})"
        )
    else:
        missing_parts: list[str] = []
        for label, key in (("tables", "missing_tables"),
                           ("indexes", "missing_indexes"),
                           ("views", "missing_views")):
            if sc[key]:
                missing_parts.append(f"{label}: {', '.join(sc[key])}")
        schema_detail = (
            f"missing {'; '.join(missing_parts)}  "
            f"(user_version={sc['user_version']})"
        )
    table.add_row(
        "DB schema",
        _check_icon(sc["exists"] and sc["ok"]),
        _rich_esc(schema_detail),
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
    # schema：库文件不存在不拉低 verdict（全新安装尚未建库），
    # 但库存在而缺表/索引必须算失败——那正是搜索静默 0 结果的根因。
    schema_ok = (not sc["exists"]) or sc["ok"]
    all_ok = (
        s["exists"]
        and schema_ok
        and cfg_ok
        and llm_ok
        and emb_ok
        and api["available"]
    )
    if all_ok:
        out.console.print("\n[bold green]All checks passed.[/bold green]")
    else:
        out.console.print("\n[bold yellow]Some checks failed — see details above.[/bold yellow]")
