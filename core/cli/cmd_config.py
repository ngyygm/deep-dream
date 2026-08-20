"""``config`` command group — view and manage Deep-Dream configuration.

Subcommands
-----------
show   Display resolved configuration (API keys redacted by default).
get    Get a specific config value by dot-path.
set    Set a config value (with confirmation).

All commands respect the global ``--json`` / ``--quiet`` / ``--no-color``
flags handled by :class:`OutputManager`.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import click

from ._ctx import CliContext
from ._exit_codes import ARGS, NOT_FOUND
from ._output import OutputManager


# ------------------------------------------------------------------
# Dot-path helpers
# ------------------------------------------------------------------

# Keys whose values should be redacted when displaying config.
_SECRET_KEYS = frozenset({
    "api_key",
    "secret",
    "password",
    "token",
})

# Paths that expect integer values.
_INT_PATHS = frozenset({
    "port",
    "chunking.window_size",
    "chunking.overlap",
    "llm.context_window_tokens",
    "llm.max_tokens",
    "llm.max_concurrency",
    "storage.vector_dim",
    "runtime.concurrency.queue_workers",
    "runtime.concurrency.window_workers",
    "runtime.retry.queue_max_retries",
    "runtime.retry.queue_retry_delay_seconds",
    "pipeline.extraction.prompt_episode_max_chars",
    "pipeline.search.max_similar_entities",
    "pipeline.search.content_snippet_length",
    "pipeline.search.relation_content_snippet_length",
})

# Paths that expect float values (0.0-1.0 range).
_FLOAT_PATHS = frozenset({
    "pipeline.search.similarity_threshold",
    "pipeline.search.relation_endpoint_jaccard_threshold",
    "pipeline.search.relation_endpoint_embedding_threshold",
    "pipeline.search.jaccard_search_threshold",
    "pipeline.search.embedding_name_search_threshold",
    "pipeline.search.embedding_full_search_threshold",
    "pipeline.relation_endpoint_jaccard_threshold",
    "pipeline.relation_endpoint_embedding_threshold",
    "pipeline.similarity_threshold",
})

# Paths that expect boolean values.
_BOOL_PATHS = frozenset({
    "llm.think",
    "runtime.task.load_cache_memory",
    "runtime.integrity.auto_check_documents",
})


def _get_nested(d: Dict[str, Any], path: str) -> Any:
    """Retrieve a value from a nested dict using a dot-separated *path*."""
    keys = path.split(".")
    for k in keys:
        if isinstance(d, dict):
            d = d.get(k)
        else:
            return None
    return d


def _set_nested(d: Dict[str, Any], path: str, value: Any) -> None:
    """Set a value in a nested dict using a dot-separated *path*.

    Intermediate dictionaries are created via ``setdefault`` as needed.
    """
    keys = path.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def _redact_key(key: str) -> bool:
    """Return *True* if *key* looks like a secret field name."""
    return key.lower() in _SECRET_KEYS


def _redact_config(cfg: Dict[str, Any], show_secrets: bool = False) -> Dict[str, Any]:
    """Return a deep copy of *cfg* with secret values redacted."""
    out: Dict[str, Any] = {}
    for k, v in cfg.items():
        if isinstance(v, dict):
            out[k] = _redact_config(v, show_secrets)
        elif _redact_key(k) and not show_secrets:
            out[k] = "***" if v else v
        else:
            out[k] = v
    return out


def _flatten_config(
    cfg: Dict[str, Any],
    prefix: str = "",
) -> list[tuple[str, Any]]:
    """Flatten a nested dict into ``[(dot_path, value), ...]``."""
    items: list[tuple[str, Any]] = []
    for k, v in cfg.items():
        full = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            items.extend(_flatten_config(v, full))
        else:
            items.append((full, v))
    return items


def _coerce_value(path: str, raw: str) -> Any:
    """Coerce the string *raw* to the expected Python type for *path*."""
    if path in _INT_PATHS:
        try:
            return int(raw)
        except (ValueError, TypeError):
            raise click.BadParameter(
                f"Expected an integer for '{path}', got: {raw!r}"
            )
    if path in _FLOAT_PATHS:
        try:
            val = float(raw)
        except (ValueError, TypeError):
            raise click.BadParameter(
                f"Expected a number for '{path}', got: {raw!r}"
            )
        if not (0.0 <= val <= 1.0):
            raise click.BadParameter(
                f"Value for '{path}' must be between 0.0 and 1.0, got: {val}"
            )
        return val
    if path in _BOOL_PATHS:
        low = raw.strip().lower()
        if low in ("true", "1", "yes", "on"):
            return True
        if low in ("false", "0", "no", "off"):
            return False
        raise click.BadParameter(
            f"Expected true/false for '{path}', got: {raw!r}"
        )
    # Special case: "auto" is accepted for worker counts.
    if path in (
        "runtime.concurrency.window_workers",
        "runtime.concurrency.queue_workers",
    ) and raw.strip().lower() == "auto":
        return "auto"
    return raw


def _resolve_config_path(ctx: click.Context) -> str:
    """Extract the ``--config`` path from the Click context chain."""
    # Walk up to find the root context that has _click_params.
    cur = ctx
    while cur.parent is not None:
        cur = cur.parent
    params = getattr(ctx.obj, "_click_params", None) or {}
    return params.get("config", "service_config.json")


def _load_raw_config(config_path: str) -> Dict[str, Any]:
    """Load the raw JSON config file without merging defaults."""
    path = Path(config_path)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_raw_config(config_path: str, cfg: Dict[str, Any]) -> None:
    """Write *cfg* back to the JSON config file."""
    path = Path(config_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
        f.write("\n")


# ------------------------------------------------------------------
# Click group
# ------------------------------------------------------------------

@click.group()
def config() -> None:
    """View and manage configuration."""
    pass


# ------------------------------------------------------------------
# config show
# ------------------------------------------------------------------

@config.command()
@click.option(
    "--secrets",
    is_flag=True,
    default=False,
    help="Show API keys and other secrets (redacted by default).",
)
@click.pass_context
def show(ctx: click.Context, secrets: bool) -> None:
    """Display resolved configuration."""
    obj: CliContext = ctx.obj
    out = OutputManager(ctx)

    config_path = _resolve_config_path(ctx)
    obj.load_config(config_path)
    resolved = obj.config

    displayed = _redact_config(resolved, show_secrets=secrets)

    if out.is_json:
        payload = {
            "success": True,
            "config_path": str(Path(config_path).resolve()),
            "data": displayed,
        }
        click.echo(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    if out.is_quiet:
        return

    # Rich mode: show path header then key-value table.
    out.console.print(
        f"[dim]Config file:[/dim] {Path(config_path).resolve()}"
    )
    out.console.print()

    rows = _flatten_config(displayed)
    columns = ("Key", "Value")
    table_rows: list[list[str]] = []
    for key, value in rows:
        if value is None:
            formatted = "(not set)"
        elif isinstance(value, bool):
            formatted = "true" if value else "false"
        elif isinstance(value, (int, float)):
            formatted = str(value)
        else:
            formatted = str(value)
        table_rows.append([key, formatted])

    out.table("Configuration", columns, table_rows)


# ------------------------------------------------------------------
# config get
# ------------------------------------------------------------------

@config.command()
@click.argument("key")
@click.pass_context
def get(ctx: click.Context, key: str) -> None:
    """Get a config value by dot-path (e.g. llm.model)."""
    obj: CliContext = ctx.obj
    out = OutputManager(ctx)

    config_path = _resolve_config_path(ctx)
    obj.load_config(config_path)

    value = _get_nested(obj.config, key)
    if value is None:
        # Distinguish "explicitly null" from "key not found" by checking
        # if the path leads into the dict at all.
        parent_path = ".".join(key.split(".")[:-1])
        child_key = key.split(".")[-1]
        parent = _get_nested(obj.config, parent_path) if parent_path else obj.config
        if not isinstance(parent, dict) or child_key not in parent:
            out.error(
                f"Key not found: {key}",
                hint="Use 'deep-dream config show' to see all available keys.",
                code=NOT_FOUND,
            )
            return  # unreachable; error raises SystemExit

    if out.is_json:
        payload = {
            "success": True,
            "key": key,
            "value": value,
        }
        click.echo(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    if out.is_quiet:
        if value is not None:
            click.echo(str(value))
        return

    # Rich mode: show the value with the key.
    from rich.markup import escape as _rich_esc
    if value is None:
        out.console.print(f"[dim]{_rich_esc(key)}:[/dim] [dim](not set)[/dim]")
    elif _redact_key(key.split(".")[-1]):
        out.console.print(f"{_rich_esc(key)}: [bold red]***[/bold red]")
    else:
        out.console.print(f"{_rich_esc(key)}: [bold]{_rich_esc(str(value))}[/bold]")


# ------------------------------------------------------------------
# config set
# ------------------------------------------------------------------

@config.command("set")
@click.argument("key")
@click.argument("value")
@click.option(
    "--yes",
    is_flag=True,
    default=False,
    help="Skip confirmation prompt.",
)
@click.pass_context
def set_value(ctx: click.Context, key: str, value: str, yes: bool) -> None:
    """Set a config value (with confirmation)."""
    obj: CliContext = ctx.obj
    out = OutputManager(ctx)

    config_path = _resolve_config_path(ctx)

    # Coerce the string value to the appropriate type.
    try:
        coerced = _coerce_value(key, value)
    except click.BadParameter as exc:
        out.error(str(exc), code=ARGS)
        return  # unreachable

    # Read the raw config file (not merged with defaults) so we only
    # write back user-specified values.
    raw = _load_raw_config(config_path)
    old_value = _get_nested(raw, key)

    if old_value == coerced:
        if out.is_json:
            click.echo(json.dumps({
                "success": True,
                "message": "Value unchanged",
                "key": key,
                "value": coerced,
            }, ensure_ascii=False, indent=2))
        elif not out.is_quiet:
            out.console.print(
                f"[dim]{key} is already set to {coerced!r}. No change.[/dim]"
            )
        return

    # Confirmation prompt (unless --yes).
    if not yes:
        display_old = old_value if not _redact_key(key.split(".")[-1]) else "***"
        display_new = coerced if not _redact_key(key.split(".")[-1]) else "***"
        if not out.is_json:
            out.console.print(
                f"  [bold]{key}[/bold]: {display_old!r} -> {display_new!r}"
            )
        if not click.confirm("Apply this change?", default=False):
            if out.is_json:
                click.echo(json.dumps({
                    "success": False,
                    "message": "Cancelled",
                }, ensure_ascii=False, indent=2))
            raise SystemExit(0)

    # Apply and save.
    _set_nested(raw, key, coerced)
    _save_raw_config(config_path, raw)

    # Invalidate the cached config so next access reloads.
    obj._config = None
    obj._config_path = None

    if out.is_json:
        click.echo(json.dumps({
            "success": True,
            "message": f"Updated {key}",
            "key": key,
            "old_value": old_value,
            "new_value": coerced,
            "config_path": str(Path(config_path).resolve()),
        }, ensure_ascii=False, indent=2))
    elif not out.is_quiet:
        out.success(f"Updated {key} -> {coerced!r}")
        out.console.print(f"[dim]Written to {Path(config_path).resolve()}[/dim]")
