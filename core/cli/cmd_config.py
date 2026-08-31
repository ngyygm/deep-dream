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
import os
import tempfile
import copy
from pathlib import Path
from typing import Any, Dict

import click

from ._ctx import CliContext
from ._exit_codes import ARGS, NOT_FOUND
from ._helpers import resolve_config_path
from ._output import OutputManager


# ------------------------------------------------------------------
# Dot-path helpers
# ------------------------------------------------------------------

# Keys whose values should be redacted when displaying config.
_SECRET_KEYS = frozenset({
    "api_key",
    "secret",
    "secret_key",
    "password",
    "token",
    "credential",
    "authorization",
    "private_key",
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
    "runtime.retry.queue_retry_delay_seconds",
    "monitor_refresh_seconds",
    "pipeline.relation_endpoint_jaccard_threshold",
    "pipeline.relation_endpoint_embedding_threshold",
    "pipeline.similarity_threshold",
})

# Paths that expect boolean values.
_BOOL_PATHS = frozenset({
    "llm.think",
    "llm.mock",
    "flask_threaded",
    "auto_port_fallback",
    "auth.enabled",
    "auth.strict_mode",
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
    # These are numeric generation limits, not credentials.  Do not hide
    # them merely because their names contain the substring ``token``.
    if str(key) in {"max_tokens", "context_window_tokens"}:
        return False
    normalized = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(key))
    compact = normalized.replace("_", "")
    return normalized in _SECRET_KEYS or any(
        token in compact
        for token in ("apikey", "secret", "password", "token", "credential", "authorization", "privatekey")
    )


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


def _default_for_path(path: str) -> tuple[bool, Any]:
    """按 dot-path 在 DEFAULTS 中查找叶子默认值，返回 ``(found, default)``。

    中间节点不是 dict 或末级 key 不存在时 ``found=False``（如
    ``llm.alignment.enabled``——DEFAULTS 的 llm.alignment 是空 dict 占位）。
    """
    from core.server.config import DEFAULTS

    node: Any = DEFAULTS
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return False, None
        node = node[part]
    return True, node


def _parse_bool(raw: str, path: str) -> bool:
    """Parse *raw* as a boolean CLI value or raise BadParameter."""
    low = raw.strip().lower()
    if low in ("true", "1", "yes", "on"):
        return True
    if low in ("false", "0", "no", "off"):
        return False
    raise click.BadParameter(
        f"Expected true/false for '{path}', got: {raw!r}"
    )


def _coerce_value(path: str, raw: str) -> Any:
    """Coerce the string *raw* to the expected Python type for *path*.

    类型来源（按序）：
      1. 显式 ``auto`` 哨兵（worker 数）；
      2. 旧手写 int/float/bool 三张表（含 DEFAULTS 之外的旧平铺 key，以及
         默认值为 None 但语义上是布尔的 key，如 auth.enabled）；
      3. DEFAULTS 推导——默认值为 bool/int/float 的 key 按默认值类型强转，
         默认值为 None 的 key 只把字面量 ``null`` 解析为 None（其余保持
         原串，防把 "12345" 这类 API key 误转成数字）；
      4. DEFAULTS 未收录的 key：字面量 true/false/null 解析为对应类型，
         其余保持原串。

    此前只有第 2 步，大量 DEFAULTS key（pipeline.remember.* 布尔开关、
    runtime.task.stall_timeout_seconds 等）不 coerc——``config set key
    false --yes`` 落库为 truthy 字符串 "false"（文档化的关闭开关静默失效），
    数值键落库 "900" 后在 ``max(60.0, "900")`` 处直接 TypeError。
    """
    # Worker counts accept the explicit ``auto`` sentinel.  Check this before
    # the integer table below; otherwise ``int('auto')`` raises and the
    # documented sentinel can never reach the runtime normalizer.
    if path in (
        "runtime.concurrency.window_workers",
        "runtime.concurrency.queue_workers",
    ) and raw.strip().lower() == "auto":
        return "auto"
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
        if path in {"runtime.retry.queue_retry_delay_seconds", "monitor_refresh_seconds"}:
            if not (0.0 <= val <= 3600.0):
                raise click.BadParameter(
                    f"Value for '{path}' must be between 0 and 3600, got: {val}"
                )
        elif not (0.0 <= val <= 1.0):
            raise click.BadParameter(
                f"Value for '{path}' must be between 0.0 and 1.0, got: {val}"
            )
        return val
    if path in _BOOL_PATHS:
        return _parse_bool(raw, path)

    found, default = _default_for_path(path)
    if found:
        if isinstance(default, bool):
            return _parse_bool(raw, path)
        if isinstance(default, int):
            try:
                return int(raw)
            except (ValueError, TypeError):
                raise click.BadParameter(
                    f"Expected an integer for '{path}', got: {raw!r}"
                )
        if isinstance(default, float):
            try:
                return float(raw)
            except (ValueError, TypeError):
                raise click.BadParameter(
                    f"Expected a number for '{path}', got: {raw!r}"
                )
        if default is None and raw.strip().lower() == "null":
            return None
        return raw

    low = raw.strip().lower()
    if low == "null":
        return None
    if low in ("true", "false"):
        return low == "true"
    return raw


def _load_raw_config(config_path: str) -> Dict[str, Any]:
    """Load the raw JSON config file without merging defaults."""
    path = Path(config_path)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_raw_config(config_path: str, cfg: Dict[str, Any]) -> None:
    """Atomically write *cfg* back to the JSON config file.

    A direct ``open(..., 'w')`` can leave a truncated configuration after a
    crash or a full disk.  Keep the old file intact until the fully flushed
    temporary file is ready, and preserve its permissions when replacing it.
    """
    path = Path(config_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    old_mode = path.stat().st_mode & 0o777 if path.exists() else 0o600
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        fchmod = getattr(os, "fchmod", None)
        if fchmod is not None:
            fchmod(fd, old_mode)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, path)
        try:
            dir_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except OSError:
            # Directory fsync is not available on every platform; the file
            # replacement is still atomic and durable enough for those hosts.
            pass
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


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

    config_path = resolve_config_path(ctx)
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
@click.option(
    "--secrets",
    is_flag=True,
    default=False,
    help="Show the raw value for a secret key (otherwise it is redacted).",
)
@click.pass_context
def get(ctx: click.Context, key: str, secrets: bool) -> None:
    """Get a config value by dot-path (e.g. llm.model)."""
    obj: CliContext = ctx.obj
    out = OutputManager(ctx)

    config_path = resolve_config_path(ctx)
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

    displayed_value = value if secrets or not _redact_key(key.split(".")[-1]) else ("***" if value else value)

    if out.is_json:
        payload = {
            "success": True,
            "key": key,
            "value": displayed_value,
        }
        click.echo(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    if out.is_quiet:
        if displayed_value is not None:
            click.echo(str(displayed_value))
        return

    # Rich mode: show the value with the key.
    from rich.markup import escape as _rich_esc
    if value is None:
        out.console.print(f"[dim]{_rich_esc(key)}:[/dim] [dim](not set)[/dim]")
    elif not secrets and _redact_key(key.split(".")[-1]):
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
@click.option(
    "--secrets",
    is_flag=True,
    default=False,
    help="Include raw secret values in the command result (avoid in CI logs).",
)
@click.pass_context
def set_value(ctx: click.Context, key: str, value: str, yes: bool, secrets: bool) -> None:
    """Set a config value (with confirmation)."""
    obj: CliContext = ctx.obj
    out = OutputManager(ctx)

    config_path = resolve_config_path(ctx)

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
    is_secret = _redact_key(key.split(".")[-1])
    display_old = old_value if secrets or not is_secret else ("***" if old_value else old_value)
    display_new = coerced if secrets or not is_secret else ("***" if coerced else coerced)

    if old_value == coerced:
        if out.is_json:
            click.echo(json.dumps({
                "success": True,
                "message": "Value unchanged",
                "key": key,
                "value": display_new,
            }, ensure_ascii=False, indent=2))
        elif not out.is_quiet:
            out.console.print(
                f"[dim]{key} is already set to {display_new!r}. No change.[/dim]"
            )
        return

    # Confirmation prompt (unless --yes).
    if not yes:
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

    # Validate the complete prospective configuration before replacing the
    # file.  Config files may intentionally be sparse while being assembled
    # through ``config set``; in that case validate all available fields but
    # temporarily treat an entirely absent LLM endpoint as mock-only for the
    # validation pass.  An explicitly malformed ``llm.mock`` is still
    # rejected by the schema.
    try:
        from core.server.config import (
            DEFAULTS,
            _deep_merge,
            _normalize_runtime_config,
            _validate_config,
        )

        validation_cfg = _normalize_runtime_config(
            _deep_merge(copy.deepcopy(DEFAULTS), raw)
        )
        validation_llm = validation_cfg.setdefault("llm", {})
        raw_llm = raw.get("llm") if isinstance(raw.get("llm"), dict) else {}
        if (
            "api_key" not in raw_llm
            and "base_url" not in raw_llm
            and "mock" not in raw_llm
        ):
            validation_llm["mock"] = True
        _validate_config(validation_cfg)
    except Exception as exc:
        out.error(f"配置校验失败: {exc}", code=ARGS)
        return  # unreachable; error raises SystemExit

    _save_raw_config(config_path, raw)

    # Invalidate the cached config so next access reloads.
    obj._config = None
    obj._config_path = None

    if out.is_json:
        click.echo(json.dumps({
            "success": True,
            "message": f"Updated {key}",
            "key": key,
            "old_value": display_old,
            "new_value": display_new,
            "config_path": str(Path(config_path).resolve()),
        }, ensure_ascii=False, indent=2))
    elif not out.is_quiet:
        out.success(f"Updated {key} -> {coerced!r}")
        out.console.print(f"[dim]Written to {Path(config_path).resolve()}[/dim]")
