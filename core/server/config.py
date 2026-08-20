"""
统一服务 API 配置加载：从 JSON 文件读取并返回配置字典，缺失项使用默认值。
"""
import json
from pathlib import Path
from typing import Any, Dict, Tuple

from core.exceptions import ConfigError


DEFAULTS = {
    "host": "0.0.0.0",
    "port": 5001,
    "flask_threaded": True,
    "monitor_refresh_seconds": 1.0,
    "auto_port_fallback": False,
    "log_mode": "detail",
    "storage_path": "./library",
    "storage": {
        "backend": "sqlite",
        "vector_dim": 1024,
    },
    "llm": {
        "api_key": None,
        "model": "gpt-4",
        "base_url": None,
        "think": False,
        # 离线测试用：mock=true 时 LLMClient 走模拟响应
        "mock": False,
        # 请求输入 prompt 的本地预检上限；与 service_config.json 中 llm.context_window_tokens 对应
        "context_window_tokens": 8000,
        "max_concurrency": 1,
        "max_tokens": None,
        "timeout_seconds": 300,
        "connect_timeout_seconds": 30,
        # 透传 OpenAI 兼容端点的额外 body（如 chat_template_kwargs.enable_thinking）
        "extra_body": {},
        # 对齐阶段专用模型（双模型管线）；enabled=false 时与主模型共用
        "alignment": {},
    },
    "embedding": {
        "model": None,
        "device": "cpu",
    },
    "chunking": {
        "window_size": 1000,
        "overlap": 200,
    },
    "runtime": {
        "concurrency": {
            "queue_workers": 1,
            "window_workers": "auto",
        },
        "retry": {
            "queue_max_retries": 2,
            "queue_retry_delay_seconds": 2,
        },
        "task": {
            "load_cache_memory": False,
            "stall_timeout_seconds": 600,
        },
        "integrity": {
            "auto_check_documents": True,
        },
    },
    "pipeline": {
        "search": {
            "similarity_threshold": 0.7,
            "max_similar_entities": 10,
            "content_snippet_length": 50,
            "relation_content_snippet_length": 50,
            "relation_endpoint_jaccard_threshold": 0.9,
            "relation_endpoint_embedding_threshold": 0.9,
            "jaccard_search_threshold": None,
            "embedding_name_search_threshold": None,
            "embedding_full_search_threshold": None,
        },
        "alignment": {
            "max_alignment_candidates": None,
        },
        "extraction": {
            "prompt_episode_max_chars": 2000,
        },
        "remember": {
            "alignment_policy": "conservative",
            "profile": "strong-v1",
            "preserve_source_language": False,
            "max_entities_per_window": 16,
            "max_relations_per_window": 24,
            "episode_slice_chars": 0,
            "family_write_gate_enabled": True,
        },
        "debug": {
            "distill_data_dir": None,
        },
    },
}


def resolve_embedding_model(embedding: Dict[str, Any]) -> Tuple[Any, Any, bool]:
    """
    从 embedding 配置解析出 (model_path, model_name, use_local)。
    优先使用单一字段 model：若为已存在的路径则视为本地模型，否则视为 HuggingFace 模型名（自动下载）。
    若未设置 model，则回退到 model_path / model_name / use_local（兼容旧配置）。
    """
    model = embedding.get("model")
    if model is not None and isinstance(model, str):
        model = model.strip()
        if model:
            path = Path(model).expanduser().resolve()
            if path.exists():
                return str(path), None, True
            return None, model, False
    model_path = embedding.get("model_path")
    model_name = embedding.get("model_name")
    use_local = bool(embedding.get("use_local", True))
    return model_path, model_name, use_local


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _normalize_runtime_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    归一新格式配置：runtime.concurrency / runtime.retry / runtime.task /
    pipeline.{search,alignment,extraction,remember,debug}。

    只认新格式（v0.2+）：旧平铺字段（remember_workers、pipeline.max_concurrent_windows、
    pipeline.similarity_threshold、prompt_memory_cache_max_chars 等）不再回读，
    也不再回填。worker 数受 llm.max_concurrency 钳制。
    """
    cfg = dict(config)
    runtime = dict(cfg.get("runtime") or {})
    conc = dict(runtime.get("concurrency") or {})
    retry = dict(runtime.get("retry") or {})
    task = dict(runtime.get("task") or {})
    pipeline = dict(cfg.get("pipeline") or {})
    debug = dict(pipeline.get("debug") or {})

    # 队列/窗口线程不超过 LLM 并发：max_concurrency=1 时整体按串行语义运行
    llm = cfg.get("llm") or {}
    max_llm_conc = llm.get("max_concurrency")
    cap = int(max_llm_conc) if max_llm_conc is not None else None

    def _resolve_worker_count(value, default: int, *, auto_default: int) -> int:
        if value is None:
            return int(default)
        if isinstance(value, str) and value.strip().lower() in {"auto", "default", ""}:
            return int(auto_default)
        return int(value)

    # 回填新结构。用户只需要设置 llm.max_concurrency；window_workers 默认自动跟随，
    # 用于让多个 episode 形成流水线，但最终 LLM 请求仍由全局闸门限制。
    _auto_windows = max(1, min(cap or 1, 3))
    conc["queue_workers"] = _resolve_worker_count(conc.get("queue_workers"), 1, auto_default=1)
    conc["window_workers"] = _resolve_worker_count(
        conc.get("window_workers"), _auto_windows, auto_default=_auto_windows)
    if max_llm_conc is not None:
        if cap >= 1:
            conc["queue_workers"] = min(conc["queue_workers"], cap)
            conc["window_workers"] = min(conc["window_workers"], cap)
    retry["queue_max_retries"] = int(
        retry.get("queue_max_retries") if retry.get("queue_max_retries") is not None else 2)
    retry["queue_retry_delay_seconds"] = float(
        retry.get("queue_retry_delay_seconds")
        if retry.get("queue_retry_delay_seconds") is not None else 2
    )
    task["load_cache_memory"] = bool(task.get("load_cache_memory", False))
    runtime["concurrency"] = conc
    runtime["retry"] = retry
    runtime["task"] = task
    cfg["runtime"] = runtime

    # Resolve distill_data_dir relative to storage_path so files land inside library/, not CWD
    distill_data_dir = debug.get("distill_data_dir")
    if distill_data_dir is not None:
        if not Path(distill_data_dir).is_absolute():
            storage_path = cfg.get("storage_path", "./library")
            resolved = Path(storage_path) / distill_data_dir
            # Skip if already resolved (path parts match storage_path prefix)
            parts = Path(distill_data_dir).parts
            sp_parts = Path(storage_path).parts
            if parts[:len(sp_parts)] != sp_parts:
                distill_data_dir = str(resolved)
        debug["distill_data_dir"] = distill_data_dir
    pipeline["debug"] = debug
    cfg["pipeline"] = pipeline

    return cfg


def merge_llm_alignment(llm: Dict[str, Any]) -> Dict[str, Any]:
    """
    合并步骤 6/7（对齐）专用 LLM 配置。
    - llm.alignment.enabled == false：关闭对齐专用通道，步骤 6/7 与 1–5 共用同一模型与并发策略。
    - enabled == true 或未写 enabled 但存在 base_url/api_key/model 等：启用对齐配置。
    - 并发只看 llm.max_concurrency；alignment 只覆盖模型/端点，不再单独配置并发。
    """
    if llm.get("alignment_enabled") is False:
        return {}

    nested = llm.get("alignment")
    if nested is True:
        nested = {"enabled": True}
    if nested is False:
        return {}
    if not isinstance(nested, dict):
        nested = {}

    if nested.get("enabled") is False:
        return {}

    def pick(nested_key: str, flat_key: str):
        v = nested.get(nested_key)
        if v is None and nested_key == "think":
            v = nested.get("think_mode")
        if v is not None:
            return v
        return llm.get(flat_key)

    out: Dict[str, Any] = {}
    mapping = (
        ("base_url", "alignment_base_url"),
        ("api_key", "alignment_api_key"),
        ("model", "alignment_model"),
        ("max_tokens", "alignment_max_tokens"),
        ("think", "alignment_think"),
        ("content_snippet_length", "alignment_content_snippet_length"),
        ("relation_content_snippet_length", "alignment_relation_content_snippet_length"),
        ("extra_body", "alignment_extra_body"),
    )
    for nk, fk in mapping:
        val = pick(nk, fk)
        if val is not None:
            out["think_mode" if nk == "think" else nk] = val

    if nested.get("enabled") is True:
        out["enabled"] = True
        return out

    if out:
        out["enabled"] = True
    return out


def _validate_config(config: Dict[str, Any]) -> None:
    """校验配置值合法性，不合法时抛出 ConfigError。"""
    errors: list = []

    port = config.get("port")
    if port is not None and not (1 <= int(port) <= 65535):
        errors.append(f"port 应在 1-65535 之间，当前值: {port}")

    llm = config.get("llm") or {}
    if not llm.get("api_key") and not llm.get("base_url") and llm.get("mock") is not True:
        errors.append("llm.api_key 或 llm.base_url 至少需要配置一个（离线测试可显式设置 llm.mock=true）")
    _cwt = llm.get("context_window_tokens")
    if _cwt is not None:
        try:
            _cwt_i = int(_cwt)
            if _cwt_i < 256:
                errors.append(f"llm.context_window_tokens 应 >= 256，当前值: {_cwt}")
        except (TypeError, ValueError):
            errors.append(f"llm.context_window_tokens 应为整数，当前值: {_cwt}")
    if llm.get("extra_body") is not None and not isinstance(llm.get("extra_body"), dict):
        errors.append("llm.extra_body 必须是 JSON object")

    chunking = config.get("chunking") or {}
    ws = chunking.get("window_size", 1000)
    ol = chunking.get("overlap", 200)
    if ws is not None and ol is not None and int(ol) >= int(ws):
        errors.append(f"chunking.overlap ({ol}) 必须小于 chunking.window_size ({ws})")

    pipeline = config.get("pipeline") or {}
    thresholds = [
        ("pipeline.similarity_threshold", pipeline.get("similarity_threshold")),
        ("pipeline.jaccard_search_threshold", pipeline.get("jaccard_search_threshold")),
        ("pipeline.embedding_name_search_threshold", pipeline.get("embedding_name_search_threshold")),
        ("pipeline.embedding_full_search_threshold", pipeline.get("embedding_full_search_threshold")),
    ]
    for name, val in thresholds:
        if val is not None and not (0.0 <= float(val) <= 1.0):
            errors.append(f"{name} 应在 0.0-1.0 之间，当前值: {val}")

    extraction = (pipeline.get("extraction") or {})
    _pmcmc = extraction.get("prompt_episode_max_chars", pipeline.get("prompt_episode_max_chars"))
    if _pmcmc is not None:
        try:
            _pmcmc_i = int(_pmcmc)
            if _pmcmc_i < 0:
                errors.append(
                    f"pipeline.extraction.prompt_episode_max_chars 应 >= 0，当前值: {_pmcmc}"
                )
        except (TypeError, ValueError):
            errors.append(
                "pipeline.extraction.prompt_episode_max_chars 应为整数"
                f"，当前值: {_pmcmc}"
            )

    if errors:
        raise ConfigError("配置校验失败:\n  " + "\n  ".join(errors))


def _warn_unknown_keys(user: Dict[str, Any]) -> None:
    """用户配置里出现 DEFAULTS 未覆盖的键时告警（P5：配置面收敛，防拼写错误静默失效）。"""
    import logging
    logger = logging.getLogger(__name__)
    for key in user:
        if key not in DEFAULTS:
            logger.warning("配置键未知（已忽略）: %r — 请对照 service_config.example.json", key)
    for section, defaults in DEFAULTS.items():
        user_section = user.get(section)
        if not isinstance(user_section, dict) or not isinstance(defaults, dict):
            continue
        for key in user_section:
            if key not in defaults:
                logger.warning(
                    "配置键未知（已忽略）: %s.%s — 请对照 service_config.example.json", section, key)


def load_config(config_path: str) -> Dict[str, Any]:
    """
    从 JSON 文件加载配置，与默认值合并。

    Args:
        config_path: 配置文件路径（如 service_config.json）

    Returns:
        合并后的配置字典，包含 host, port, storage_path, llm, embedding, chunking 等。
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with open(path, "r", encoding="utf-8") as f:
        user = json.load(f)

    _warn_unknown_keys(user)
    # 先对用户原始配置做一次字段归一，避免默认值掩盖来源。
    user = _normalize_runtime_config(user)
    merged = _deep_merge(DEFAULTS, user)
    if not str(merged.get("storage_path") or "").strip():
        merged["storage_path"] = "./library"
    merged = _normalize_runtime_config(merged)
    _validate_config(merged)
    return merged
