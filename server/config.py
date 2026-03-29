"""
统一服务 API 配置加载：从 JSON 文件读取并返回配置字典，缺失项使用默认值。
"""
import json
from pathlib import Path
from typing import Any, Dict, Tuple


DEFAULTS = {
    "host": "0.0.0.0",
    "port": 5001,
    "storage_path": "./graph",
    "llm": {
        "api_key": None,
        "model": "gpt-4",
        "base_url": None,
        "think": False,
        # 请求输入 prompt 的本地预检上限；与 service_config.json 中 llm.context_window_tokens 对应
        "context_window_tokens": 8000,
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
            "window_workers": 1,
        },
        "retry": {
            "queue_max_retries": 2,
            "queue_retry_delay_seconds": 2,
        },
        "task": {
            "load_cache_memory": False,
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
            "extraction_rounds": 1,
            "entity_extraction_rounds": 1,
            "relation_extraction_rounds": 1,
            "entity_post_enhancement": False,
            "prompt_memory_cache_max_chars": 2000,
            "compress_multi_round_extraction": False,
        },
        "debug": {
            "distill_data_dir": "distill_pipeline",
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
    if model is not None and (isinstance(model, str) and model.strip()):
        model = model.strip()
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
    统一新旧配置结构，最终同时产出：
    1) 新结构：runtime.concurrency / runtime.retry / runtime.task / pipeline.search / ...
    2) 旧兼容字段：remember_workers / remember_max_retries / pipeline.* 扁平字段
    """
    cfg = dict(config)
    runtime = dict(cfg.get("runtime") or {})
    conc = dict(runtime.get("concurrency") or {})
    retry = dict(runtime.get("retry") or {})
    task = dict(runtime.get("task") or {})
    pipeline = dict(cfg.get("pipeline") or {})
    search = dict(pipeline.get("search") or {})
    alignment = dict(pipeline.get("alignment") or {})
    extraction = dict(pipeline.get("extraction") or {})
    debug = dict(pipeline.get("debug") or {})

    def _pick(*values):
        for v in values:
            if v is not None:
                return v
        return None

    queue_workers = _pick(
        conc.get("queue_workers"),
        cfg.get("remember_workers"),
    )
    window_workers = _pick(
        conc.get("window_workers"),
        pipeline.get("max_concurrent_windows"),
        cfg.get("max_concurrent_windows"),
    )
    queue_max_retries = _pick(
        retry.get("queue_max_retries"),
        cfg.get("remember_max_retries"),
    )
    queue_retry_delay_seconds = _pick(
        retry.get("queue_retry_delay_seconds"),
        cfg.get("remember_retry_delay_seconds"),
    )
    load_cache_memory = _pick(
        task.get("load_cache_memory"),
        pipeline.get("load_cache_memory"),
    )

    search_keys = (
        "similarity_threshold",
        "max_similar_entities",
        "content_snippet_length",
        "relation_content_snippet_length",
        "relation_endpoint_jaccard_threshold",
        "relation_endpoint_embedding_threshold",
        "jaccard_search_threshold",
        "embedding_name_search_threshold",
        "embedding_full_search_threshold",
    )
    normalized_search = {
        key: _pick(search.get(key), pipeline.get(key))
        for key in search_keys
    }
    max_alignment_candidates = _pick(
        alignment.get("max_alignment_candidates"),
        pipeline.get("max_alignment_candidates"),
    )
    extraction_keys = (
        "extraction_rounds",
        "entity_extraction_rounds",
        "relation_extraction_rounds",
        "entity_post_enhancement",
        "prompt_memory_cache_max_chars",
        "compress_multi_round_extraction",
    )
    normalized_extraction = {
        key: _pick(extraction.get(key), pipeline.get(key))
        for key in extraction_keys
    }
    distill_data_dir = _pick(
        debug.get("distill_data_dir"),
        pipeline.get("distill_data_dir"),
    )

    # 回填新结构
    conc["queue_workers"] = int(queue_workers if queue_workers is not None else 1)
    conc["window_workers"] = int(window_workers if window_workers is not None else 1)
    # 队列/窗口线程不超过 LLM 并发：max_concurrency=1 时整体按串行语义运行
    llm = cfg.get("llm") or {}
    max_llm_conc = llm.get("max_concurrency")
    if max_llm_conc is not None:
        cap = int(max_llm_conc)
        if cap >= 1:
            conc["queue_workers"] = min(conc["queue_workers"], cap)
            conc["window_workers"] = min(conc["window_workers"], cap)
    retry["queue_max_retries"] = int(queue_max_retries if queue_max_retries is not None else 2)
    retry["queue_retry_delay_seconds"] = float(
        queue_retry_delay_seconds if queue_retry_delay_seconds is not None else 2
    )
    task["load_cache_memory"] = bool(load_cache_memory) if load_cache_memory is not None else False
    runtime["concurrency"] = conc
    runtime["retry"] = retry
    runtime["task"] = task
    cfg["runtime"] = runtime

    search.update({k: v for k, v in normalized_search.items() if v is not None})
    alignment["max_alignment_candidates"] = max_alignment_candidates
    extraction.update({k: v for k, v in normalized_extraction.items() if v is not None})
    debug["distill_data_dir"] = distill_data_dir if distill_data_dir is not None else "distill_pipeline"
    pipeline["search"] = search
    pipeline["alignment"] = alignment
    pipeline["extraction"] = extraction
    pipeline["debug"] = debug

    # 回填旧命名（供现有代码无缝使用）
    cfg["remember_workers"] = conc["queue_workers"]
    cfg["remember_max_retries"] = retry["queue_max_retries"]
    cfg["remember_retry_delay_seconds"] = retry["queue_retry_delay_seconds"]

    pipeline["max_concurrent_windows"] = conc["window_workers"]
    pipeline["load_cache_memory"] = task["load_cache_memory"]
    for key, value in normalized_search.items():
        pipeline[key] = value
    pipeline["max_alignment_candidates"] = max_alignment_candidates
    for key, value in normalized_extraction.items():
        pipeline[key] = value
    pipeline["distill_data_dir"] = debug["distill_data_dir"]
    cfg["pipeline"] = pipeline

    return cfg


def merge_llm_alignment(llm: Dict[str, Any]) -> Dict[str, Any]:
    """
    合并步骤 6/7（对齐）专用 LLM 配置。
    - llm.alignment.enabled == false：关闭对齐专用通道，步骤 6/7 与 1–5 共用同一模型与并发策略。
    - enabled == true 或未写 enabled 但存在 base_url/api_key/model 等：启用对齐配置。
    - llm.alignment.max_concurrency：对齐阶段（步骤 6–7）独立 LLM 并发上限，与 llm.max_concurrency（步骤 1–5）解耦。
    """
    if llm.get("alignment_enabled") is False:
        return {}

    nested = llm.get("alignment")
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
    )
    for nk, fk in mapping:
        val = pick(nk, fk)
        if val is not None:
            out["think_mode" if nk == "think" else nk] = val

    mc_align = pick("max_concurrency", "alignment_max_concurrency")
    if mc_align is not None:
        out["max_concurrency"] = int(mc_align)

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
    if not llm.get("api_key") and not llm.get("base_url"):
        errors.append("llm.api_key 或 llm.base_url 至少需要配置一个")
    _cwt = llm.get("context_window_tokens")
    if _cwt is not None:
        try:
            _cwt_i = int(_cwt)
            if _cwt_i < 256:
                errors.append(f"llm.context_window_tokens 应 >= 256，当前值: {_cwt}")
        except (TypeError, ValueError):
            errors.append(f"llm.context_window_tokens 应为整数，当前值: {_cwt}")

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
    _pmcmc = extraction.get("prompt_memory_cache_max_chars", pipeline.get("prompt_memory_cache_max_chars"))
    if _pmcmc is not None:
        try:
            _pmcmc_i = int(_pmcmc)
            if _pmcmc_i < 0:
                errors.append(
                    f"pipeline.extraction.prompt_memory_cache_max_chars 应 >= 0，当前值: {_pmcmc}"
                )
        except (TypeError, ValueError):
            errors.append(
                "pipeline.extraction.prompt_memory_cache_max_chars 应为整数"
                f"，当前值: {_pmcmc}"
            )

    if errors:
        from processor.exceptions import ConfigError
        raise ConfigError("配置校验失败:\n  " + "\n  ".join(errors))


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

    # 先对用户原始配置做一次新旧字段归一，避免默认值掩盖旧字段来源。
    user = _normalize_runtime_config(user)
    merged = _deep_merge(DEFAULTS, user)
    merged = _normalize_runtime_config(merged)
    _validate_config(merged)
    return merged
