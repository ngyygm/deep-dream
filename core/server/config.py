"""
统一服务 API 配置加载：从 JSON 文件读取并返回配置字典，缺失项使用默认值。
"""
import json
import math
from pathlib import Path
from typing import Any, Dict, Tuple

from core.exceptions import ConfigError


DEFAULTS = {
    # Bind to loopback by default; exposing a local memory store on every
    # interface must be an explicit deployment decision.
    "host": "127.0.0.1",
    # 端口真相（P5）：与 README/CLAUDE.md/cmd_task._API_BASE 统一为 16200。
    # 旧默认 5001 与文档宣称的 16200 不符——不读配置启动时落在 5001，
    # CLI task 子命令却指向 16200，首个 remember 提交即连接失败。
    "port": 16200,
    "flask_threaded": True,
    "max_request_bytes": 12 * 1024 * 1024,
    "rate_limit_per_minute": 0,
    "monitor_refresh_seconds": 1.0,
    "auto_port_fallback": False,
    "log_mode": "detail",
    "storage_path": "./library",
    "auth": {
        # Loopback development stays frictionless.  Non-loopback listeners
        # are forced into strict authentication by create_app regardless of
        # these local defaults.
        "enabled": None,
        "strict_mode": None,
        "allow_dev_key": False,
        "api_keys_file": None,
    },
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
        # 端点协议显式声明："openai"（/v1/chat/completions）或 "ollama"（/api/chat）；
        # null/"auto" 时按 base_url/api_key 嗅探（旧默认行为）。自建网关等
        # 嗅探无法识别的端点应显式设为 "openai"。
        "protocol": None,
        # OpenAI 兼容端点请求 response_format={"type":"json_object"}（结构化调用
        # 更稳，少重试）；默认关，确认端点支持后开启。端点 4xx 拒绝时自动降级重试
        "json_object_mode": False,
        # 对齐阶段专用模型（双模型管线）；enabled=false 时与主模型共用
        "alignment": {},
    },
    "embedding": {
        "model": None,
        "device": "cpu",
        "trust_remote_code": False,
        # 远程 OpenAI 兼容 /v1/embeddings 端点（设置任一即走远程，不加载本地模型）
        "api_key": None,
        "api_base": None,
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
            "queue_max_size": 1000,
        },
        "integrity": {
            # P3.7：默认关闭——GET /documents 列表页逐文档全量评估代价过高，
            # 完整性改为按需拉取（GET /api/v1/documents/<id>/integrity）。
            # 显式设为 true 可恢复旧的列表页自动评估行为。
            "auto_check_documents": False,
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
            # 窗口级批量对齐（主 run 实证等效 B1 档，默认显式开启）
            "window_batch_alignment": True,
            # ALIGN-V2 簇收敛对齐引擎：窗口等价组收集 + step9 跨窗口并行 +
            # scope 末全库收敛合并（详见 core/remember/align_v2.py）。
            # 2026-08-29 全量对比后转默认引擎：calls/doc -65%、tok/doc -20%、
            # 三轨全升、重复家族率 0.8%→0.2%（research/reports/lme_v1_vs_v2_full_2026-08-29.md）
            "cluster_convergence": True,
            # strong-v1 大窗口覆盖；null = 用 chunking.window_size
            "window_size_chars": None,
            "overlap_chars": None,
            # 置信度 0.3 共现兜底关系（P2 默认关）
            "fallback_cooccurrence_relations": False,
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
            # A model name (for example ``Qwen/...``) is a local/HuggingFace
            # model request unless the operator explicitly disables local
            # loading.  Returning False here silently degraded the example
            # configuration to text-only search.
            return None, model, embedding.get("use_local") is not False
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
    try:
        cap = None if max_llm_conc is None else int(max_llm_conc)
    except (TypeError, ValueError):
        # Preserve the malformed value so _validate_config can raise the
        # documented ConfigError instead of leaking a raw ValueError here.
        cap = None

    def _resolve_worker_count(value, default: int, *, auto_default: int) -> Any:
        if value is None:
            return int(default)
        if isinstance(value, str) and value.strip().lower() in {"auto", "default", ""}:
            return int(auto_default)
        # Preserve malformed scalar types for the schema validator.  Calling
        # int(1.5) here would silently turn an invalid worker count into 1;
        # bool is likewise not a valid integer configuration even though it is
        # an int subclass in Python.
        if isinstance(value, bool):
            return value
        if isinstance(value, float) and not value.is_integer():
            return value
        try:
            return int(value)
        except (TypeError, ValueError):
            return value

    # 回填新结构。用户只需要设置 llm.max_concurrency；window_workers 默认自动跟随，
    # 用于让多个 episode 形成流水线，但最终 LLM 请求仍由全局闸门限制。
    _auto_windows = max(1, min(cap or 1, 3))
    conc["queue_workers"] = _resolve_worker_count(conc.get("queue_workers"), 1, auto_default=1)
    conc["window_workers"] = _resolve_worker_count(
        conc.get("window_workers"), _auto_windows, auto_default=_auto_windows)
    if isinstance(cap, int) and cap >= 1:
        if isinstance(conc.get("queue_workers"), int):
            conc["queue_workers"] = min(conc["queue_workers"], cap)
        if isinstance(conc.get("window_workers"), int):
            conc["window_workers"] = min(conc["window_workers"], cap)
    _retries = retry.get("queue_max_retries")
    try:
        if isinstance(_retries, bool) or (
            isinstance(_retries, float) and not _retries.is_integer()
        ):
            retry["queue_max_retries"] = _retries
        else:
            retry["queue_max_retries"] = int(_retries if _retries is not None else 2)
    except (TypeError, ValueError):
        retry["queue_max_retries"] = _retries
    _retry_delay = retry.get("queue_retry_delay_seconds")
    try:
        retry["queue_retry_delay_seconds"] = float(_retry_delay if _retry_delay is not None else 2)
    except (TypeError, ValueError):
        retry["queue_retry_delay_seconds"] = _retry_delay
    _cache_value = task.get("load_cache_memory", False)
    if isinstance(_cache_value, str):
        _cache_lower = _cache_value.strip().lower()
        if _cache_lower in {"true", "1", "yes", "on"}:
            _cache_value = True
        elif _cache_lower in {"false", "0", "no", "off", ""}:
            _cache_value = False
    task["load_cache_memory"] = _cache_value
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

    def _number(value: Any, name: str, *, integer: bool = False,
                minimum: float | None = None, maximum: float | None = None):
        """Validate a scalar without leaking raw TypeError/ValueError."""
        if isinstance(value, bool):
            errors.append(f"{name} 应为{'整数' if integer else '数字'}，当前值: {value}")
            return None
        try:
            if integer:
                if isinstance(value, float) and not value.is_integer():
                    raise ValueError
                parsed = int(value)
            else:
                parsed = float(value)
        except (TypeError, ValueError):
            errors.append(f"{name} 应为{'整数' if integer else '数字'}，当前值: {value}")
            return None
        if not integer and not math.isfinite(parsed):
            errors.append(f"{name} 必须是有限数字，当前值: {value}")
            return None
        if minimum is not None and parsed < minimum:
            errors.append(f"{name} 应 >= {minimum}，当前值: {value}")
        if maximum is not None and parsed > maximum:
            errors.append(f"{name} 应 <= {maximum}，当前值: {value}")
        return parsed

    port = config.get("port")
    if port is not None:
        _number(port, "port", integer=True, minimum=1, maximum=65535)
    for bool_name in ("flask_threaded", "auto_port_fallback"):
        if bool_name in config and not isinstance(config[bool_name], bool):
            errors.append(f"{bool_name} 必须是 true/false")
    _number(config.get("max_request_bytes", 12 * 1024 * 1024),
            "max_request_bytes", integer=True,
            minimum=1_048_576, maximum=256 * 1024 * 1024)
    _number(config.get("rate_limit_per_minute", 0),
            "rate_limit_per_minute", integer=True, minimum=0, maximum=100_000)
    _number(config.get("monitor_refresh_seconds", 1.0),
            "monitor_refresh_seconds", minimum=0.05, maximum=3600)
    if "host" in config and not isinstance(config["host"], str):
        errors.append("host 必须是字符串")

    auth = config.get("auth")
    if auth is not None:
        if not isinstance(auth, dict):
            errors.append("auth 必须是 JSON object")
        else:
            for auth_bool in ("enabled", "strict_mode", "allow_dev_key"):
                if auth_bool in auth and auth[auth_bool] is not None and not isinstance(auth[auth_bool], bool):
                    errors.append(f"auth.{auth_bool} 必须是 true/false")
            if auth.get("api_keys_file") is not None and not isinstance(auth.get("api_keys_file"), str):
                errors.append("auth.api_keys_file 必须是字符串")

    embedding = config.get("embedding") or {}
    if "trust_remote_code" in embedding and not isinstance(embedding["trust_remote_code"], bool):
        errors.append("embedding.trust_remote_code 必须是 true/false")

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
    _number(llm.get("max_concurrency", 1), "llm.max_concurrency", integer=True, minimum=1, maximum=128)
    _number(llm.get("timeout_seconds", 300), "llm.timeout_seconds", minimum=1, maximum=86400)
    _number(llm.get("connect_timeout_seconds", 30), "llm.connect_timeout_seconds", minimum=0.1, maximum=3600)
    if llm.get("max_tokens") is not None:
        _number(llm.get("max_tokens"), "llm.max_tokens", integer=True, minimum=1, maximum=2_000_000)
    _proto = llm.get("protocol")
    if _proto is not None and str(_proto).strip().lower() not in {"auto", "openai", "ollama"}:
        errors.append(f"llm.protocol 只支持 openai/ollama/auto，当前值: {_proto}")

    chunking = config.get("chunking") or {}
    ws = chunking.get("window_size", 1000)
    ol = chunking.get("overlap", 200)
    ws_i = _number(ws, "chunking.window_size", integer=True, minimum=1, maximum=1_000_000)
    ol_i = _number(ol, "chunking.overlap", integer=True, minimum=0, maximum=999_999)
    if ws_i is not None and ol_i is not None and ol_i >= ws_i:
        errors.append(f"chunking.overlap ({ol}) 必须小于 chunking.window_size ({ws})")

    pipeline = config.get("pipeline") or {}
    search = pipeline.get("search") or {}
    thresholds = [
        ("pipeline.search.similarity_threshold", search.get("similarity_threshold")),
        ("pipeline.search.jaccard_search_threshold", search.get("jaccard_search_threshold")),
        ("pipeline.search.embedding_name_search_threshold", search.get("embedding_name_search_threshold")),
        ("pipeline.search.embedding_full_search_threshold", search.get("embedding_full_search_threshold")),
    ]
    for name, val in thresholds:
        if val is not None:
            _number(val, name, minimum=0.0, maximum=1.0)

    runtime = config.get("runtime") or {}
    concurrency = runtime.get("concurrency") or {}
    retry = runtime.get("retry") or {}
    _number(concurrency.get("queue_workers", 1), "runtime.concurrency.queue_workers", integer=True, minimum=1, maximum=64)
    window_workers = concurrency.get("window_workers")
    if window_workers is not None and not (isinstance(window_workers, str) and window_workers.strip().lower() in {"auto", "default"}):
        _number(window_workers, "runtime.concurrency.window_workers", integer=True, minimum=1, maximum=64)
    _number(retry.get("queue_max_retries", 2), "runtime.retry.queue_max_retries", integer=True, minimum=0, maximum=20)
    _number(retry.get("queue_retry_delay_seconds", 2), "runtime.retry.queue_retry_delay_seconds", minimum=0, maximum=3600)
    _number((runtime.get("task") or {}).get("queue_max_size", 1000),
            "runtime.task.queue_max_size", integer=True, minimum=1, maximum=100_000)

    storage = config.get("storage") or {}
    _number(storage.get("vector_dim", 1024), "storage.vector_dim", integer=True, minimum=1, maximum=65536)

    embedding = config.get("embedding") or {}
    if embedding.get("max_concurrency") is not None:
        _number(embedding.get("max_concurrency"), "embedding.max_concurrency", integer=True, minimum=1, maximum=64)
    if embedding.get("cache_max_size") is not None:
        _number(embedding.get("cache_max_size"), "embedding.cache_max_size", integer=True, minimum=1, maximum=1_000_000)
    if embedding.get("cache_ttl") is not None:
        _number(embedding.get("cache_ttl"), "embedding.cache_ttl", minimum=1, maximum=31_536_000)

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
    """用户配置里出现 DEFAULTS 未覆盖的键时告警（P5：配置面收敛，防拼写错误静默失效）。

    `_` 前缀键视为注释（JSON 无原生注释的通行约定），不告警。
    """
    import logging
    logger = logging.getLogger(__name__)
    for key in user:
        if key not in DEFAULTS and not key.startswith("_"):
            logger.warning("配置键未收录于默认值: %r — 请对照 service_config.example.json（拼写错误将不生效）", key)
    for section, defaults in DEFAULTS.items():
        user_section = user.get(section)
        if not isinstance(user_section, dict) or not isinstance(defaults, dict):
            continue
        for key in user_section:
            if key not in defaults and not key.startswith("_"):
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

    if not isinstance(user, dict):
        raise ConfigError("配置文件根节点必须是 JSON object")

    _warn_unknown_keys(user)
    # 先对用户原始配置做一次字段归一，避免默认值掩盖来源。
    user = _normalize_runtime_config(user)
    merged = _deep_merge(DEFAULTS, user)
    if not str(merged.get("storage_path") or "").strip():
        merged["storage_path"] = "./library"
    merged = _normalize_runtime_config(merged)
    _validate_config(merged)
    return merged
