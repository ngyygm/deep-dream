"""
统一服务 API 配置加载：从 JSON 文件读取并返回配置字典，缺失项使用默认值。
"""
import json
from pathlib import Path
from typing import Any, Dict, Tuple


DEFAULTS = {
    "host": "0.0.0.0",
    "port": 5001,
    "storage_path": "./graph/tmg_storage",
    "llm": {
        "api_key": None,
        "model": "gpt-4",
        "base_url": None,
        "think": False,
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
            "llm_call_workers": 1,
            "max_total_workers": None,
        },
        "retry": {
            "queue_max_retries": 2,
            "queue_retry_delay_seconds": 2,
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
    统一新旧并发/重试配置命名，最终同时产出：
    1) 新命名（runtime.concurrency / runtime.retry）
    2) 旧命名（remember_workers / remember_max_retries / ... / pipeline.*）
    """
    cfg = dict(config)
    runtime = dict(cfg.get("runtime") or {})
    conc = dict(runtime.get("concurrency") or {})
    retry = dict(runtime.get("retry") or {})
    pipeline = dict(cfg.get("pipeline") or {})

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
    llm_call_workers = _pick(
        conc.get("llm_call_workers"),
        pipeline.get("llm_threads"),
        cfg.get("llm_threads"),
    )
    max_total_workers = _pick(
        conc.get("max_total_workers"),
        cfg.get("max_total_worker_threads"),
    )
    queue_max_retries = _pick(
        retry.get("queue_max_retries"),
        cfg.get("remember_max_retries"),
    )
    queue_retry_delay_seconds = _pick(
        retry.get("queue_retry_delay_seconds"),
        cfg.get("remember_retry_delay_seconds"),
    )

    # 回填新命名
    conc["queue_workers"] = int(queue_workers if queue_workers is not None else 1)
    conc["window_workers"] = int(window_workers if window_workers is not None else 1)
    conc["llm_call_workers"] = int(llm_call_workers if llm_call_workers is not None else 1)
    conc["max_total_workers"] = max_total_workers
    retry["queue_max_retries"] = int(queue_max_retries if queue_max_retries is not None else 2)
    retry["queue_retry_delay_seconds"] = float(
        queue_retry_delay_seconds if queue_retry_delay_seconds is not None else 2
    )
    runtime["concurrency"] = conc
    runtime["retry"] = retry
    cfg["runtime"] = runtime

    # 回填旧命名（供现有代码无缝使用）
    cfg["remember_workers"] = conc["queue_workers"]
    cfg["remember_max_retries"] = retry["queue_max_retries"]
    cfg["remember_retry_delay_seconds"] = retry["queue_retry_delay_seconds"]
    cfg["max_total_worker_threads"] = conc["max_total_workers"]

    pipeline["max_concurrent_windows"] = conc["window_workers"]
    pipeline["llm_threads"] = conc["llm_call_workers"]
    cfg["pipeline"] = pipeline
    cfg["llm_threads"] = conc["llm_call_workers"]

    return cfg


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
    
    merged = _deep_merge(DEFAULTS, user)
    return _normalize_runtime_config(merged)
