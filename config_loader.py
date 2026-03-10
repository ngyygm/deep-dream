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
    "subgraph_max_count": 100,
    "subgraph_ttl_seconds": 3600,
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
    
    return _deep_merge(DEFAULTS, user)
