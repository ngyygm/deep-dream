"""Dream Agent 工具定义与执行。"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from .models import DreamActionResult

logger = logging.getLogger(__name__)


# ============================================================
# 工具定义 schema（供 LLM 参考）
# ============================================================

TOOL_SCHEMAS = {
    "get_seeds": {
        "description": "按策略获取种子实体，作为梦境探索的起点",
        "arguments": {
            "strategy": {"type": "string", "required": True,
                         "enum": ["random", "orphan", "hub", "time_gap", "cross_community", "low_confidence"],
                         "description": "种子选取策略"},
            "count": {"type": "integer", "required": False, "default": 5,
                      "description": "种子数量 (1-20)"},
        },
    },
    "get_entity": {
        "description": "获取指定实体的详细信息及其直接关系",
        "arguments": {
            "entity_id": {"type": "string", "required": True,
                          "description": "实体的 entity_id 或 UUID"},
        },
    },
    "traverse": {
        "description": "从指定实体出发进行 BFS 扩展，获取多跳邻居",
        "arguments": {
            "entity_id": {"type": "string", "required": True,
                          "description": "起始实体的 entity_id"},
            "depth": {"type": "integer", "required": False, "default": 2,
                      "description": "BFS 扩展深度 (1-3)"},
            "max_neighbors": {"type": "integer", "required": False, "default": 20,
                              "description": "每层最大邻居数"},
        },
    },
    "search_similar": {
        "description": "语义相似度搜索，查找与指定实体语义相近的其他实体",
        "arguments": {
            "entity_id": {"type": "string", "required": True,
                          "description": "参考实体的 entity_id"},
            "top_k": {"type": "integer", "required": False, "default": 10,
                      "description": "返回数量"},
            "threshold": {"type": "float", "required": False, "default": 0.5,
                          "description": "相似度阈值"},
        },
    },
    "search_bm25": {
        "description": "BM25 关键词搜索实体",
        "arguments": {
            "query": {"type": "string", "required": True,
                      "description": "搜索关键词"},
            "top_k": {"type": "integer", "required": False, "default": 10,
                      "description": "返回数量"},
        },
    },
    "get_community": {
        "description": "获取社区信息及其成员实体",
        "arguments": {
            "community_id": {"type": "integer", "required": True,
                             "description": "社区 ID"},
        },
    },
    "create_relation": {
        "description": "保存一条梦境发现的新关系（仅限已有实体之间）",
        "arguments": {
            "entity1_id": {"type": "string", "required": True,
                           "description": "第一个实体的 entity_id"},
            "entity2_id": {"type": "string", "required": True,
                           "description": "第二个实体的 entity_id"},
            "content": {"type": "string", "required": True,
                        "description": "关系描述"},
            "confidence": {"type": "float", "required": True,
                           "description": "置信度 (0.0-1.0)"},
            "reasoning": {"type": "string", "required": True,
                          "description": "推理依据，说明为什么这两实体有关联"},
        },
    },
    "create_episode": {
        "description": "保存梦境 episode，记录本轮发现",
        "arguments": {
            "content": {"type": "string", "required": True,
                        "description": "梦境叙述内容"},
            "entities_examined": {"type": "array", "required": False,
                                  "description": "检查过的实体 ID 列表"},
            "relations_created": {"type": "array", "required": False,
                                  "description": "创建的关系 ID 列表"},
        },
    },
}


def format_tool_descriptions() -> str:
    """生成工具描述文本，嵌入 system prompt。"""
    lines = []
    for name, schema in TOOL_SCHEMAS.items():
        args_desc = []
        for arg_name, arg_info in schema["arguments"].items():
            req = "必填" if arg_info.get("required") else "可选"
            type_str = arg_info.get("type", "string")
            desc = arg_info.get("description", "")
            args_desc.append(f"    - {arg_name} ({type_str}, {req}): {desc}")
        args_text = "\n".join(args_desc) if args_desc else "    (无参数)"
        lines.append(f"### {name}\n{schema['description']}\n参数:\n{args_text}")
    return "\n\n".join(lines)


# ============================================================
# 工具执行器
# ============================================================

class DreamToolExecutor:
    """执行 Dream Agent 的工具调用。"""

    def __init__(self, storage: Any, session_id: str, config: Any = None):
        self.storage = storage
        self.session_id = session_id
        self.config = config
        self._saved_relations: List[Dict[str, Any]] = []
        self._saved_episode_ids: List[str] = []

    @property
    def saved_relations(self) -> List[Dict[str, Any]]:
        return self._saved_relations

    @property
    def saved_episode_ids(self) -> List[str]:
        return self._saved_episode_ids

    def execute(self, tool_name: str, arguments: Dict[str, Any]) -> DreamActionResult:
        """执行单个工具调用。"""
        handler = getattr(self, f"_exec_{tool_name}", None)
        if handler is None:
            return DreamActionResult(success=False, error=f"未知工具: {tool_name}")
        try:
            result = handler(**arguments)
            return DreamActionResult(success=True, data=result)
        except ValueError as e:
            return DreamActionResult(success=False, error=str(e))
        except Exception as e:
            logger.warning("Dream tool %s 执行失败: %s", tool_name, e)
            return DreamActionResult(success=False, error=str(e))

    # ---- 工具实现 ----

    def _exec_get_seeds(self, *, strategy: str = "random", count: int = 5, **_) -> Any:
        count = max(1, min(int(count), 20))
        if not hasattr(self.storage, 'get_dream_seeds'):
            raise ValueError("存储层不支持 dream seeds")
        return self.storage.get_dream_seeds(strategy=strategy, count=count)

    def _exec_get_entity(self, *, entity_id: str, **_) -> Dict[str, Any]:
        resolved = self.storage.resolve_entity_id(entity_id)
        if not resolved:
            raise ValueError(f"实体不存在: {entity_id}")

        entity = self.storage.get_entity_by_entity_id(resolved)
        if not entity:
            raise ValueError(f"实体不存在: {entity_id}")

        # 获取直接关系
        relations = []
        if hasattr(self.storage, 'get_relations_by_entity_ids'):
            try:
                relations = self.storage.get_relations_by_entity_ids([resolved])
            except Exception:
                pass

        return {
            "entity": {
                "entity_id": entity.entity_id,
                "name": entity.name,
                "content": entity.content[:500],
                "confidence": entity.confidence,
                "event_time": str(entity.event_time) if entity.event_time else None,
            },
            "relations": [
                {"relation_id": r.relation_id, "content": r.content[:200]}
                for r in relations[:10]
            ],
        }

    def _exec_traverse(self, *, entity_id: str, depth: int = 2, max_neighbors: int = 20, **_) -> Dict[str, Any]:
        depth = max(1, min(int(depth), 3))
        max_neighbors = max(1, min(int(max_neighbors), 50))

        resolved = self.storage.resolve_entity_id(entity_id)
        if not resolved:
            raise ValueError(f"实体不存在: {entity_id}")

        visited = {resolved}
        current_layer = [resolved]
        all_entities = []
        all_relations = []

        for d in range(depth):
            next_layer = []
            if not current_layer:
                break
            if hasattr(self.storage, 'get_relations_by_entity_ids'):
                try:
                    relations = self.storage.get_relations_by_entity_ids(current_layer)
                except Exception:
                    break

                for r in relations[:max_neighbors]:
                    all_relations.append({
                        "relation_id": r.relation_id,
                        "content": r.content[:200],
                        "entity1_absolute_id": r.entity1_absolute_id,
                        "entity2_absolute_id": r.entity2_absolute_id,
                    })
                    # 收集邻居实体
                    if r.entity1_absolute_id not in visited:
                        next_layer.append(r.entity1_absolute_id)
                    if r.entity2_absolute_id not in visited:
                        next_layer.append(r.entity2_absolute_id)

            visited.update(next_layer)
            current_layer = next_layer[:max_neighbors]

        # 批量获取发现的实体信息
        for eid in list(visited)[:50]:
            try:
                entity = self.storage.get_entity_by_entity_id(eid)
                if entity:
                    all_entities.append({
                        "entity_id": entity.entity_id,
                        "name": entity.name,
                        "content": entity.content[:300],
                    })
            except Exception:
                pass

        return {
            "depth_reached": depth,
            "entities_found": len(all_entities),
            "relations_found": len(all_relations),
            "entities": all_entities[:30],
            "relations": all_relations[:30],
        }

    def _exec_search_similar(self, *, entity_id: str, top_k: int = 10, threshold: float = 0.5, **_) -> Any:
        top_k = max(1, min(int(top_k), 30))
        threshold = max(0.0, min(float(threshold), 1.0))

        resolved = self.storage.resolve_entity_id(entity_id)
        if not resolved:
            raise ValueError(f"实体不存在: {entity_id}")

        entity = self.storage.get_entity_by_entity_id(resolved)
        if not entity or not entity.embedding:
            raise ValueError(f"实体无 embedding: {entity_id}")

        if hasattr(self.storage, 'search_entities_by_similarity'):
            results = self.storage.search_entities_by_similarity(
                entity.embedding, top_k=top_k, threshold=threshold,
            )
            return {"results": results}

        raise ValueError("存储层不支持相似度搜索")

    def _exec_search_bm25(self, *, query: str, top_k: int = 10, **_) -> Any:
        top_k = max(1, min(int(top_k), 30))
        if hasattr(self.storage, 'search_entities_by_bm25'):
            results = self.storage.search_entities_by_bm25(query, top_k=top_k)
            return {"results": results}
        raise ValueError("存储层不支持 BM25 搜索")

    def _exec_get_community(self, *, community_id: int, **_) -> Any:
        if hasattr(self.storage, 'get_communities'):
            communities, _ = self.storage.get_communities(limit=1000)
            for c in communities:
                if c.get("community_id") == community_id:
                    return {
                        "community_id": community_id,
                        "size": len(c.get("members", [])),
                        "members": [
                            {"entity_id": m.get("entity_id"), "name": m.get("name")}
                            for m in c.get("members", [])[:30]
                        ],
                    }
            raise ValueError(f"社区不存在: {community_id}")
        raise ValueError("存储层不支持社区查询")

    def _exec_create_relation(self, *, entity1_id: str, entity2_id: str,
                              content: str, confidence: float, reasoning: str, **_) -> Dict[str, Any]:
        confidence = max(0.0, min(float(confidence), 1.0))
        if entity1_id == entity2_id:
            raise ValueError("不能创建自环关系")
        if not content.strip():
            raise ValueError("关系描述不能为空")
        if not reasoning.strip():
            raise ValueError("推理依据不能为空")

        if not hasattr(self.storage, 'save_dream_relation'):
            raise ValueError("存储层不支持保存梦境关系")

        result = self.storage.save_dream_relation(
            entity1_id=entity1_id,
            entity2_id=entity2_id,
            content=content.strip(),
            confidence=confidence,
            reasoning=reasoning.strip(),
            dream_cycle_id=self.session_id,
        )
        self._saved_relations.append(result)
        return result

    def _exec_create_episode(self, *, content: str,
                             entities_examined: Optional[List[str]] = None,
                             relations_created: Optional[List[Dict]] = None, **_) -> Dict[str, Any]:
        if not content.strip():
            raise ValueError("episode 内容不能为空")

        if not hasattr(self.storage, 'save_dream_episode'):
            raise ValueError("存储层不支持保存梦境 episode")

        result = self.storage.save_dream_episode(
            content=content.strip(),
            entities_examined=entities_examined or [],
            relations_created=relations_created or [],
            strategy_used="",
            dream_cycle_id=self.session_id,
        )
        self._saved_episode_ids.append(result.get("episode_id", ""))
        return result
