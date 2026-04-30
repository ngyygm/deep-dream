#!/usr/bin/env python3
"""
DeepDream 查询工具

使用示例：
    python query_tool.py --storage_path ./data --query "林嘿嘿是谁？"
    python query_tool.py --storage_path ./data --query "白狐和林嘿嘿的关系" --mode relation
    python query_tool.py --storage_path ./data --entity "林嘿嘿" --time "2025-01-30 12:00"
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime

# 添加项目路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "DeepDream"))

from core.storage.neo4j import Neo4jStorageManager
from core.models import Entity, Relation


class DeepDreamQueryTool:
    """DeepDream 查询工具"""

    def __init__(self, storage_path: str):
        self.storage = Neo4jStorageManager(
            storage_path,
            neo4j_uri=os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
            neo4j_auth=(
                os.environ.get("NEO4J_USER", "neo4j"),
                os.environ.get("NEO4J_PASSWORD", "password"),
            ),
        )

    # ==================== Find 阶段 ====================

    def find_entities(self, query_name: str, query_content: str = None,
                      threshold: float = 0.7, max_results: int = 10) -> list[Entity]:
        """查找相关实体"""
        print(f"\n🔍 Find阶段: 搜索实体 '{query_name}'")

        entities = self.storage.search_entities_by_similarity(
            query_name=query_name,
            query_content=query_content,
            threshold=threshold,
            max_results=max_results,
            text_mode="name_and_content",
            similarity_method="embedding"
        )

        print(f"   找到 {len(entities)} 个候选实体:")
        for i, e in enumerate(entities, 1):
            print(f"   [{i}] {e.name} (ID: {e.family_id})")
            print(f"       描述: {e.content[:80]}...")
            print(f"       时间: {e.event_time}")

        return entities

    def find_relations(self, family_id: str, time_point: datetime = None) -> list[Relation]:
        """查找实体的关系"""
        print(f"\n🔗 Find阶段: 查询实体 {family_id} 的关系")

        relations = self.storage.get_entity_relations_by_family_id(
            family_id=family_id,
            time_point=time_point,
            limit=50
        )

        print(f"   找到 {len(relations)} 条关系:")
        # Batch fetch all endpoint entities
        all_abs_ids = list(set(
            [r.entity1_absolute_id for r in relations] +
            [r.entity2_absolute_id for r in relations]
        ))
        _ent_map = {}
        if all_abs_ids:
            for e in self.storage.get_entities_by_absolute_ids(all_abs_ids):
                _ent_map[e.absolute_id] = e
        for i, rel in enumerate(relations, 1):
            e1 = _ent_map.get(rel.entity1_absolute_id)
            e2 = _ent_map.get(rel.entity2_absolute_id)
            if e1 and e2:
                print(f"   [{i}] {e1.name} ↔ {e2.name}")
                print(f"       描述: {rel.content}")
                print(f"       时间: {rel.event_time}")

        return relations

    def find_entity_at_time(self, family_id: str, time_point: datetime) -> Entity:
        """查找指定时间点的实体版本"""
        print(f"\n⏰ Find阶段: 查询 {family_id} 在 {time_point} 的版本")

        entity = self.storage.get_entity_version_at_time(family_id, time_point)

        if entity:
            print(f"   ✓ 找到版本:")
            print(f"     名称: {entity.name}")
            print(f"     描述: {entity.content}")
            print(f"     时间: {entity.event_time}")
        else:
            print(f"   ✗ 未找到该时间点的版本")

        return entity

    # ==================== Select 阶段 ====================

    def select_answer(self, entities: list[Entity], relations: list[Relation] = None,
                      query: str = None) -> dict:
        """选择能回答问题的信息"""
        print(f"\n✅ Select阶段: 筛选答案")

        results = {
            "entities": [],
            "relations": [],
            "answer": ""
        }

        # 收集实体信息
        for entity in entities:
            info = {
                "name": entity.name,
                "content": entity.content,
                "time": entity.event_time,
                "family_id": entity.family_id
            }
            results["entities"].append(info)

        # 收集关系信息
        if relations:
            for rel in relations:
                info = {
                    "content": rel.content,
                    "time": rel.event_time
                }
                results["relations"].append(info)

        # 生成答案（简单实现，实际应该用LLM生成）
        if results["entities"]:
            results["answer"] = self._generate_answer(results)

        return results

    def _generate_answer(self, results: dict) -> str:
        """生成答案"""
        answer_parts = []

        if results["entities"]:
            answer_parts.append("### 实体信息")
            for e in results["entities"]:
                answer_parts.append(f"- **{e['name']}**: {e['content']}")

        if results["relations"]:
            answer_parts.append("\n### 关系信息")
            for r in results["relations"]:
                answer_parts.append(f"- {r['content']}")

        return "\n".join(answer_parts)

    # ==================== 完整查询流程 ====================

    def query_entity(self, query_name: str, query_content: str = None,
                     threshold: float = 0.7) -> dict:
        """查询实体信息"""
        # Find: 搜索实体
        entities = self.find_entities(query_name, query_content, threshold)

        if not entities:
            return {"answer": "未找到相关实体"}

        # Select: 选择最相关的实体
        results = self.select_answer(entities[:3])  # 取前3个

        return results

    def query_relation(self, entity1_name: str, entity2_name: str) -> dict:
        """查询两个实体之间的关系"""
        # Find: 查找两个实体
        entities1 = self.find_entities(entity1_name, threshold=0.6)
        entities2 = self.find_entities(entity2_name, threshold=0.6)

        if not entities1 or not entities2:
            return {"answer": "无法找到其中一个或两个实体"}

        # 使用第一个匹配的实体
        e1_id = entities1[0].family_id
        e2_id = entities2[0].family_id

        # Find: 查询关系
        print(f"\n🔗 Find阶段: 查询 {entities1[0].name} 和 {entities2[0].name} 的关系")

        relations = self.storage.get_relations_by_entities(e1_id, e2_id)

        # Select: 组织答案
        if relations:
            print(f"\n✅ Select阶段: 找到 {len(relations)} 条关系")
            results = {
                "entities": [entities1[0], entities2[0]],
                "relations": [{"content": r.content, "time": r.event_time} for r in relations],
                "answer": f"找到 {len(relations)} 条关系:\n" + "\n".join([r.content for r in relations])
            }
        else:
            print(f"\n✅ Select阶段: 未找到直接关系")
            results = {
                "answer": f"{entities1[0].name} 和 {entities2[0].name} 之间没有直接关系",
                "entities": [entities1[0], entities2[0]]
            }

        return results

    def query_history(self, entity_name: str, time_point: datetime) -> dict:
        """查询实体在指定时间点的状态"""
        # Find: 查找实体
        entities = self.find_entities(entity_name, threshold=0.7)

        if not entities:
            return {"answer": "未找到相关实体"}

        family_id = entities[0].family_id

        # Find: 查询历史版本
        entity = self.find_entity_at_time(family_id, time_point)

        # Select: 如果没有找到，返回最新版本并说明
        if not entity:
            entity = entities[0]
            print(f"\n✅ Select阶段: 该时间点不存在，返回最新版本")

        return self.select_answer([entity])

    def query_graph(self, entity_name: str, depth: int = 2) -> dict:
        """查询实体的关系图"""
        # Find: 查找中心实体
        entities = self.find_entities(entity_name, threshold=0.7)

        if not entities:
            return {"answer": "未找到相关实体"}

        center_entity = entities[0]
        center_id = center_entity.family_id

        print(f"\n🕸️  构建记忆图 (深度={depth})")

        # BFS构建关系图
        visited = {center_id}
        level_entities = {center_id: center_entity}
        level_relations = []

        current_level = [center_id]
        for d in range(depth):
            next_level = []
            # Batch-fetch relations for all entities in current level
            level_rels_by_eid = {}
            _batch_fn = getattr(self.storage, 'get_relations_by_family_ids', None)
            if _batch_fn and len(current_level) > 1:
                _latest_aids = self.storage.get_latest_absolute_ids_by_family_ids(current_level) if hasattr(self.storage, 'get_latest_absolute_ids_by_family_ids') else {}
                _all_rels = _batch_fn(current_level, limit=500)
                _aid_to_fid = {_aid: _fid for _fid, _aid in _latest_aids.items()}
                level_rels_by_eid = {eid: [] for eid in current_level}
                for rel in _all_rels:
                    f1 = _aid_to_fid.get(rel.entity1_absolute_id)
                    f2 = _aid_to_fid.get(rel.entity2_absolute_id)
                    if f1 in level_rels_by_eid:
                        level_rels_by_eid[f1].append(rel)
                    if f2 in level_rels_by_eid and f2 != f1:
                        level_rels_by_eid[f2].append(rel)
            else:
                for e_id in current_level:
                    relations = self.storage.get_entity_relations_by_family_id(e_id)
                    level_rels_by_eid[e_id] = relations
            # Batch fetch all endpoint entities for this level's relations
            all_abs_ids = list(set(
                aid
                for rels in level_rels_by_eid.values()
                for r in rels
                for aid in [r.entity1_absolute_id, r.entity2_absolute_id]
            ))
            _ent_map = {}
            if all_abs_ids:
                for e in self.storage.get_entities_by_absolute_ids(all_abs_ids):
                    _ent_map[e.absolute_id] = e
            for e_id, relations in level_rels_by_eid.items():
                for rel in relations:
                    e1 = _ent_map.get(rel.entity1_absolute_id)
                    e2 = _ent_map.get(rel.entity2_absolute_id)
                    if e1 and e2:
                        other_id = e2.family_id if e1.family_id == e_id else e1.family_id
                        level_relations.append({
                            "from": e1.family_id,
                            "to": e2.family_id,
                            "content": rel.content
                        })
                        if other_id not in visited:
                            visited.add(other_id)
                            next_level.append(other_id)
                            level_entities[other_id] = e2 if other_id == e2.family_id else e1
            current_level = next_level

        # 生成可视化
        print(f"\n✅ 记忆图构建完成:")
        print(f"   实体数量: {len(level_entities)}")
        print(f"   关系数量: {len(level_relations)}")

        return {
            "center": center_entity,
            "entities": level_entities,
            "relations": level_relations,
            "answer": self._visualize_graph(center_entity, level_entities, level_relations)
        }

    def _visualize_graph(self, center, entities, relations) -> str:
        """可视化关系图"""
        lines = [f"# 以 {center.name} 为中心的记忆图\n"]

        lines.append("## 实体\n")
        for eid, e in entities.items():
            if eid == center.family_id:
                lines.append(f"- **{e.name}** (中心): {e.content}")
            else:
                lines.append(f"- {e.name}: {e.content[:50]}...")

        lines.append("\n## 关系\n")
        for rel in relations:
            e1_name = entities[rel["from"]].name if rel["from"] in entities else rel["from"]
            e2_name = entities[rel["to"]].name if rel["to"] in entities else rel["to"]
            lines.append(f"- {e1_name} ↔ {e2_name}: {rel['content']}")

        return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="DeepDream 查询工具")
    parser.add_argument("--storage_path", "-s", required=True, help="存储路径")
    parser.add_argument("--query", "-q", help="查询问题")
    parser.add_argument("--entity", "-e", help="实体名称")
    parser.add_argument("--mode", "-m", choices=["entity", "relation", "history", "graph"],
                        default="entity", help="查询模式")
    parser.add_argument("--entity2", help="第二个实体名称（用于relation模式）")
    parser.add_argument("--time", help="时间点（用于history模式），格式: YYYY-MM-DD HH:MM:SS")
    parser.add_argument("--threshold", "-t", type=float, default=0.7, help="相似度阈值")
    parser.add_argument("--depth", "-d", type=int, default=2, help="图查询深度")

    args = parser.parse_args()

    # 初始化工具
    tool = DeepDreamQueryTool(args.storage_path)

    # 根据模式执行查询
    if args.mode == "entity":
        if args.query:
            # 从问题中提取实体名（简单实现）
            result = tool.query_entity(args.query, threshold=args.threshold)
        elif args.entity:
            result = tool.query_entity(args.entity, threshold=args.threshold)
        else:
            print("错误: entity模式需要 --query 或 --entity 参数")
            return

    elif args.mode == "relation":
        if not args.entity or not args.entity2:
            print("错误: relation模式需要 --entity 和 --entity2 参数")
            return
        result = tool.query_relation(args.entity, args.entity2)

    elif args.mode == "history":
        if not args.entity:
            print("错误: history模式需要 --entity 参数")
            return
        if not args.time:
            time_point = datetime.now()
        else:
            time_point = datetime.strptime(args.time, "%Y-%m-%d %H:%M:%S")
        result = tool.query_history(args.entity, time_point)

    elif args.mode == "graph":
        if not args.entity:
            print("错误: graph模式需要 --entity 参数")
            return
        result = tool.query_graph(args.entity, depth=args.depth)

    # 输出结果
    print("\n" + "="*60)
    print("📋 查询结果")
    print("="*60)
    print(result.get("answer", "无结果"))
    print("="*60)


if __name__ == "__main__":
    main()
