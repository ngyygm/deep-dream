#!/usr/bin/env python3
"""
通用实体关系查询脚本（增强版）
支持智能实体名解析和别名映射

功能：
1. 支持模糊搜索实体（降低阈值、多种搜索模式）
2. 智能实体名解析：支持别名、称谓、昵称映射
3. 查询两个实体之间的直接关系
4. 查询每个实体的所有关系，筛选包含对方名称的关系
5. 加载相关的记忆缓存内容
6. 提供时间线视图

使用方法：
    python query_entity_relation.py --entity1 "汪教授" --entity2 "牛顿"
    python query_entity_relation.py --entity1 "牛顿" --entity2 "冯·诺依曼" --detail
"""

import sys
import argparse
import re
from pathlib import Path
from datetime import datetime

# 添加DeepDream到路径（支持多种路径）
import os
from pathlib import Path

# 方法1: 从scripts目录向上找
script_dir = Path(__file__).parent
tmg_dir = script_dir.parent.parent.parent
if (tmg_dir / "processor").exists():
    sys.path.insert(0, str(tmg_dir))
else:
    # 方法2: 使用绝对路径
    sys.path.insert(0, str(Path("/home/linkco/exa/DeepDream")))

try:
    from core.storage.neo4j import Neo4jStorageManager
except ImportError:
    print("错误: 无法导入 Neo4jStorageManager")
    print("请确保 DeepDream 在正确的位置")
    sys.exit(1)



# 称谓-关键词映射表
TITLE_KEYWORDS = {
    '教授': ['教授', 'Prof', 'Professor'],
    '博士': ['博士', 'Dr', 'Doctor', 'PhD'],
    '先生': ['先生', 'Mr', 'Mister'],
    '女士': ['女士', 'Ms', 'Mrs', 'Miss'],
    '皇帝': ['皇帝', '陛下', 'Emperor', '帝'],
    '国王': ['国王', 'King'],
    '主席': ['主席', 'Chairman'],
    '将军': ['将军', 'General'],
    '统帅': ['统帅', 'Commander', '元帅'],
}

# 常见别名映射（可扩展）
ALIAS_MAP = {
    # 汪淼相关
    '汪教授': '汪淼',
    '汪博士': '汪淼',
    '汪先生': '汪淼',

    # 牛顿相关
    '伊萨克·牛顿': '牛顿',
    '艾萨克·牛顿': '牛顿',
    '牛顿爵士': '牛顿',

    # 冯·诺伊曼相关
    '冯·诺依曼': '冯·诺伊曼',
    '约翰·冯·诺伊曼': '冯·诺伊曼',
    '冯诺伊曼': '冯·诺伊曼',

    # 秦始皇相关
    '嬴政': '秦始皇',
    '秦王': '秦始皇',
    '始皇帝': '秦始皇',
    '秦始皇陛下': '秦始皇',
}


def extract_name_from_query(query_name):
    """
    从查询名称中提取实际名称
    处理称谓、别名等情况

    例如:
        "汪教授" -> ("汪淼", "从'汪教授'解析而来")
        "牛顿" -> ("牛顿", "直接匹配")
    """
    query_name = query_name.strip()

    # 1. 检查是否在别名映射表中
    if query_name in ALIAS_MAP:
        return ALIAS_MAP[query_name], f"别名映射: '{query_name}' -> '{ALIAS_MAP[query_name]}'"

    # 2. 检查是否包含称谓
    for title, keywords in TITLE_KEYWORDS.items():
        for keyword in keywords:
            if keyword in query_name:
                # 提取称谓前的部分作为实际名称
                base_name = query_name.replace(keyword, '').strip()

                # 如果提取后的名称为空，说明查询可能就是称谓本身
                # 例如"教授"，需要返回特殊标记
                if not base_name:
                    return query_name, "称谓查询（需在content中查找）"

                return base_name, f"从称谓解析: '{query_name}' -> '{base_name}'"

    # 3. 直接返回原名称
    return query_name, "直接匹配"


def search_entity_with_alias(storage, query_name, threshold=0.3, max_results=20):
    """
    智能搜索实体，支持别名解析

    Args:
        storage: StorageManager实例
        query_name: 查询的实体名称
        threshold: 相似度阈值
        max_results: 最大结果数

    Returns:
        (实体列表, 解析信息)
    """
    # 先进行实体名解析
    actual_name, parse_info = extract_name_from_query(query_name)

    print(f"    [实体名解析] {parse_info}")
    print(f"    [搜索目标] '{actual_name}'")

    entities_found = []
    _seen_fids = set()

    # 策略1: 使用解析后的名称进行精确搜索
    try:
        entities = storage.search_entities_by_similarity(
            query_name=actual_name,
            threshold=threshold,
            max_results=max_results,
            text_mode="name_and_content",
            similarity_method="text"
        )
        if entities:
            entities_found.extend(entities)
            _seen_fids.update(e.family_id for e in entities)
            print(f"    [结果] 精确搜索找到 {len(entities)} 个实体")
    except Exception as e:
        print(f"    [警告] 精确搜索失败: {e}")

    # 策略2: 如果精确搜索结果少，尝试更宽松的搜索
    if len(entities_found) < 3:
        try:
            entities = storage.search_entities_by_similarity(
                query_name=actual_name,
                threshold=max(0.2, threshold - 0.1),  # 降低阈值
                max_results=max_results,
                text_mode="name_only",
                similarity_method="text"
            )
            for entity in entities:
                if entity.family_id not in _seen_fids:
                    _seen_fids.add(entity.family_id)
                    entities_found.append(entity)
            print(f"    [结果] 宽松搜索找到 {len(entities)} 个实体")
        except Exception as e:
            print(f"    [警告] 宽松搜索失败: {e}")

    # 策略3: 如果原查询包含称谓，尝试在实体的content中搜索
    if query_name != actual_name and len(entities_found) < 5:
        try:
            # 搜索所有实体，然后筛选content中包含原查询的
            all_entities = storage.get_all_entities(limit=200)
            matched = [e for e in all_entities if query_name in e.content or query_name in e.name]

            for entity in matched[:10]:
                if entity.family_id not in _seen_fids:
                    _seen_fids.add(entity.family_id)
                    entities_found.append(entity)

            if matched:
                print(f"    [结果] Content搜索找到 {len(matched)} 个实体")
        except Exception as e:
            print(f"    [警告] Content搜索失败: {e}")

    # 去重并保留最匹配的
    seen_ids = set()
    unique_entities = []
    for entity in entities_found:
        if entity.family_id not in seen_ids:
            seen_ids.add(entity.family_id)
            unique_entities.append(entity)

    return unique_entities[:max_results], parse_info


def query_entity_relations(storage, entity1_id, entity2_id):
    """
    查询两个实体之间的直接关系
    """
    return storage.get_relations_by_entities(entity1_id, entity2_id)


def get_relations_containing_keyword(storage, entity, keyword):
    """
    获取实体的所有关系中包含特定关键词的关系
    """
    relations = storage.get_entity_relations_by_family_id(entity.family_id, limit=200)
    return [r for r in relations if keyword in r.content]


def load_relevant_cache_content(storage, relations, entity1_name, entity2_name, max_caches=5):
    """
    加载相关的记忆缓存内容
    """
    cache_ids = set(rel.episode_id for rel in relations if rel.episode_id)
    _limited_ids = list(cache_ids)[:max_caches]

    # Batch-load episodes if available
    _batch_fn = getattr(storage, 'load_episodes', None)
    if _batch_fn:
        episodes = _batch_fn(_limited_ids)
        results = []
        for cache in episodes:
            if cache and entity1_name in cache.content and entity2_name in cache.content:
                results.append({
                    'cache_id': cache.absolute_id,
                    'time': cache.event_time,
                    'activity_type': getattr(cache, 'activity_type', None),
                    'content': cache.content
                })
    else:
        results = []
        for cache_id in _limited_ids:
            cache = storage.load_episode(cache_id)
            if cache and entity1_name in cache.content and entity2_name in cache.content:
                results.append({
                    'cache_id': cache_id,
                    'time': cache.event_time,
                    'activity_type': cache.activity_type,
                    'content': cache.content
                })

    return results


def print_entity_info(entity, query_name):
    """打印实体信息"""
    match_indicator = "✓" if query_name in entity.name else "→"
    print(f"  {match_indicator} 名称: {entity.name}")
    print(f"     Entity ID: {entity.family_id}")
    print(f"     描述: {entity.content[:100]}...")
    print(f"     时间: {entity.event_time}")
    print()


def print_relation_info(rel, idx, entities_map):
    """打印关系信息"""
    entity1 = entities_map.get(rel.entity1_absolute_id)
    entity2 = entities_map.get(rel.entity2_absolute_id)

    name1 = entity1.name if entity1 else "未知"
    name2 = entity2.name if entity2 else "未知"

    print(f"\n  【关系 {idx}】")
    print(f"  {name1} <-> {name2}")
    print(f"  内容: {rel.content}")
    print(f"  时间: {rel.event_time}")
    print(f"  缓存ID: {rel.episode_id}")


def _build_entities_map(storage, relations):
    """Batch-fetch all entity absolute_ids from relations, return {abs_id: Entity}."""
    abs_ids = set()
    for rel in relations:
        if rel.entity1_absolute_id:
            abs_ids.add(rel.entity1_absolute_id)
        if rel.entity2_absolute_id:
            abs_ids.add(rel.entity2_absolute_id)
    if not abs_ids:
        return {}
    entities = storage.get_entities_by_absolute_ids(list(abs_ids))
    return {e.absolute_id: e for e in entities if e}


def print_timeline(relations, entities_map):
    """打印时间线"""
    if not relations:
        return

    sorted_relations = sorted(relations, key=lambda r: r.processed_time)

    print("\n  【时间线】")
    print("  " + "-" * 76)
    for i, rel in enumerate(sorted_relations, 1):
        entity1 = entities_map.get(rel.entity1_absolute_id)
        entity2 = entities_map.get(rel.entity2_absolute_id)

        name1 = entity1.name if entity1 else "?"
        name2 = entity2.name if entity2 else "?"

        print(f"  {i}. [{rel.event_time.strftime('%Y-%m-%d %H:%M')}] {name1} <-> {name2}")
        print(f"     {rel.content[:80]}...")
        print()


def main():
    parser = argparse.ArgumentParser(
        description='查询两个实体之间的关系（支持智能实体名解析）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s --entity1 "汪教授" --entity2 "牛顿"
  %(prog)s --entity1 "牛顿" --entity2 "冯·诺依曼" --detail
  %(prog)s --entity1 "秦始皇" --entity2 "牛顿" --storage-path /path/to/graph

支持的别名和称谓:
  - 称谓: 教授、博士、先生、女士、皇帝、国王、主席、将军、统帅
  - 别名: 汪教授→汪淼, 冯·诺依曼→冯·诺伊曼, 嬴政→秦始皇
        """
    )

    parser.add_argument('--entity1', required=True, help='第一个实体的名称（支持别名、称谓）')
    parser.add_argument('--entity2', required=True, help='第二个实体的名称（支持别名、称谓）')
    parser.add_argument('--storage-path', default='/home/linkco/exa/DeepDream/graph/santi',
                       help='存储路径（默认: /home/linkco/exa/DeepDream/graph/santi）')
    parser.add_argument('--threshold', type=float, default=0.3,
                       help='实体搜索的相似度阈值（默认: 0.3）')
    parser.add_argument('--detail', action='store_true',
                       help='显示详细信息（包括记忆缓存内容）')
    parser.add_argument('--max-caches', type=int, default=3,
                       help='显示的最大记忆缓存数量（默认: 3）')
    parser.add_argument('--show-all-matches', action='store_true',
                       help='显示所有匹配的实体（包括非精确匹配）')

    args = parser.parse_args()

    # 初始化存储管理器
    storage = Neo4jStorageManager(
        args.storage_path,
        neo4j_uri=os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
        neo4j_auth=(
            os.environ.get("NEO4J_USER", "neo4j"),
            os.environ.get("NEO4J_PASSWORD", "password"),
        ),
    )

    print("=" * 80)
    print(f"查询实体关系: {args.entity1} <-> {args.entity2}")
    print("=" * 80)
    print()

    # 步骤1: 查找实体（使用智能别名解析）
    print("【步骤1】查找实体（智能别名解析）")
    print("-" * 80)

    print(f"\n查询实体1: '{args.entity1}'")
    entities1, parse_info1 = search_entity_with_alias(storage, args.entity1, threshold=args.threshold)

    print(f"\n查询实体2: '{args.entity2}'")
    entities2, parse_info2 = search_entity_with_alias(storage, args.entity2, threshold=args.threshold)

    if not entities1:
        print(f"\n错误: 未找到实体 '{args.entity1}'")
        print(f"  解析信息: {parse_info1}")
        print("提示:")
        print("  1. 尝试降低阈值 (--threshold 0.2)")
        print("  2. 检查实体名称是否正确")
        print("  3. 使用 --show-all-matches 查看所有可能的匹配")
        return

    if not entities2:
        print(f"\n错误: 未找到实体 '{args.entity2}'")
        print(f"  解析信息: {parse_info2}")
        print("提示:")
        print("  1. 尝试降低阈值 (--threshold 0.2)")
        print("  2. 检查实体名称是否正确")
        print("  3. 使用 --show-all-matches 查看所有可能的匹配")
        return

    print(f"\n找到 '{args.entity1}' 相关实体: {len(entities1)} 个")
    if args.show_all_matches and len(entities1) > 1:
        for i, entity in enumerate(entities1, 1):
            print(f"  [{i}] ", end="")
            print_entity_info(entity, args.entity1)
    else:
        print_entity_info(entities1[0], args.entity1)
        if len(entities1) > 1:
            print(f"  (还有 {len(entities1) - 1} 个同名实体，使用第一个)")
            print("  提示: 使用 --show-all-matches 查看所有匹配")

    print(f"\n找到 '{args.entity2}' 相关实体: {len(entities2)} 个")
    if args.show_all_matches and len(entities2) > 1:
        for i, entity in enumerate(entities2, 1):
            print(f"  [{i}] ", end="")
            print_entity_info(entity, args.entity2)
    else:
        print_entity_info(entities2[0], args.entity2)
        if len(entities2) > 1:
            print(f"  (还有 {len(entities2) - 1} 个同名实体，使用第一个)")
            print("  提示: 使用 --show-all-matches 查看所有匹配")

    # 步骤2: 查询直接关系
    print("\n" + "=" * 80)
    print("【步骤2】查询直接关系")
    print("-" * 80)

    entity1_id = entities1[0].family_id
    entity2_id = entities2[0].family_id

    direct_relations = query_entity_relations(storage, entity1_id, entity2_id)

    if direct_relations:
        print(f"\n找到 {len(direct_relations)} 条直接关系:\n")

        direct_entities_map = _build_entities_map(storage, direct_relations)
        for i, rel in enumerate(direct_relations, 1):
            print_relation_info(rel, i, direct_entities_map)

        print_timeline(direct_relations, direct_entities_map)
    else:
        print("\n未找到直接关系")

    # 步骤3: 查询包含对方名称的关系
    print("\n" + "=" * 80)
    print("【步骤3】查询包含对方名称的关系")
    print("-" * 80)

    # 使用实际的实体名称搜索关系
    actual_name1 = entities1[0].name
    actual_name2 = entities2[0].name

    relations1 = get_relations_containing_keyword(storage, entities1[0], actual_name2)
    relations2 = get_relations_containing_keyword(storage, entities2[0], actual_name1)

    all_indirect = relations1 + relations2

    if all_indirect:
        seen_ids = set()
        unique_relations = []
        for rel in all_indirect:
            if rel.absolute_id not in seen_ids:
                seen_ids.add(rel.absolute_id)
                unique_relations.append(rel)

        print(f"\n找到 {len(unique_relations)} 条包含对方名称的关系:\n")

        indirect_entities_map = _build_entities_map(storage, unique_relations)
        for i, rel in enumerate(unique_relations[:10], 1):
            print_relation_info(rel, i, indirect_entities_map)

        if len(unique_relations) > 10:
            print(f"\n  ... (还有 {len(unique_relations) - 10} 条关系)")
    else:
        print("\n未找到包含对方名称的关系")

    # 步骤4: 加载相关记忆缓存
    if args.detail:
        print("\n" + "=" * 80)
        print("【步骤4】加载相关记忆缓存")
        print("-" * 80)

        all_relations = direct_relations + all_indirect
        cache_contents = load_relevant_cache_content(
            storage,
            all_relations,
            actual_name1,
            actual_name2,
            max_caches=args.max_caches
        )

        if cache_contents:
            print(f"\n找到 {len(cache_contents)} 个同时包含两个实体的记忆缓存:\n")

            for i, cache_info in enumerate(cache_contents, 1):
                print(f"\n  【缓存 {i}】")
                print(f"  缓存ID: {cache_info['cache_id']}")
                print(f"  时间: {cache_info['time']}")
                print(f"  活动类型: {cache_info['activity_type']}")
                print("  " + "-" * 76)

                content = cache_info['content']
                if len(content) > 500:
                    content = content[:500] + "\n... (内容过长，已截断)"

                for line in content.split('\n'):
                    print(f"  {line}")

                print("  " + "-" * 76)
        else:
            print("\n未找到同时包含两个实体的记忆缓存")

    # 总结
    print("\n" + "=" * 80)
    print("【查询总结】")
    print("-" * 80)
    print(f"\n  查询输入: '{args.entity1}' <-> '{args.entity2}'")
    print(f"  实体1解析: {parse_info1}")
    print(f"  实体2解析: {parse_info2}")
    print(f"  实体1: {entities1[0].name} (ID: {entity1_id})")
    print(f"  实体2: {entities2[0].name} (ID: {entity2_id})")
    print(f"  直接关系数: {len(direct_relations)}")
    print(f"  间接关系数: {len(all_indirect)}")
    print(f"  总关系数: {len(direct_relations) + len(set(rel.absolute_id for rel in all_indirect) - set(rel.absolute_id for rel in direct_relations))}")

    print("\n" + "=" * 80)
    print("查询完成")
    print("=" * 80)


if __name__ == "__main__":
    main()
