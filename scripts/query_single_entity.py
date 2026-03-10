#!/usr/bin/env python3
"""
单实体深度查询脚本

功能：
- 查询实体基本信息
- 查询所有关系
- 查询历史版本
- 查询相关记忆缓存
- 生成实体报告

使用方法：
    python query_single_entity.py --entity "汪淼"
    python query_single_entity.py --entity "汪淼" --include-versions --include-cache
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# 添加Temporal_Memory_Graph到路径（支持多种路径）
import os
from pathlib import Path

# 方法1: 从scripts目录向上找
script_dir = Path(__file__).parent
tmg_dir = script_dir.parent.parent.parent
if (tmg_dir / "processor").exists():
    sys.path.insert(0, str(tmg_dir))
else:
    # 方法2: 使用绝对路径
    sys.path.insert(0, str(Path("/home/linkco/exa/Temporal_Memory_Graph")))

try:
    from processor.storage import StorageManager
except ImportError:
    print("错误: 无法导入 StorageManager")
    print("请确保 Temporal_Memory_Graph 在正确的位置")
    sys.exit(1)



def search_entity(storage, entity_name, threshold=0.3, max_results=10):
    """搜索实体"""
    entities = storage.search_entities_by_similarity(
        query_name=entity_name,
        threshold=threshold,
        max_results=max_results,
        text_mode="name_and_content",
        similarity_method="text"
    )
    return entities


def query_single_entity(storage, entity_id, include_versions=False, include_cache=False, max_relations=50):
    """深度查询单个实体"""

    # 获取实体基本信息
    entity = storage.get_entity_by_id(entity_id)
    if not entity:
        return None

    result = {
        'basic_info': entity,
        'relations': [],
        'versions': [],
        'caches': set(),
    }

    # 获取所有关系
    relations = storage.get_entity_relations_by_entity_id(entity_id, limit=max_relations)
    result['relations'] = relations

    # 获取历史版本
    if include_versions:
        versions = storage.get_entity_versions(entity_id)
        result['versions'] = versions

    # 获取相关记忆缓存
    if include_cache:
        cache_ids = set(rel.memory_cache_id for rel in relations if rel.memory_cache_id)
        result['caches'] = cache_ids

    return result


def print_basic_info(entity):
    """打印基本信息"""
    print("\n" + "=" * 80)
    print("【基本信息】")
    print("-" * 80)
    print(f"名称: {entity.name}")
    print(f"Entity ID: {entity.entity_id}")
    print(f"Absolute ID: {entity.id}")
    print(f"描述: {entity.content}")
    print(f"时间: {entity.physical_time}")
    print(f"文档: {entity.doc_name}")
    print(f"缓存ID: {entity.memory_cache_id}")


def print_relations(relations, storage, show_all=False):
    """打印关系"""
    print("\n" + "=" * 80)
    print("【关系网络】")
    print("-" * 80)
    print(f"共找到 {len(relations)} 条关系\n")

    # 按关系类型分组
    relation_groups = defaultdict(list)
    for rel in relations:
        # 获取另一端实体（需要传入中心实体的absolute_id）
        other_id = rel.entity2_absolute_id
        other_entity = storage.get_entity_by_absolute_id(other_id)

        if other_entity:
            relation_groups[other_entity.name].append({
                'relation': rel,
                'entity': other_entity
            })

    # 显示关系
    for i, (name, items) in enumerate(sorted(relation_groups.items())[:20], 1):
        item = items[0]
        rel = item['relation']
        other_entity = item['entity']

        print(f"{i}. {name}")
        print(f"   关系: {rel.content[:100]}...")
        print(f"   时间: {rel.physical_time}")
        print(f"   实体ID: {other_entity.entity_id}")
        print()

        if not show_all and i >= 10:
            remaining = len(relation_groups) - 10
            if remaining > 0:
                print(f"... (还有 {remaining} 个相关实体)")
            break


def print_versions(versions):
    """打印历史版本"""
    print("\n" + "=" * 80)
    print("【历史版本】")
    print("-" * 80)
    print(f"共找到 {len(versions)} 个版本\n")

    # 按时间排序
    sorted_versions = sorted(versions, key=lambda v: v.physical_time)

    for i, version in enumerate(sorted_versions, 1):
        print(f"版本 {i}:")
        print(f"  时间: {version.physical_time}")
        print(f"  描述: {version.content[:150]}...")
        print(f"  ID: {version.id}")
        print()


def print_memory_caches(cache_ids, storage, max_show=3):
    """打印记忆缓存"""
    print("\n" + "=" * 80)
    print(f"【相关记忆缓存】(共 {len(cache_ids)} 个，显示前 {max_show} 个)")
    print("-" * 80)

    for i, cache_id in enumerate(list(cache_ids)[:max_show], 1):
        cache = storage.load_memory_cache(cache_id)
        if cache:
            print(f"\n缓存 {i}: {cache_id}")
            print(f"时间: {cache.physical_time}")
            print(f"活动类型: {cache.activity_type}")
            print(f"内容预览: {cache.content[:200]}...")
            print("-" * 40)


def print_summary(result, storage):
    """打印总结"""
    print("\n" + "=" * 80)
    print("【查询总结】")
    print("-" * 80)

    entity = result['basic_info']
    relations = result['relations']
    versions = result['versions']
    caches = result['caches']

    print(f"实体: {entity.name}")
    print(f"Entity ID: {entity.entity_id}")
    print(f"关系数量: {len(relations)}")
    print(f"版本数量: {len(versions)}")
    print(f"相关缓存: {len(caches)}")

    # 关系统计
    if relations:
        related_entities = set()
        for rel in relations:
            other_id = rel.entity2_absolute_id
            other = storage.get_entity_by_absolute_id(other_id)
            if other:
                related_entities.add(other.name)

        print(f"关联实体数: {len(related_entities)}")
        print(f"关联实体: {', '.join(list(related_entities)[:10])}")


def main():
    parser = argparse.ArgumentParser(
        description='深度查询单个实体的所有信息',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--entity', required=True, help='实体名称')
    parser.add_argument('--storage-path', default='/home/linkco/exa/Temporal_Memory_Graph/graph/santi',
                       help='存储路径')
    parser.add_argument('--threshold', type=float, default=0.3,
                       help='相似度阈值')
    parser.add_argument('--include-versions', action='store_true',
                       help='是否显示历史版本')
    parser.add_argument('--include-cache', action='store_true',
                       help='是否显示相关记忆缓存')
    parser.add_argument('--max-relations', type=int, default=50,
                       help='最大关系数量')
    parser.add_argument('--show-all-relations', action='store_true',
                       help='显示所有关系')
    parser.add_argument('--max-cache', type=int, default=3,
                       help='显示的最大缓存数量')

    args = parser.parse_args()

    # 初始化存储管理器
    storage = StorageManager(args.storage_path)

    print("=" * 80)
    print(f"单实体深度查询: {args.entity}")
    print("=" * 80)

    # 搜索实体
    print("\n【步骤1】搜索实体")
    print("-" * 80)

    entities = search_entity(storage, args.entity, threshold=args.threshold)

    if not entities:
        print(f"\n未找到实体 '{args.entity}'")
        print("提示:")
        print("  1. 尝试降低阈值 (--threshold 0.2)")
        print("  2. 检查实体名称是否正确")
        return

    print(f"\n找到 {len(entities)} 个匹配实体，使用第一个")
    entity = entities[0]
    print(f"Entity ID: {entity.entity_id}")
    print(f"名称: {entity.name}")

    # 深度查询
    print("\n【步骤2】深度查询")
    print("-" * 80)

    result = query_single_entity(
        storage,
        entity.entity_id,
        include_versions=args.include_versions,
        include_cache=args.include_cache,
        max_relations=args.max_relations
    )

    if not result:
        print("\n查询失败")
        return

    # 显示结果
    print_basic_info(result['basic_info'])
    print_relations(result['relations'], storage, show_all=args.show_all_relations)

    if args.include_versions:
        print_versions(result['versions'])

    if args.include_cache:
        print_memory_caches(result['caches'], storage, max_show=args.max_cache)

    print_summary(result, storage)

    print("\n" + "=" * 80)
    print("查询完成")
    print("=" * 80)


if __name__ == "__main__":
    main()
