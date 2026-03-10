#!/usr/bin/env python3
"""
实体搜索脚本

功能：
- 多策略搜索实体
- 支持模糊匹配
- 显示所有候选
- 支持交互式选择

使用方法：
    python search_entities.py --query "汪"
    python search_entities.py --query "教授" --search-mode all --max-results 20
"""

import sys
import argparse
from pathlib import Path

# 添加Temporal_Memory_Graph到路径（支持多种路径）
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



def search_by_name(storage, query, threshold=0.5, max_results=10):
    """按名称搜索"""
    return storage.search_entities_by_similarity(
        query_name=query,
        threshold=threshold,
        max_results=max_results,
        text_mode="name_only",
        similarity_method="text"
    )


def search_by_content(storage, query, threshold=0.3, max_results=10):
    """按内容搜索"""
    return storage.search_entities_by_similarity(
        query_name=query,
        query_content=query,
        threshold=threshold,
        max_results=max_results,
        text_mode="content_only",
        similarity_method="text"
    )


def search_by_all(storage, query, threshold=0.3, max_results=10):
    """综合搜索"""
    return storage.search_entities_by_similarity(
        query_name=query,
        query_content=query,
        threshold=threshold,
        max_results=max_results,
        text_mode="name_and_content",
        similarity_method="text"
    )


def search_in_all_entities(storage, keyword, max_results=20):
    """在所有实体中搜索关键词"""
    all_entities = storage.get_all_entities(limit=500)
    matched = []

    for entity in all_entities:
        if keyword.lower() in entity.name.lower() or keyword.lower() in entity.content.lower():
            matched.append(entity)
            if len(matched) >= max_results:
                break

    return matched


def print_entities(entities, show_numbered=False):
    """打印实体列表"""
    if not entities:
        print("未找到匹配的实体")
        return

    print(f"\n找到 {len(entities)} 个匹配实体:\n")

    for i, entity in enumerate(entities, 1 if show_numbered else 0):
        if show_numbered:
            print(f"[{i}] ", end="")

        print(f"名称: {entity.name}")
        print(f"Entity ID: {entity.entity_id}")
        print(f"描述: {entity.content[:100]}...")
        print(f"时间: {entity.physical_time}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description='多策略搜索实体',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--query', required=True, help='搜索关键词')
    parser.add_argument('--storage-path', default='/home/linkco/exa/Temporal_Memory_Graph/graph/santi')
    parser.add_argument('--search-mode', choices=['name', 'content', 'all'],
                       default='all', help='搜索模式')
    parser.add_argument('--threshold', type=float, default=0.3)
    parser.add_argument('--max-results', type=int, default=20)
    parser.add_argument('--numbered', action='store_true',
                       help='显示编号，便于选择')

    args = parser.parse_args()

    storage = StorageManager(args.storage_path)

    print("=" * 80)
    print(f"实体搜索: {args.query}")
    print(f"搜索模式: {args.search_mode}")
    print("=" * 80)

    entities = []

    print(f"\n【步骤1】执行搜索 (模式: {args.search_mode})")
    print("-" * 80)

    if args.search_mode == 'name':
        entities = search_by_name(storage, args.query, args.threshold, args.max_results)
    elif args.search_mode == 'content':
        entities = search_by_content(storage, args.query, args.threshold, args.max_results)
    else:
        # 综合搜索：先精确，后模糊
        entities = search_by_all(storage, args.query, args.threshold, args.max_results)

        # 如果结果少，尝试关键词搜索
        if len(entities) < 5:
            print(f"  精确搜索找到 {len(entities)} 个，尝试关键词搜索...")
            keyword_entities = search_in_all_entities(storage, args.query, args.max_results)
            for entity in keyword_entities:
                if entity.entity_id not in [e.entity_id for e in entities]:
                    entities.append(entity)

    print(f"\n搜索完成，找到 {len(entities)} 个匹配实体")

    print_entities(entities, show_numbered=args.numbered)

    print("=" * 80)
    print("查询完成")
    print("=" * 80)


if __name__ == "__main__":
    main()
