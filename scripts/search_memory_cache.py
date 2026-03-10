#!/usr/bin/env python3
"""
记忆缓存搜索脚本

功能：
- 在记忆缓存中搜索关键词
- 提取原文片段
- 上下文扩展
- 支持多实体搜索

使用方法：
    python search_memory_cache.py --keywords "汪淼 牛顿"
    python search_memory_cache.py --keywords "对话" --entities "汪淼,牛顿" --context-size 500
"""

import sys
import argparse
from pathlib import Path
from collections import defaultdict

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



def search_caches(storage, keywords, entity_names=None, max_results=20):
    """搜索记忆缓存"""
    cache_json_dir = storage.cache_json_dir
    cache_files = list(cache_json_dir.glob("*.json"))

    # 如果指定了实体，优先搜索包含实体的缓存
    if entity_names:
        # 获取实体ID
        entity_ids = set()
        for name in entity_names.split(','):
            name = name.strip()
            entities = storage.search_entities_by_similarity(
                query_name=name,
                threshold=0.5,
                max_results=1
            )
            if entities:
                entity_ids.add(entities[0].entity_id)

    matched_caches = []
    keywords_list = [k.lower() for k in keywords.split(',')]

    for cache_file in cache_files:
        try:
            cache = storage.load_memory_cache(cache_file.stem)
            if cache:
                content = cache.content.lower()

                # 检查是否包含所有关键词
                if all(kw in content for kw in keywords_list):
                    matched_caches.append(cache)
                    if len(matched_caches) >= max_results:
                        break
        except:
            continue

    return matched_caches


def extract_context(text, keywords, context_size=200):
    """提取包含关键词的上下文"""
    keywords_list = [k.lower() for k in keywords.split(',')]

    # 找到关键词位置
    positions = []
    for kw in keywords_list:
        pos = text.lower().find(kw)
        if pos != -1:
            positions.append(pos)

    if not positions:
        return text[:context_size * 2]

    # 找到最早和最晚位置
    min_pos = max(0, min(positions) - context_size)
    max_pos = min(len(text), max(positions) + context_size)

    return text[min_pos:max_pos]


def print_search_results(caches, keywords, context_size=200):
    """打印搜索结果"""
    if not caches:
        print("\n未找到匹配的记忆缓存")
        return

    print(f"\n找到 {len(caches)} 个匹配的记忆缓存:\n")

    for i, cache in enumerate(caches, 1):
        print("=" * 80)
        print(f"缓存 {i}: {cache.id}")
        print(f"时间: {cache.physical_time}")
        print(f"活动类型: {cache.activity_type}")
        print("-" * 80)

        # 提取上下文
        context = extract_context(cache.content, keywords, context_size)

        # 高亮关键词
        for line in context.split('\n'):
            print(line)

        if len(cache.content) > len(context):
            print(f"\n... (还有 {len(cache.content) - len(context)} 个字符)")

        print()


def main():
    parser = argparse.ArgumentParser(
        description='在记忆缓存中搜索内容',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--keywords', required=True, help='搜索关键词（逗号分隔）')
    parser.add_argument('--storage-path', default='/home/linkco/exa/Temporal_Memory_Graph/graph/santi')
    parser.add_argument('--entities', help='相关实体（逗号分隔）')
    parser.add_argument('--max-results', type=int, default=20)
    parser.add_argument('--context-size', type=int, default=300,
                       help='上下文大小（字符数）')

    args = parser.parse_args()

    storage = StorageManager(args.storage_path)

    print("=" * 80)
    print(f"记忆缓存搜索")
    print(f"关键词: {args.keywords}")
    if args.entities:
        print(f"相关实体: {args.entities}")
    print("=" * 80)

    print("\n【步骤1】搜索记忆缓存")
    print("-" * 80)

    caches = search_caches(
        storage,
        args.keywords,
        entity_names=args.entities,
        max_results=args.max_results
    )

    print(f"\n搜索完成")

    print_search_results(caches, args.keywords, args.context_size)

    print("=" * 80)
    print("查询完成")
    print("=" * 80)


if __name__ == "__main__":
    main()
