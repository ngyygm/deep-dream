#!/usr/bin/env python3
"""
上下文扩展查询脚本

功能：
- 当信息不足时扩展上下文
- 加载前后缓存
- 扩展相关实体
- 深度检索

使用方法：
    python expand_context.py --cache-id "cache_xxx" --expand-range 5
    python expand_context.py --entity "汪淼" --auto-expand
"""

import sys
import argparse
from pathlib import Path
import json
from datetime import datetime

# 添加DeepDream到路径（支持多种路径）
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
    from core.storage import StorageManager
except ImportError:
    print("错误: 无法导入 StorageManager")
    print("请确保 DeepDream 在正确的位置")
    sys.exit(1)



def get_cache_list_sorted(storage):
    """获取按时间排序的缓存列表"""
    cache_json_dir = storage.cache_json_dir
    cache_files = list(cache_json_dir.glob("*.json"))

    cache_list = []
    for cache_file in cache_files:
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
                cache_time = datetime.fromisoformat(metadata['event_time'])
                cache_list.append({
                    'id': metadata['id'],
                    'time': cache_time,
                    'file': cache_file
                })
        except:
            continue

    # 按时间排序
    cache_list.sort(key=lambda x: x['time'])
    return cache_list


def expand_from_cache(storage, cache_id, expand_range=5):
    """从指定缓存扩展上下文"""
    cache_list = get_cache_list_sorted(storage)

    # 找到当前缓存的位置 — O(1) dict lookup instead of O(n) scan
    _id_to_idx = {c['id']: i for i, c in enumerate(cache_list)}
    current_idx = _id_to_idx.get(cache_id, -1)

    if current_idx == -1:
        return []

    # 获取前后缓存
    start_idx = max(0, current_idx - expand_range)
    end_idx = min(len(cache_list), current_idx + expand_range + 1)

    expanded_caches = cache_list[start_idx:end_idx]

    # 加载缓存内容 — batch load
    _batch_fn = getattr(storage, 'load_episodes', None)
    if _batch_fn and expanded_caches:
        _ids = [info['id'] for info in expanded_caches]
        results = [c for c in _batch_fn(_ids) if c]
    else:
        results = []
        for cache_info in expanded_caches:
            cache = storage.load_episode(cache_info['id'])
            if cache:
                results.append(cache)

    return results


def expand_from_entity(storage, entity_name, expand_depth=2):
    """从实体扩展上下文"""
    # 搜索实体
    entities = storage.search_entities_by_similarity(
        query_name=entity_name,
        threshold=0.3,
        max_results=1
    )

    if not entities:
        return []

    entity = entities[0]

    # 获取关系
    relations = storage.get_entity_relations_by_family_id(
        entity.family_id,
        limit=100
    )

    # 收集缓存
    cache_ids = set(rel.episode_id for rel in relations if rel.episode_id)

    # 如果需要更深层的扩展
    if expand_depth > 1:
        # Batch-fetch all other entities at once (avoid N+1)
        other_abs_ids = []
        for rel in relations[:20]:
            other_id = rel.entity2_absolute_id if rel.entity1_absolute_id != entity.absolute_id else rel.entity1_absolute_id
            other_abs_ids.append(other_id)

        other_entities_list = storage.get_entities_by_absolute_ids(other_abs_ids)
        other_family_ids = list({e.family_id for e in other_entities_list if e})

        # Batch-fetch relations for all other entities' family_ids
        _batch_rels_fn = getattr(storage, 'get_relations_by_family_ids', None)
        _use_fallback = True
        if _batch_rels_fn:
            try:
                all_other_rels = _batch_rels_fn(other_family_ids, limit=20)
                for other_rel in all_other_rels:
                    if other_rel.episode_id:
                        cache_ids.add(other_rel.episode_id)
                _use_fallback = False
            except Exception:
                pass
        if _use_fallback:
            for fid in other_family_ids:
                other_relations = storage.get_entity_relations_by_family_id(fid, limit=20)
                for other_rel in other_relations:
                    if other_rel.episode_id:
                        cache_ids.add(other_rel.episode_id)

    # Batch-load episodes
    _batch_fn = getattr(storage, 'load_episodes', None)
    if _batch_fn:
        results = _batch_fn(list(cache_ids)[:30])
    else:
        results = []
        for cache_id in list(cache_ids)[:30]:
            cache = storage.load_episode(cache_id)
            if cache:
                results.append(cache)

    # 按时间排序
    results.sort(key=lambda c: c.event_time)

    return results


def assess_information_completeness(cache_content, expected_keywords=None):
    """评估信息完整度"""
    score = 0
    max_score = 0

    issues = []

    # 检查内容长度
    max_score += 1
    if len(cache_content) > 500:
        score += 1
    else:
        issues.append("内容长度较短")

    # 检查关键词
    if expected_keywords:
        max_score += 1
        keywords = [kw.lower() for kw in expected_keywords.split(',')]
        _content_lower = cache_content.lower()
        found = sum(1 for kw in keywords if kw in _content_lower)
        if found >= len(keywords):
            score += 1
        else:
            issues.append(f"缺少部分关键词: {keywords}")

    # 检查对话内容
    max_score += 1
    if any(marker in cache_content for marker in ['"', '"', '：', '?', '？', '说', '问', '答']):
        score += 1
    else:
        issues.append("没有明显的对话内容")

    # 检查描述完整性
    max_score += 1
    if len(cache_content) > 1000 and '。' in cache_content:
        score += 1
    else:
        issues.append("描述可能不够完整")

    completeness = score / max_score if max_score > 0 else 0

    return completeness, issues


def print_expanded_caches(caches, center_cache_id=None):
    """打印扩展的缓存"""
    if not caches:
        print("\n未找到可扩展的缓存")
        return

    print(f"\n找到 {len(caches)} 个相关缓存:\n")

    for i, cache in enumerate(caches, 1):
        is_center = cache.absolute_id == center_cache_id if center_cache_id else False
        marker = "★ " if is_center else f"{i}. "

        print(f"{marker}缓存: {cache.absolute_id}")
        print(f"   时间: {cache.event_time}")
        print(f"   类型: {cache.activity_type}")

        # 评估完整度
        completeness, issues = assess_information_completeness(cache.content)
        print(f"   完整度: {completeness*100:.0f}%")
        if issues and completeness < 0.8:
            print(f"   问题: {', '.join(issues)}")

        # 内容预览
        preview = cache.content[:150].replace('\n', ' ')
        print(f"   预览: {preview}...")
        print()


def auto_expand_and_search(storage, entity_name, max_iterations=3):
    """自动扩展并搜索，直到找到足够信息"""
    print("\n【自动扩展模式】")
    print("-" * 80)

    # 第一轮：直接搜索实体
    print(f"第1轮: 搜索实体 '{entity_name}'")
    entities = storage.search_entities_by_similarity(
        query_name=entity_name,
        threshold=0.3,
        max_results=1
    )

    if not entities:
        print("  → 未找到实体")
        return []

    entity = entities[0]
    print(f"  → 找到实体: {entity.name} (ID: {entity.family_id})")

    # 第二轮：获取关系和缓存
    print(f"第2轮: 获取相关缓存")
    caches = expand_from_entity(storage, entity_name, expand_depth=1)
    print(f"  → 找到 {len(caches)} 个相关缓存")

    # 检查是否需要继续扩展
    if caches:
        # 评估最新缓存的完整度
        latest_cache = caches[-1]
        completeness, issues = assess_information_completeness(latest_cache.content)

        print(f"  → 最新缓存完整度: {completeness*100:.0f}%")

        if completeness < 0.7 and len(caches) < max_iterations * 10:
            print(f"第3轮: 深度扩展（因完整度不足）")
            caches = expand_from_entity(storage, entity_name, expand_depth=2)
            print(f"  → 扩展到 {len(caches)} 个缓存")
        else:
            print(f"  → 信息完整度足够，无需扩展")
    else:
        print(f"  → 未找到缓存，尝试从相关实体扩展")

    return caches


def main():
    parser = argparse.ArgumentParser(
        description='上下文扩展查询',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--cache-id', help='起始缓存ID')
    parser.add_argument('--entity', help='实体名称')
    parser.add_argument('--storage-path', default='/home/linkco/exa/DeepDream/graph/santi')
    parser.add_argument('--expand-range', type=int, default=5,
                       help='从缓存扩展时的范围（前后各N个）')
    parser.add_argument('--expand-depth', type=int, default=2,
                       help='从实体扩展时的深度')
    parser.add_argument('--auto-expand', action='store_true',
                       help='自动判断并扩展信息不足的情况')
    parser.add_argument('--max-results', type=int, default=30,
                       help='最大结果数量')

    args = parser.parse_args()

    storage = StorageManager(args.storage_path)

    print("=" * 80)
    print("上下文扩展查询")
    print("=" * 80)

    caches = []

    if args.auto_expand and args.entity:
        # 自动扩展模式
        caches = auto_expand_and_search(storage, args.entity)

    elif args.cache_id:
        # 从缓存扩展
        print(f"\n【从缓存扩展】")
        print(f"起始缓存: {args.cache_id}")
        print(f"扩展范围: 前后各 {args.expand_range} 个")
        print("-" * 80)

        caches = expand_from_cache(storage, args.cache_id, args.expand_range)

    elif args.entity:
        # 从实体扩展
        print(f"\n【从实体扩展】")
        print(f"实体: {args.entity}")
        print(f"扩展深度: {args.expand_depth}")
        print("-" * 80)

        caches = expand_from_entity(storage, args.entity, args.expand_depth)

    else:
        print("\n错误: 必须指定 --cache-id 或 --entity")
        return

    # 显示结果
    print_expanded_caches(caches, args.cache_id)

    print("=" * 80)
    print(f"查询完成，共找到 {len(caches)} 个缓存")
    print("=" * 80)


if __name__ == "__main__":
    main()
