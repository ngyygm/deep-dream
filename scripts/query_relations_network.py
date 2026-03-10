#!/usr/bin/env python3
"""
关系网络查询脚本

功能：
- 查询实体的完整关系网络
- 多层级扩展
- 关键节点识别
- 生成关系图

使用方法：
    python query_relations_network.py --entity "汪淼" --depth 2
    python query_relations_network.py --entity "汪淼" --format json > network.json
"""

import sys
import argparse
import json
from pathlib import Path
from collections import defaultdict, deque

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



def build_network(storage, entity_id, max_depth=2):
    """构建关系网络（BFS）"""
    network = {
        'nodes': {},  # entity_id -> {name, depth}
        'edges': [],  # list of (from_id, to_id, relation)
        'levels': defaultdict(set)  # depth -> set of entity_ids
    }

    # 获取起始实体
    start_entity = storage.get_entity_by_id(entity_id)
    if not start_entity:
        return None

    network['nodes'][start_entity.entity_id] = {
        'name': start_entity.name,
        'depth': 0
    }
    network['levels'][0].add(start_entity.entity_id)

    # BFS扩展
    visited = set([start_entity.id])  # 使用absolute_id
    queue = deque([(start_entity.id, 0)])

    while queue and queue[0][1] < max_depth:
        current_id, depth = queue.popleft()

        # 获取关系 - 使用absolute_id获取关系
        # 先获取当前实体，再获取关系
        current_entity = storage.get_entity_by_absolute_id(current_id)
        if not current_entity:
            continue

        relations = storage.get_entity_relations_by_entity_id(
            current_entity.entity_id,  # 使用entity_id
            limit=50
        )

        for rel in relations:
            # 获取另一端实体
            other_id = rel.entity2_absolute_id if rel.entity1_absolute_id != current_id else rel.entity1_absolute_id
            other_entity = storage.get_entity_by_absolute_id(other_id)

            if not other_entity:
                continue

            # 添加节点
            if other_entity.entity_id not in visited:
                visited.add(other_entity.entity_id)
                network['nodes'][other_entity.entity_id] = {
                    'name': other_entity.name,
                    'depth': depth + 1
                }
                network['levels'][depth + 1].add(other_entity.entity_id)
                queue.append((other_entity.entity_id, depth + 1))

            # 添加边
            network['edges'].append({
                'from': current_id,
                'to': other_entity.entity_id,
                'relation': rel.content[:100],
                'time': rel.physical_time.isoformat()
            })

    return network


def find_key_nodes(network):
    """识别关键节点"""
    # 计算连接数
    connections = defaultdict(int)
    for edge in network['edges']:
        connections[edge['from']] += 1
        connections[edge['to']] += 1

    # 排序
    sorted_nodes = sorted(connections.items(), key=lambda x: x[1], reverse=True)

    return sorted_nodes[:10]


def print_network_text(network):
    """以文本格式打印网络"""
    print("\n" + "=" * 80)
    print("【关系网络】")
    print("-" * 80)

    # 按层级打印
    for depth in sorted(network['levels'].keys()):
        entity_ids = network['levels'][depth]
        print(f"\n层级 {depth} ({len(entity_ids)} 个实体):")

        for entity_id in list(entity_ids)[:20]:
            node = network['nodes'][entity_id]
            print(f"  - {node['name']} (ID: {entity_id})")

        if len(entity_ids) > 20:
            print(f"  ... (还有 {len(entity_ids) - 20} 个)")

    # 关键节点
    print("\n" + "=" * 80)
    print("【关键节点】(按连接数排序)")
    print("-" * 80)

    key_nodes = find_key_nodes(network)
    for i, (entity_id, count) in enumerate(key_nodes, 1):
        node = network['nodes'].get(entity_id)
        if node:
            print(f"{i}. {node['name']} - {count} 个连接")


def export_network_json(network, output_file):
    """导出为JSON"""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(network, f, ensure_ascii=False, indent=2)
    print(f"\n网络已导出到: {output_file}")


def export_network_graphviz(network, output_file):
    """导出为Graphviz DOT格式"""
    dot = ["digraph G {"]
    dot.append("  rankdir=LR;")
    dot.append("  node [shape=box];")

    # 添加节点
    for entity_id, node in network['nodes'].items():
        label = node['name'].replace('"', '\\"')
        depth = node['depth']
        color = ["lightblue", "lightgreen", "lightyellow"][min(depth, 2)]
        dot.append(f'  "{entity_id}" [label="{label}", fillcolor="{color}", style="filled"];')

    # 添加边
    for edge in network['edges'][:50]:  # 限制边的数量
        dot.append(f'  "{edge["from"]}" -> "{edge["to"]}";')

    dot.append("}")

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(dot))

    print(f"\nGraphviz文件已导出到: {output_file}")
    print("生成命令: dot -Tpng network.dot -o network.png")


def export_network_markdown(network, output_file):
    """导出为Markdown表格"""
    lines = []
    lines.append("# 关系网络\n")
    lines.append("## 节点列表\n")
    lines.append("| 层级 | 实体名称 | Entity ID |\n")
    lines.append("|------|----------|------------|\n")

    for depth in sorted(network['levels'].keys()):
        for entity_id in network['levels'][depth]:
            node = network['nodes'][entity_id]
            lines.append(f"| {depth} | {node['name']} | {entity_id} |\n")

    lines.append("\n## 关系列表\n")
    lines.append("| 从 | 到 | 关系 |\n")
    lines.append("|----|---|------|\n")

    for edge in network['edges'][:50]:
        from_node = network['nodes'].get(edge['from'], {}).get('name', edge['from'])
        to_node = network['nodes'].get(edge['to'], {}).get('name', edge['to'])
        relation = edge['relation'][:50].replace('\n', ' ')
        lines.append(f"| {from_node} | {to_node} | {relation}... |\n")

    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(lines)

    print(f"\nMarkdown文件已导出到: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='查询实体的关系网络',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--entity', required=True, help='中心实体名称')
    parser.add_argument('--storage-path', default='/home/linkco/exa/Temporal_Memory_Graph/graph/santi')
    parser.add_argument('--depth', type=int, default=2,
                       help='扩展深度（默认2层）')
    parser.add_argument('--format', choices=['text', 'json', 'graphviz', 'markdown'],
                       default='text', help='输出格式')
    parser.add_argument('--output', help='输出文件路径（json/graphviz/markdown格式需要）')
    parser.add_argument('--threshold', type=float, default=0.3)

    args = parser.parse_args()

    storage = StorageManager(args.storage_path)

    print("=" * 80)
    print(f"关系网络查询: {args.entity}")
    print(f"扩展深度: {args.depth}")
    print("=" * 80)

    # 搜索实体
    print("\n【步骤1】搜索实体")
    print("-" * 80)

    entities = storage.search_entities_by_similarity(
        query_name=args.entity,
        threshold=args.threshold,
        max_results=1
    )

    if not entities:
        print(f"\n未找到实体 '{args.entity}'")
        return

    entity = entities[0]
    print(f"\n找到实体: {entity.name} (ID: {entity.entity_id})")

    # 构建网络
    print(f"\n【步骤2】构建关系网络（深度: {args.depth}）")
    print("-" * 80)

    network = build_network(storage, entity.entity_id, max_depth=args.depth)

    if not network or not network['nodes']:
        print("\n构建网络失败")
        return

    node_count = len(network['nodes'])
    edge_count = len(network['edges'])

    print(f"\n网络构建完成")
    print(f"节点数: {node_count}")
    print(f"边数: {edge_count}")

    # 输出
    if args.format == 'text':
        print_network_text(network)

    elif args.format == 'json':
        output_file = args.output or f"network_{entity.entity_id}.json"
        export_network_json(network, output_file)

    elif args.format == 'graphviz':
        output_file = args.output or f"network_{entity.entity_id}.dot"
        export_network_graphviz(network, output_file)

    elif args.format == 'markdown':
        output_file = args.output or f"network_{entity.entity_id}.md"
        export_network_markdown(network, output_file)

    print("\n" + "=" * 80)
    print("查询完成")
    print("=" * 80)


if __name__ == "__main__":
    main()
