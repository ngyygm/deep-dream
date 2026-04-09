"""
图谱可视化 Web 服务
提供实时查看图谱可视化的 Web 界面
"""
import sys
import json
import os
from collections import deque
from pathlib import Path
from typing import List, Optional
from flask import Flask, render_template, jsonify

# 添加项目根目录到 sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from processor import StorageManager, EmbeddingClient
from .visualizer import GraphVisualizer


# Load HTML template
_TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"


class GraphWebServer:
    """图谱可视化 Web 服务器"""
    
    def __init__(self, storage_path: str = "./graph/default", port: int = 5000,
                 embedding_model_path: Optional[str] = None,
                 embedding_model_name: Optional[str] = None,
                 embedding_device: str = "cpu",
                 embedding_use_local: bool = True):
        """
        初始化 Web 服务器
        
        Args:
            storage_path: 存储路径
            port: 服务器端口
            embedding_model_path: 本地embedding模型路径（优先使用）
            embedding_model_name: HuggingFace embedding模型名称
            embedding_device: 计算设备 ("cpu" 或 "cuda")
            embedding_use_local: 是否优先使用本地模型
        """
        self.storage_path = storage_path
        self._base_storage_path = os.path.dirname(storage_path) if os.path.isdir(storage_path) and not os.path.basename(storage_path) else storage_path
        self.port = port
        self.app = Flask(__name__)
        
        # 初始化embedding客户端
        self.embedding_client = EmbeddingClient(
            model_path=embedding_model_path,
            model_name=embedding_model_name,
            device=embedding_device,
            use_local=embedding_use_local
        )
        
        # 初始化存储和可视化器
        self.storage = StorageManager(storage_path, embedding_client=self.embedding_client)
        self.visualizer = GraphVisualizer(self.storage)
        
        # 缓存当前使用的存储路径（用于路径切换检测）
        self._current_storage_path = storage_path
        
        # 设置路由
        self._setup_routes()
    
    def _switch_storage_path(self, new_path: str):
        """
        切换存储路径

        Args:
            new_path: 新的存储路径（必须在 base_storage_path 目录下）
        """
        if new_path != self._current_storage_path:
            # 安全校验：路径必须在存储根目录下
            base = Path(self._base_storage_path).resolve()
            target = (base / new_path).resolve() if not os.path.isabs(new_path) else Path(new_path).resolve()
            try:
                target.relative_to(base)
            except ValueError:
                raise ValueError(f"存储路径必须在 {base} 目录下: {new_path}")
            try:
                # 重新初始化存储和可视化器
                self.storage = StorageManager(str(target), embedding_client=self.embedding_client)
                self.visualizer = GraphVisualizer(self.storage)
                self._current_storage_path = str(target)
                print(f"✅ 已切换到新的存储路径: {target}")
            except Exception as e:
                print(f"❌ 切换存储路径失败: {str(e)}")
                raise
    
    def _setup_routes(self):
        """设置 Flask 路由"""
        
        @self.app.route('/api/status')
        def health():
            """健康检查，与 service_api 响应格式一致。/api/status 为常见监控/前端探测路径别名。"""
            try:
                embedding_available = (
                    self.embedding_client is not None
                    and getattr(self.embedding_client, 'is_available', lambda: True)()
                )
                return jsonify({
                    'success': True,
                    'data': {
                        'storage_path': str(self._current_storage_path),
                        'embedding_available': embedding_available,
                    },
                })
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/')
        def index():
            """主页"""
            return render_template('graph.html')
        
        @self.app.route('/api/graphs/data')
        def get_graph_data():
            """获取图谱数据 API
            
            支持参数:
            - limit_entities: 限制返回的实体数量（默认100）
            - limit_edges_per_entity: 每个实体最多返回的关系边数量（默认50）
            - time_point: ISO格式的时间点（可选），如果提供，只返回该时间点之前或等于该时间点的数据
            - storage_path: 图谱存储路径（可选），如果提供且与当前路径不同，会切换存储路径
            - focus_family_id: 聚焦的实体family ID（可选），如果提供，只显示该实体从最早版本到指定版本的所有关系
            - focus_absolute_id: 聚焦的实体版本absolute_id（可选），需要与focus_family_id一起使用
            - hops: 跳数（默认1），在focus模式下，表示要显示多少层关联实体和关系
            """
            try:
                from flask import request
                from datetime import datetime
                
                # 获取参数
                limit_entities = request.args.get('limit_entities', type=int, default=100)
                limit_edges_per_entity = request.args.get('limit_edges_per_entity', type=int, default=50)
                time_point_str = request.args.get('time_point')
                storage_path_param = request.args.get('storage_path')
                focus_family_id = request.args.get('focus_family_id')
                focus_absolute_id = request.args.get('focus_absolute_id')
                hops = request.args.get('hops', type=int, default=1)
                
                # 如果提供了存储路径参数，且与当前路径不同，则切换存储路径
                if storage_path_param and storage_path_param.strip():
                    storage_path_param = storage_path_param.strip()
                    try:
                        self._switch_storage_path(storage_path_param)
                    except Exception as e:
                        return jsonify({
                            'success': False,
                            'error': f'切换存储路径失败: {str(e)}'
                        }), 400
                
                time_point = None
                if time_point_str:
                    try:
                        time_point = datetime.fromisoformat(time_point_str)
                    except (ValueError, TypeError):
                        pass

                # 如果指定了focus_family_id和focus_absolute_id，以该实体为中心显示图谱
                if focus_family_id and focus_absolute_id:
                    # 获取聚焦实体的指定版本
                    focus_entity = self.storage.get_entity_by_absolute_id(focus_absolute_id)
                    if not focus_entity or focus_entity.family_id != focus_family_id:
                        return jsonify({
                            'success': False,
                            'error': f'未找到指定的实体版本: {focus_family_id}/{focus_absolute_id}'
                        }), 404
                    
                    # 只显示该实体从最早版本到指定版本的所有关系
                    # 时间点自动从该版本获取，不需要单独传递time_point参数
                    entities = [focus_entity]
                    focus_time_point = focus_entity.event_time
                else:
                    # 获取最近更新的实体（限制数量）
                    if time_point:
                        # 根据时间点获取实体
                        entities = self.storage.get_all_entities_before_time(time_point, limit=limit_entities)
                    else:
                        entities = self.storage.get_all_entities(limit=limit_entities)
                    
                    if not entities:
                        return jsonify({
                            'success': False,
                            'error': '没有找到实体数据'
                        })
                
                # 收集所有需要显示的实体ID（初始实体 + 关联实体）
                entity_absolute_ids = {entity.absolute_id for entity in entities}
                family_id_to_name = {}
                family_id_to_absolute_id = {}
                
                # 定义跳数颜色映射函数
                def get_hop_color(hop_level):
                    """根据跳数层级返回对应的颜色"""
                    colors = [
                        '#4A90E2',  # 第0跳（focus实体）：蓝色
                        '#E67E22',  # 第1跳：橙色
                        '#27AE60',  # 第2跳：绿色
                        '#9B59B6',  # 第3跳：紫色
                        '#E74C3C',  # 第4跳：红色
                        '#F39C12',  # 第5跳：黄色
                        '#1ABC9C',  # 第6跳：青色
                        '#34495E',  # 第7跳：深灰色
                    ]
                    # 如果跳数超过预定义颜色数量，循环使用
                    return colors[hop_level % len(colors)]
                
                # 记录每个实体所在的跳数层级（用于颜色区分）
                family_id_to_hop_level = {}
                
                # 构建初始节点数据
                nodes = []
                for entity in entities:
                    try:
                        # 防御性检查：确保实体有必要的属性
                        if not entity or not hasattr(entity, 'family_id') or not hasattr(entity, 'absolute_id'):
                            print(f"⚠️  跳过无效实体: {entity}")
                            continue
                        
                        family_id_to_name[entity.family_id] = entity.name
                        family_id_to_absolute_id[entity.family_id] = entity.absolute_id
                        # focus实体是第0跳
                        if focus_family_id and focus_family_id == entity.family_id:
                            family_id_to_hop_level[entity.family_id] = 0
                        
                        # 获取版本数量
                        try:
                            versions = self.storage.get_entity_versions(entity.family_id)
                            version_count = len(versions) if versions else 0
                        except Exception as e:
                            print(f"⚠️  获取实体版本失败 (family_id={entity.family_id}): {str(e)}")
                            versions = []
                            version_count = 1  # 默认值
                        
                        # 在focus模式下，显示当前版本索引（如 "实体名 (3/5版本)"）
                        # 否则显示总版本数（如 "实体名 (5版本)"）
                        if focus_family_id and focus_family_id == entity.family_id and focus_absolute_id:
                            # 找到当前版本在版本列表中的索引（从1开始）
                            # 版本列表按时间倒序排列（最新版本在前），需要反转后计算索引
                            try:
                                versions_sorted = sorted(
                                    versions,
                                    key=lambda v: self.storage._normalize_datetime_for_compare(v.processed_time)
                                )
                                current_version_index = None
                                for idx, v in enumerate(versions_sorted, 1):
                                    if v.absolute_id == focus_absolute_id:
                                        current_version_index = idx
                                        break

                                if current_version_index:
                                    label = f"{entity.name} ({current_version_index}/{version_count}版本)" if version_count > 1 else entity.name
                                else:
                                    label = f"{entity.name} ({version_count}版本)" if version_count > 1 else entity.name
                            except Exception as e:
                                print(f"⚠️  处理版本索引时出错 (family_id={entity.family_id}): {str(e)}")
                                label = f"{entity.name} ({version_count}版本)" if version_count > 1 else entity.name
                        else:
                            # 在标签中显示版本数量
                            label = f"{entity.name} ({version_count}版本)" if version_count > 1 else entity.name
                        
                        # 根据跳数设置颜色
                        hop_level = family_id_to_hop_level.get(entity.family_id, 0)
                        node_color = get_hop_color(hop_level)
                        
                        # 安全地处理event_time和processed_time
                        try:
                            event_time_str = entity.event_time.isoformat() if entity.event_time else None
                            processed_time_str = entity.processed_time.isoformat() if entity.processed_time else None
                        except Exception as e:
                            print(f"⚠️  实体时间格式错误 (family_id={entity.family_id}): {str(e)}")
                            event_time_str = None
                            processed_time_str = None

                        # 安全地处理content
                        content = entity.content if hasattr(entity, 'content') and entity.content else ''
                        name = entity.name if hasattr(entity, 'name') and entity.name else '未知实体'

                        nodes.append({
                            'id': entity.family_id,
                            'family_id': entity.family_id,
                            'absolute_id': entity.absolute_id,
                            'label': label,
                            'title': f"{name}\n\n{content[:100]}..." if len(content) > 100 else f"{name}\n\n{content}",
                            'content': content,
                            'event_time': event_time_str,
                            'processed_time': processed_time_str,
                            'version_count': version_count,
                            'color': node_color,
                            'shape': 'dot',
                            'size': 20,
                            'font': {'color': 'white'}
                        })
                    except Exception as e:
                        import traceback
                        print(f"⚠️  处理实体时发生错误 (entity={entity}): {str(e)}")
                        print(traceback.format_exc())
                        continue  # 跳过有问题的实体，继续处理其他实体
                
                # 为每个实体获取关系边（限制数量）
                edges = []
                edges_seen = set()  # 用于去重，使用 (from_id, to_id, family_id) 作为唯一标识
                all_related_family_ids = set()
                
                # 辅助函数：统计实体拥有的关系边数量（去重后）
                def count_entity_relations(family_id, max_abs_id=None):
                    """统计实体拥有的关系边数量（去重后）"""
                    if max_abs_id:
                        entity_abs_ids = self.storage.get_entity_absolute_ids_up_to_version(family_id, max_abs_id)
                    else:
                        versions = self.storage.get_entity_versions(family_id)
                        entity_abs_ids = [v.absolute_id for v in versions]
                    
                    if not entity_abs_ids:
                        return 0
                    
                    relations = self.storage.get_relations_by_entity_absolute_ids(entity_abs_ids, limit=None)
                    # 按 family_id 去重
                    unique_family_ids = set(r.family_id for r in relations)
                    return len(unique_family_ids)
                
                # 在focus模式下，实现多跳逻辑
                # 关键：从最早版本到当前版本的所有 absolute_id 关联的关系边
                if focus_family_id and focus_absolute_id and hops > 0:
                    # ========== 第一步：收集所有边和节点，确定最终要显示的图 ==========
                    # 存储最终确定要显示的边信息
                    final_edge_candidates = []  # 存储边的完整信息
                    graph_edges_for_bfs = []  # 仅用于 BFS 的边列表 (entity1_id, entity2_id)
                    graph_nodes = set()  # 存储所有节点
                    graph_nodes.add(focus_family_id)
                    
                    # 递归获取多跳的实体和关系
                    current_level_entities = {focus_family_id: focus_absolute_id}
                    processed_family_ids = set()

                    for current_hop in range(1, hops + 1):
                        next_level_entities = {}

                        for family_id, max_abs_id in current_level_entities.items():
                            if family_id in processed_family_ids:
                                continue
                            processed_family_ids.add(family_id)

                            # 获取该实体从最早版本到当前版本的所有 absolute_id
                            entity_abs_ids = self.storage.get_entity_absolute_ids_up_to_version(
                                family_id, max_abs_id
                            )
                            
                            if not entity_abs_ids:
                                continue
                            
                            # 获取这些 absolute_id 关联的所有关系边
                            entity_relations = self.storage.get_relations_by_entity_absolute_ids(
                                entity_abs_ids, 
                                limit=None
                            )
                            
                            # 收集关系边和对应的另一端实体信息，用于排序
                            relation_candidates = []
                            
                            for relation in entity_relations:
                                entity1 = self.storage.get_entity_by_absolute_id(relation.entity1_absolute_id)
                                entity2 = self.storage.get_entity_by_absolute_id(relation.entity2_absolute_id)

                                if entity1 and entity2:
                                    entity1_fid = entity1.family_id
                                    entity2_fid = entity2.family_id

                                    normalized_pair = LLMClient._normalize_entity_pair(entity1_fid, entity2_fid)
                                    normalized_entity1_id = normalized_pair[0]
                                    normalized_entity2_id = normalized_pair[1]

                                    edge_key = (normalized_entity1_id, normalized_entity2_id, relation.absolute_id)
                                    if edge_key not in edges_seen:
                                        other_entity = entity2 if relation.entity1_absolute_id in entity_abs_ids else entity1
                                        other_entity_fid = other_entity.family_id
                                        other_entity_abs_id = other_entity.absolute_id

                                        other_entity_edge_count = count_entity_relations(other_entity_fid, other_entity_abs_id)

                                        relation_candidates.append({
                                            'relation': relation,
                                            'entity1': entity1,
                                            'entity2': entity2,
                                            'normalized_entity1_id': normalized_entity1_id,
                                            'normalized_entity2_id': normalized_entity2_id,
                                            'edge_key': edge_key,
                                            'other_entity': other_entity,
                                            'other_entity_fid': other_entity_fid,
                                            'other_entity_abs_id': other_entity_abs_id,
                                            'other_entity_edge_count': other_entity_edge_count
                                        })
                            
                            # 按照另一端实体的边数从多到少排序
                            relation_candidates.sort(key=lambda x: x['other_entity_edge_count'], reverse=True)
                            
                            # 应用 limit_edges_per_entity 限制
                            if limit_edges_per_entity:
                                relation_candidates = relation_candidates[:limit_edges_per_entity]
                            
                            # 将选中的边加入最终结果
                            for candidate in relation_candidates:
                                edges_seen.add(candidate['edge_key'])
                                final_edge_candidates.append(candidate)
                                
                                # 记录图结构用于 BFS
                                graph_edges_for_bfs.append((
                                    candidate['normalized_entity1_id'], 
                                    candidate['normalized_entity2_id']
                                ))
                                graph_nodes.add(candidate['normalized_entity1_id'])
                                graph_nodes.add(candidate['normalized_entity2_id'])
                                
                                # 更新实体名称映射
                                entity1 = candidate['entity1']
                                entity2 = candidate['entity2']
                                normalized_entity1_id = candidate['normalized_entity1_id']
                                normalized_entity2_id = candidate['normalized_entity2_id']
                                
                                if normalized_entity1_id not in family_id_to_name:
                                    family_id_to_name[normalized_entity1_id] = entity1.name
                                    family_id_to_absolute_id[normalized_entity1_id] = entity1.absolute_id
                                    all_related_family_ids.add(normalized_entity1_id)
                                
                                if normalized_entity2_id not in family_id_to_name:
                                    family_id_to_name[normalized_entity2_id] = entity2.name
                                    family_id_to_absolute_id[normalized_entity2_id] = entity2.absolute_id
                                    all_related_family_ids.add(normalized_entity2_id)
                                
                                # 下一跳
                                other_entity = candidate['other_entity']
                                other_entity_fid = candidate['other_entity_fid']
                                other_entity_abs_id = candidate['other_entity_abs_id']

                                if current_hop < hops and other_entity_fid not in processed_family_ids:
                                    if other_entity_fid in next_level_entities:
                                        existing_abs_id = next_level_entities[other_entity_fid]
                                        existing_entity = self.storage.get_entity_by_absolute_id(existing_abs_id)
                                        if existing_entity and other_entity.event_time > existing_entity.event_time:
                                            next_level_entities[other_entity_fid] = other_entity_abs_id
                                    else:
                                        next_level_entities[other_entity_fid] = other_entity_abs_id
                        
                        current_level_entities = next_level_entities
                        if not current_level_entities:
                            break
                    
                    # ========== 第二步：基于最终确定的图计算最短路径 ==========
                    def bfs_shortest_paths(start_node, edges_list):
                        """使用BFS计算从起始节点到所有节点的最短路径长度"""
                        # 构建邻接表
                        graph = {}
                        for u, v in edges_list:
                            if u not in graph:
                                graph[u] = []
                            if v not in graph:
                                graph[v] = []
                            graph[u].append(v)
                            graph[v].append(u)
                        
                        # BFS (使用 deque 实现 O(1) popleft)
                        distances = {start_node: 0}
                        queue = deque([start_node])

                        while queue:
                            current = queue.popleft()
                            if current not in graph:
                                continue
                            
                            for neighbor in graph[current]:
                                if neighbor not in distances:
                                    distances[neighbor] = distances[current] + 1
                                    queue.append(neighbor)
                        
                        return distances
                    
                    # 计算最短路径
                    shortest_paths = bfs_shortest_paths(focus_family_id, graph_edges_for_bfs)
                    
                    # 根据最短路径长度设置跳数层级
                    for fid in graph_nodes:
                        if fid in shortest_paths:
                            family_id_to_hop_level[fid] = shortest_paths[fid]
                        else:
                            family_id_to_hop_level[fid] = 999
                    
                    # ========== 第三步：生成最终的边列表 ==========
                    for candidate in final_edge_candidates:
                        relation = candidate['relation']
                        normalized_entity1_id = candidate['normalized_entity1_id']
                        normalized_entity2_id = candidate['normalized_entity2_id']
                        
                        edge_label = ""
                        if relation.content:
                            edge_label = relation.content[:30] + "..." if len(relation.content) > 30 else relation.content
                        
                        edges.append({
                            'from': normalized_entity1_id,
                            'to': normalized_entity2_id,
                            'label': edge_label,
                            'title': relation.content,
                            'content': relation.content,
                            'event_time': relation.event_time.isoformat() if relation.event_time else None,
                                    'processed_time': relation.processed_time.isoformat() if relation.processed_time else None,
                            'family_id': relation.family_id,
                            'absolute_id': relation.absolute_id,
                            'color': '#888888',
                            'width': 2,
                            'arrows': ''
                        })
                else:
                    # 非focus模式或hops=0，使用原来的单层逻辑，但也要按另一端实体的边数排序
                    for entity in entities:
                        max_version_absolute_id = focus_absolute_id if (focus_family_id and focus_family_id == entity.family_id) else None
                        effective_time_point = None if max_version_absolute_id else time_point
                        
                        # 先获取所有关系边，不限制数量，用于排序
                        entity_relations = self.storage.get_entity_relations_by_family_id(
                            entity.family_id, 
                            limit=None, 
                            time_point=effective_time_point,
                            max_version_absolute_id=max_version_absolute_id
                        )
                        
                        # 收集关系边和对应的另一端实体信息，用于排序
                        relation_candidates = []
                        
                        for relation in entity_relations:
                            entity1_temp = self.storage.get_entity_by_absolute_id(relation.entity1_absolute_id)
                            entity2_temp = self.storage.get_entity_by_absolute_id(relation.entity2_absolute_id)
                            
                            if entity1_temp and entity2_temp:
                                effective_time_point = focus_time_point if focus_family_id else time_point
                                if effective_time_point:
                                    entity1 = self.storage.get_entity_version_at_time(entity1_temp.family_id, effective_time_point)
                                    entity2 = self.storage.get_entity_version_at_time(entity2_temp.family_id, effective_time_point)
                                else:
                                    entity1 = entity1_temp
                                    entity2 = entity2_temp
                                
                                if entity1 and entity2:
                                    entity1_fid = entity1.family_id
                                    entity2_fid = entity2.family_id

                                    # 标准化实体对（按字母顺序排序，使关系无向化）
                                    normalized_pair = LLMClient._normalize_entity_pair(entity1_fid, entity2_fid)
                                    normalized_entity1_id = normalized_pair[0]
                                    normalized_entity2_id = normalized_pair[1]

                                    edge_key = (normalized_entity1_id, normalized_entity2_id, relation.family_id)
                                    if edge_key not in edges_seen:
                                        # 判断哪个是"另一端"的实体（相对于当前entity）
                                        other_entity = entity2 if entity1_fid == entity.family_id else entity1
                                        other_entity_fid = other_entity.family_id

                                        # 统计另一端实体拥有的关系边数量（去重后）
                                        other_entity_edge_count = count_entity_relations(other_entity_fid)
                                        
                                        relation_candidates.append({
                                            'relation': relation,
                                            'entity1': entity1,
                                            'entity2': entity2,
                                            'normalized_entity1_id': normalized_entity1_id,
                                            'normalized_entity2_id': normalized_entity2_id,
                                            'edge_key': edge_key,
                                            'other_entity_edge_count': other_entity_edge_count
                                        })
                        
                        # 按照另一端实体的边数从多到少排序
                        relation_candidates.sort(key=lambda x: x['other_entity_edge_count'], reverse=True)
                        
                        # 应用 limit_edges_per_entity 限制
                        if limit_edges_per_entity:
                            relation_candidates = relation_candidates[:limit_edges_per_entity]
                        
                        # 按排序后的顺序添加关系边
                        for candidate in relation_candidates:
                            relation = candidate['relation']
                            entity1 = candidate['entity1']
                            entity2 = candidate['entity2']
                            normalized_entity1_id = candidate['normalized_entity1_id']
                            normalized_entity2_id = candidate['normalized_entity2_id']
                            edge_key = candidate['edge_key']
                            
                            edges_seen.add(edge_key)
                            
                            if entity1.absolute_id not in entity_absolute_ids:
                                entity_absolute_ids.add(entity1.absolute_id)
                                all_related_family_ids.add(normalized_entity1_id)
                                family_id_to_name[normalized_entity1_id] = entity1.name
                                family_id_to_absolute_id[normalized_entity1_id] = entity1.absolute_id

                            if entity2.absolute_id not in entity_absolute_ids:
                                entity_absolute_ids.add(entity2.absolute_id)
                                all_related_family_ids.add(normalized_entity2_id)
                                family_id_to_name[normalized_entity2_id] = entity2.name
                                family_id_to_absolute_id[normalized_entity2_id] = entity2.absolute_id

                            # 添加边
                            if entity1.absolute_id in entity_absolute_ids or entity2.absolute_id in entity_absolute_ids:
                                edge_label = ""
                                if relation.content:
                                    edge_label = relation.content[:30] + "..." if len(relation.content) > 30 else relation.content
                                
                                edges.append({
                                    'from': normalized_entity1_id,
                                    'to': normalized_entity2_id,
                                    'label': edge_label,
                                    'title': relation.content,
                                    'content': relation.content,
                                    'event_time': relation.event_time.isoformat() if relation.event_time else None,
                                    'processed_time': relation.processed_time.isoformat() if relation.processed_time else None,
                                    'family_id': relation.family_id,
                                    'absolute_id': relation.absolute_id,
                                    'color': '#888888',
                                    'width': 2,
                                    'arrows': ''
                                })
                
                # 添加关联实体节点（如果还没有添加）
                for fid in all_related_family_ids:
                    if fid not in [node['id'] for node in nodes]:
                        # 在 focus 模式下，直接使用记录的 absolute_id 获取实体版本
                        # 这确保我们显示的是关系边直接引用的实体版本
                        absolute_id = family_id_to_absolute_id.get(fid)
                        if absolute_id:
                            related_entity = self.storage.get_entity_by_absolute_id(absolute_id)
                        else:
                            # 回退：如果没有记录 absolute_id，使用时间点
                            effective_time_point = focus_time_point if focus_family_id else time_point
                            if effective_time_point:
                                related_entity = self.storage.get_entity_version_at_time(fid, effective_time_point)
                            else:
                                related_entity = None

                        if related_entity:
                            versions = self.storage.get_entity_versions(related_entity.family_id)
                            version_count = len(versions)

                            # 在focus模式下，显示该实体版本的索引
                            if focus_family_id and absolute_id:
                                versions_sorted = sorted(
                                    versions,
                                    key=lambda v: self.storage._normalize_datetime_for_compare(v.processed_time)
                                )
                                current_version_index = None
                                for idx, v in enumerate(versions_sorted, 1):
                                    if v.absolute_id == related_entity.absolute_id:
                                        current_version_index = idx
                                        break

                                if current_version_index:
                                    label = f"{related_entity.name} ({current_version_index}/{version_count}版本)" if version_count > 1 else related_entity.name
                                else:
                                    label = f"{related_entity.name} ({version_count}版本)" if version_count > 1 else related_entity.name
                            else:
                                label = f"{related_entity.name} ({version_count}版本)" if version_count > 1 else related_entity.name

                            # 根据跳数层级设置颜色
                            hop_level = family_id_to_hop_level.get(fid, 0)
                            node_color = get_hop_color(hop_level)
                            
                            nodes.append({
                                'id': related_entity.family_id,
                                'family_id': related_entity.family_id,
                                'absolute_id': related_entity.absolute_id,
                                'label': label,
                                'title': f"{related_entity.name}\n\n{related_entity.content[:100]}..." if len(related_entity.content) > 100 else f"{related_entity.name}\n\n{related_entity.content}",
                                'content': related_entity.content,
                                'event_time': related_entity.event_time.isoformat() if related_entity.event_time else None,
                                'processed_time': related_entity.processed_time.isoformat() if related_entity.processed_time else None,
                                'version_count': version_count,
                                'color': node_color,  # 根据跳数层级设置颜色
                                'shape': 'dot',
                                'size': 20,
                                'font': {'color': 'white'}
                            })
                
                return jsonify({
                    'success': True,
                    'nodes': nodes,
                    'edges': edges,
                    'stats': {
                        'total_entities': len(nodes),
                        'total_relations': len(edges),
                        'initial_entities': len(entities),
                        'related_entities': len(all_related_family_ids)
                    }
                })
            except Exception as e:
                import traceback
                error_traceback = traceback.format_exc()
                print(f"❌ [API错误] /api/graphs/data 发生异常:")
                print(f"   错误类型: {type(e).__name__}")
                print(f"   错误消息: {str(e)}")
                print(f"   错误堆栈:\n{error_traceback}")
                return jsonify({
                    'success': False,
                    'error': str(e),
                    'error_type': type(e).__name__
                }), 500
        
        @self.app.route('/api/graphs/config')
        def get_config():
            """获取当前配置信息 API"""
            try:
                return jsonify({
                    'success': True,
                    'storage_path': self._current_storage_path
                })
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
        
        @self.app.route('/api/graphs/stats')
        def get_stats():
            """获取统计信息 API"""
            try:
                entities = self.storage.get_all_entities()
                relations = self.storage.get_all_relations()
                
                return jsonify({
                    'success': True,
                    'stats': {
                        'total_entities': len(entities),
                        'total_relations': len(relations)
                    }
                })
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
        
        @self.app.route('/api/graphs/search', methods=['POST'])
        def search_graph():
            """搜索图谱 API
            
            接收JSON数据:
            - query: 自然语言查询文本
            - max_results: 返回的最大实体+关系数量（默认10）
            - storage_path: 图谱存储路径（可选），如果提供且与当前路径不同，会切换存储路径
            
            搜索逻辑:
            1. 将查询转成embedding向量
            2. 与所有实体embedding和关系边embedding进行相似度匹配
            3. 选取前n条实体+关系的集合（同ID去重，选取最新版本）
            4. 以这些实体和关系为起点，只显示1跳距离的数据：
               - 实体：找相关联的边以及连接的另一个实体
               - 关系边：找对应的两个实体
            """
            try:
                from flask import request
                
                data = request.get_json()
                if not data:
                    return jsonify({
                        'success': False,
                        'error': '请求数据格式错误，需要JSON格式'
                    }), 400
                
                query = data.get('query', '').strip()
                if not query:
                    return jsonify({
                        'success': False,
                        'error': '查询文本不能为空'
                    }), 400
                
                max_results = data.get('max_results', 10)
                storage_path_param = data.get('storage_path', '').strip() if data.get('storage_path') else None
                
                # 如果提供了存储路径参数，且与当前路径不同，则切换存储路径
                if storage_path_param:
                    try:
                        self._switch_storage_path(storage_path_param)
                    except Exception as e:
                        return jsonify({
                            'success': False,
                            'error': f'切换存储路径失败: {str(e)}'
                        }), 400
                
                # 使用embedding进行语义搜索
                # 设置较低的阈值以支持更灵活的匹配
                # 如果embedding不可用，会自动回退到文本相似度搜索
                try:
                    print(f"[搜索API] 开始搜索，查询: {query}, 最大结果数: {max_results}")
                    print(f"[搜索API] Embedding客户端可用: {self.embedding_client.is_available() if self.embedding_client else False}")
                    
                    # 同时搜索实体和关系
                    matched_entities = self.storage.search_entities_by_similarity(
                        query_name=query,
                        query_content=query,
                        threshold=0.3,
                        max_results=max_results,  # 先搜索max_results个实体
                        content_snippet_length=100
                    )
                    
                    matched_relations = self.storage.search_relations_by_similarity(
                        query_text=query,
                        threshold=0.3,
                        max_results=max_results  # 先搜索max_results个关系
                    )
                    
                    print(f"[搜索API] 搜索完成，找到 {len(matched_entities)} 个匹配实体，{len(matched_relations)} 个匹配关系")
                    
                    # 合并实体和关系，去重（同ID只保留最新版本）
                    # 收集匹配的实体ID和关系ID
                    matched_entity_absolute_ids = {entity.absolute_id for entity in matched_entities}
                    matched_relation_absolute_ids = {relation.absolute_id for relation in matched_relations}
                    
                    # 收集匹配实体的family_id（用于节点显示）
                    matched_family_ids = {entity.family_id for entity in matched_entities}
                    matched_relation_ids = {relation.family_id for relation in matched_relations}
                    
                except Exception as e:
                    import traceback
                    print(f"[搜索API] 搜索错误: {str(e)}")
                    print(traceback.format_exc())
                    return jsonify({
                        'success': False,
                        'error': f'搜索过程中发生错误: {str(e)}'
                    }), 500
                
                if not matched_entities and not matched_relations:
                    print("[搜索API] 未找到匹配的实体或关系")
                    return jsonify({
                        'success': True,
                        'nodes': [],
                        'edges': [],
                        'stats': {
                            'total_entities': 0,
                            'total_relations': 0,
                            'matched_entities': 0,
                            'matched_relations': 0
                        },
                        'query': query
                    })
                
                # 收集所有需要显示的实体（1跳距离）
                # 1. 匹配的实体本身
                entity_absolute_ids = set(matched_entity_absolute_ids)
                family_id_to_name = {}
                family_id_to_absolute_id = {}
                
                # 2. 匹配关系对应的两个实体
                for relation in matched_relations:
                    entity1 = self.storage.get_entity_by_absolute_id(relation.entity1_absolute_id)
                    entity2 = self.storage.get_entity_by_absolute_id(relation.entity2_absolute_id)
                    if entity1:
                        entity_absolute_ids.add(entity1.absolute_id)
                        family_id_to_name[entity1.family_id] = entity1.name
                        family_id_to_absolute_id[entity1.family_id] = entity1.absolute_id
                    if entity2:
                        entity_absolute_ids.add(entity2.absolute_id)
                        family_id_to_name[entity2.family_id] = entity2.name
                        family_id_to_absolute_id[entity2.family_id] = entity2.absolute_id
                
                # 3. 匹配实体相关联的边以及连接的另一个实体（1跳距离）
                relation_absolute_ids = set(matched_relation_absolute_ids)
                edges_seen = set()  # 用于去重，使用 (from_id, to_id, family_id) 作为唯一标识

                for entity in matched_entities:
                    # 获取该实体的所有关系边（1跳距离，不限制数量）
                    entity_relations = self.storage.get_entity_relations(entity.absolute_id, limit=None)

                    for relation in entity_relations:
                        relation_absolute_ids.add(relation.absolute_id)
                        
                        # 通过绝对ID获取实体
                        entity1 = self.storage.get_entity_by_absolute_id(relation.entity1_absolute_id)
                        entity2 = self.storage.get_entity_by_absolute_id(relation.entity2_absolute_id)
                        
                        if entity1 and entity2:
                            entity1_fid = entity1.family_id
                            entity2_fid = entity2.family_id

                            # 标准化实体对（按字母顺序排序，使关系无向化）
                            normalized_pair = LLMClient._normalize_entity_pair(entity1_fid, entity2_fid)
                            normalized_entity1_id = normalized_pair[0]
                            normalized_entity2_id = normalized_pair[1]
                            
                            # 创建唯一标识符，避免重复添加同一条边（使用标准化后的实体对）
                            edge_key = (normalized_entity1_id, normalized_entity2_id, relation.family_id)
                            if edge_key in edges_seen:
                                continue
                            edges_seen.add(edge_key)
                            
                            # 添加连接的实体（1跳距离）
                            if entity1.absolute_id not in entity_absolute_ids:
                                entity_absolute_ids.add(entity1.absolute_id)
                                family_id_to_name[normalized_entity1_id] = entity1.name
                                family_id_to_absolute_id[normalized_entity1_id] = entity1.absolute_id

                            if entity2.absolute_id not in entity_absolute_ids:
                                entity_absolute_ids.add(entity2.absolute_id)
                                family_id_to_name[normalized_entity2_id] = entity2.name
                                family_id_to_absolute_id[normalized_entity2_id] = entity2.absolute_id
                
                # 构建节点数据（使用family_id去重，每个family_id只保留一个节点）
                nodes = []
                seen_family_ids = set()  # 用于去重，确保每个family_id只添加一次
                family_id_to_latest_absolute_id = {}  # 记录每个family_id对应的最新absolute_id

                # 首先收集所有family_id及其对应的最新absolute_id
                for entity_abs_id in entity_absolute_ids:
                    entity = self.storage.get_entity_by_absolute_id(entity_abs_id)
                    if entity:
                        fid = entity.family_id
                        # 如果这个family_id还没有记录，或者当前版本更新，则更新记录
                        if fid not in family_id_to_latest_absolute_id:
                            family_id_to_latest_absolute_id[fid] = entity_abs_id
                        else:
                            # 比较时间，保留更新的版本
                            existing_entity = self.storage.get_entity_by_absolute_id(family_id_to_latest_absolute_id[fid])
                            if existing_entity and entity.event_time > existing_entity.event_time:
                                family_id_to_latest_absolute_id[fid] = entity_abs_id

                # 然后为每个唯一的family_id创建一个节点
                for fid, entity_abs_id in family_id_to_latest_absolute_id.items():
                    entity = self.storage.get_entity_by_absolute_id(entity_abs_id)
                    if entity:
                        # 判断是否为匹配的实体
                        is_matched = entity.family_id in matched_family_ids
                        
                        # 获取版本数量
                        versions = self.storage.get_entity_versions(entity.family_id)
                        version_count = len(versions)
                        
                        # 在标签中显示版本数量
                        label = f"{entity.name} ({version_count}版本)" if version_count > 1 else entity.name
                        
                        nodes.append({
                            'id': entity.family_id,
                            'family_id': entity.family_id,
                            'absolute_id': entity.absolute_id,
                            'label': label,
                            'title': f"{entity.name}\n\n{entity.content[:100]}..." if len(entity.content) > 100 else f"{entity.name}\n\n{entity.content}",
                            'content': entity.content,
                            'event_time': entity.event_time.isoformat() if entity.event_time else None,
                            'processed_time': entity.processed_time.isoformat() if entity.processed_time else None,
                            'version_count': version_count,
                            'color': '#FF6B6B' if is_matched else '#97C2FC',  # 匹配的实体用红色，其他用蓝色
                            'shape': 'dot',
                            'size': 25 if is_matched else 20,
                            'font': {'color': 'white'}  # 所有搜索结果中的节点字体都用白色
                        })
                
                # 构建边数据
                edges = []
                edges_seen = set()  # 重新初始化，用于最终去重
                
                # 1. 添加匹配关系对应的边
                for relation in matched_relations:
                    entity1 = self.storage.get_entity_by_absolute_id(relation.entity1_absolute_id)
                    entity2 = self.storage.get_entity_by_absolute_id(relation.entity2_absolute_id)

                    if entity1 and entity2:
                        entity1_fid = entity1.family_id
                        entity2_fid = entity2.family_id

                        edge_key = (entity1_fid, entity2_fid, relation.family_id)
                        if edge_key not in edges_seen:
                            edges_seen.add(edge_key)

                            edge_label = ""
                            if relation.content:
                                edge_label = relation.content[:30] + "..." if len(relation.content) > 30 else relation.content

                            edges.append({
                                'from': entity1_fid,
                                'to': entity2_fid,
                                'label': edge_label,
                                'title': relation.content,
                                'content': relation.content,
                                'event_time': relation.event_time.isoformat() if relation.event_time else None,
                                    'processed_time': relation.processed_time.isoformat() if relation.processed_time else None,
                                'family_id': relation.family_id,
                                'absolute_id': relation.absolute_id,
                                'color': '#FF6B6B',  # 匹配的关系边用红色，和匹配实体颜色一致
                                'width': 3,
                                'arrows': ''
                            })
                
                # 2. 添加匹配实体相关联的边（1跳距离）
                for entity in matched_entities:
                    entity_relations = self.storage.get_entity_relations(entity.absolute_id, limit=None)

                    for relation in entity_relations:
                        entity1 = self.storage.get_entity_by_absolute_id(relation.entity1_absolute_id)
                        entity2 = self.storage.get_entity_by_absolute_id(relation.entity2_absolute_id)
                        
                        if entity1 and entity2:
                            entity1_fid = entity1.family_id
                            entity2_fid = entity2.family_id

                            edge_key = (entity1_fid, entity2_fid, relation.family_id)
                            if edge_key not in edges_seen:
                                edges_seen.add(edge_key)

                                # 判断是否为匹配的关系
                                is_matched = relation.family_id in matched_relation_ids

                                edge_label = ""
                                if relation.content:
                                    edge_label = relation.content[:30] + "..." if len(relation.content) > 30 else relation.content

                                edges.append({
                                    'from': entity1_fid,
                                    'to': entity2_fid,
                                    'label': edge_label,
                                    'title': relation.content,
                                    'content': relation.content,
                                    'event_time': relation.event_time.isoformat() if relation.event_time else None,
                                    'processed_time': relation.processed_time.isoformat() if relation.processed_time else None,
                                    'family_id': relation.family_id,
                                    'absolute_id': relation.absolute_id,
                                    'color': '#FF6B6B' if is_matched else '#97C2FC',  # 匹配的关系边用红色（和匹配实体颜色一致），其他用蓝色
                                    'width': 3 if is_matched else 2,
                                    'arrows': ''
                                })
                
                return jsonify({
                    'success': True,
                    'nodes': nodes,
                    'edges': edges,
                    'stats': {
                        'total_entities': len(nodes),
                        'total_relations': len(edges),
                        'matched_entities': len(matched_entities),
                        'matched_relations': len(matched_relations)
                    },
                    'query': query
                })
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
        
        @self.app.route('/api/entities/<family_id>/versions')
        def get_entity_versions(family_id):
            """获取实体的所有版本列表"""
            try:
                versions = self.storage.get_entity_versions(family_id)

                if not versions:
                    return jsonify({
                        'success': False,
                        'error': f'未找到实体 {family_id} 的版本'
                    }), 404
                
                versions_data = []
                for i, entity in enumerate(versions, 1):
                    versions_data.append({
                        'index': i,
                        'total': len(versions),
                        'absolute_id': entity.absolute_id,
                        'family_id': entity.family_id,
                        'name': entity.name,
                        'content': entity.content,
                        'event_time': entity.event_time.isoformat() if entity.event_time else None,
                            'processed_time': entity.processed_time.isoformat() if entity.processed_time else None,
                        'episode_id': entity.episode_id
                    })
                
                return jsonify({
                    'success': True,
                    'family_id': family_id,
                    'versions': versions_data
                })
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500

        @self.app.route('/api/entities/<family_id>/versions/<absolute_id>')
        def get_entity_version(family_id, absolute_id):
            """获取实体的特定版本"""
            try:
                entity = self.storage.get_entity_by_absolute_id(absolute_id)
                
                if not entity:
                    return jsonify({
                        'success': False,
                        'error': f'未找到实体版本 {absolute_id}'
                    }), 404
                
                if entity.family_id != family_id:
                    return jsonify({
                        'success': False,
                        'error': f'实体ID不匹配'
                    }), 400

                # 获取版本索引
                versions = self.storage.get_entity_versions(family_id)
                version_index = next((i for i, e in enumerate(versions, 1) if e.absolute_id == absolute_id), None)
                
                # 获取embedding前4个值
                embedding_preview = self.storage.get_entity_embedding_preview(absolute_id, 4)
                
                # 获取episode对应的md文档内容和json中的原文内容
                episode_content = None  # md文档内容
                episode_text = None  # json中的原文内容
                source_document = None  # 文档名称（初始化，避免未绑定）
                doc_name = None  # 文档名称
                if entity.episode_id:
                    # 获取md文档内容（Episode的content字段）
                    episode = self.storage.load_episode(entity.episode_id)
                    if episode:
                        episode_content = episode.content
                        source_document = getattr(episode, 'source_document', None) or getattr(episode, 'doc_name', None)  # 从Episode对象获取文档名称
                    # 获取json中的原文内容
                    episode_text = self.storage.get_episode_text(entity.episode_id)

                return jsonify({
                    'success': True,
                    'entity': {
                        'absolute_id': entity.absolute_id,
                        'family_id': entity.family_id,
                        'name': entity.name,
                        'content': entity.content,
                        'event_time': entity.event_time.isoformat() if entity.event_time else None,
                            'processed_time': entity.processed_time.isoformat() if entity.processed_time else None,
                        'episode_id': entity.episode_id,
                        'episode_content': episode_content,  # md文档内容
                        'episode_text': episode_text,  # json中的原文内容
                        'source_document': source_document,  # 文档名称（新字段）
                        'doc_name': source_document,  # 文档名称（兼容旧字段）
                        'version_index': version_index,
                        'total_versions': len(versions),
                        'embedding_preview': embedding_preview  # embedding前4个值
                    }
                })
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
        
        @self.app.route('/api/relations/<family_id>/versions')
        def get_relation_versions(family_id):
            """获取关系的所有版本列表"""
            try:
                versions = self.storage.get_relation_versions(family_id)

                if not versions:
                    return jsonify({
                        'success': False,
                        'error': f'未找到关系 {family_id} 的版本'
                    }), 404
                
                versions_data = []
                for i, relation in enumerate(versions, 1):
                    # 获取实体信息
                    entity1 = self.storage.get_entity_by_absolute_id(relation.entity1_absolute_id)
                    entity2 = self.storage.get_entity_by_absolute_id(relation.entity2_absolute_id)
                    
                    versions_data.append({
                        'index': i,
                        'total': len(versions),
                        'absolute_id': relation.absolute_id,
                        'family_id': relation.family_id,
                        'content': relation.content,
                        'event_time': relation.event_time.isoformat() if relation.event_time else None,
                                    'processed_time': relation.processed_time.isoformat() if relation.processed_time else None,
                        'episode_id': relation.episode_id,
                        'entity1_absolute_id': relation.entity1_absolute_id,
                        'entity2_absolute_id': relation.entity2_absolute_id,
                        'entity1_id': entity1.family_id if entity1 else None,
                        'entity2_id': entity2.family_id if entity2 else None
                    })
                
                return jsonify({
                    'success': True,
                    'family_id': family_id,
                    'versions': versions_data
                })
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
        
        @self.app.route('/api/graphs/snapshot', methods=['POST'])
        def get_graph_snapshot():
            """根据时间点获取图谱快照

            接收JSON数据:
            - entity_versions: {family_id: absolute_id} 字典，指定要显示的实体版本
            - relation_versions: {family_id: absolute_id} 字典，指定要显示的关系版本
            - time_point: ISO格式的时间点（可选，用于筛选）
            """
            try:
                from flask import request
                from datetime import datetime
                
                data = request.get_json()
                if not data:
                    return jsonify({
                        'success': False,
                        'error': '请求数据格式错误，需要JSON格式'
                    }), 400
                
                entity_versions = data.get('entity_versions', {})  # {family_id: absolute_id}
                relation_versions = data.get('relation_versions', {})  # {family_id: absolute_id}
                time_point_str = data.get('time_point')

                time_point = None
                if time_point_str:
                    try:
                        time_point = datetime.fromisoformat(time_point_str)
                    except (ValueError, TypeError):
                        pass

                # 获取指定版本的实体信息
                nodes_data = []
                for family_id, absolute_id in entity_versions.items():
                    entity = self.storage.get_entity_by_absolute_id(absolute_id)
                    if entity:
                        versions = self.storage.get_entity_versions(family_id)
                        version_count = len(versions)
                        label = f"{entity.name} ({version_count}版本)" if version_count > 1 else entity.name

                        nodes_data.append({
                            'id': family_id,
                            'family_id': family_id,
                            'absolute_id': absolute_id,
                            'label': label,
                            'name': entity.name,
                            'content': entity.content,
                            'event_time': entity.event_time.isoformat() if entity.event_time else None,
                            'processed_time': entity.processed_time.isoformat() if entity.processed_time else None,
                            'version_count': version_count
                        })

                # 获取指定版本的关系信息
                edges_data = []
                for family_id, absolute_id in relation_versions.items():
                    versions = self.storage.get_relation_versions(family_id)
                    relation = next((r for r in versions if r.absolute_id == absolute_id), None)
                    if relation:
                        entity1 = self.storage.get_entity_by_absolute_id(relation.entity1_absolute_id)
                        entity2 = self.storage.get_entity_by_absolute_id(relation.entity2_absolute_id)
                        if entity1 and entity2:
                            edges_data.append({
                                'family_id': family_id,
                                'absolute_id': absolute_id,
                                'from': entity1.family_id,
                                'to': entity2.family_id,
                                'content': relation.content,
                                'event_time': relation.event_time.isoformat() if relation.event_time else None,
                                'processed_time': relation.processed_time.isoformat() if relation.processed_time else None
                            })
                
                return jsonify({
                    'success': True,
                    'nodes': nodes_data,
                    'edges': edges_data
                })
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
        
        @self.app.route('/api/relations/<family_id>/versions/<absolute_id>')
        def get_relation_version(family_id, absolute_id):
            """获取关系的特定版本"""
            try:
                # 获取所有版本来找到特定版本
                versions = self.storage.get_relation_versions(family_id)
                relation = next((r for r in versions if r.absolute_id == absolute_id), None)

                if not relation:
                    return jsonify({
                        'success': False,
                        'error': f'未找到关系版本 {absolute_id}'
                    }), 404

                if relation.family_id != family_id:
                    return jsonify({
                        'success': False,
                        'error': f'关系ID不匹配'
                    }), 400
                
                # 获取实体信息
                entity1 = self.storage.get_entity_by_absolute_id(relation.entity1_absolute_id)
                entity2 = self.storage.get_entity_by_absolute_id(relation.entity2_absolute_id)
                
                # 获取版本索引
                version_index = next((i for i, r in enumerate(versions, 1) if r.absolute_id == absolute_id), None)
                
                # 获取embedding前5个值
                embedding_preview = self.storage.get_relation_embedding_preview(absolute_id, 4)
                
                return jsonify({
                    'success': True,
                    'relation': {
                        'absolute_id': relation.absolute_id,
                        'family_id': relation.family_id,
                        'content': relation.content,
                        'event_time': relation.event_time.isoformat() if relation.event_time else None,
                                    'processed_time': relation.processed_time.isoformat() if relation.processed_time else None,
                        'episode_id': relation.episode_id,
                        'entity1_absolute_id': relation.entity1_absolute_id,
                        'entity2_absolute_id': relation.entity2_absolute_id,
                        'entity1_id': entity1.family_id if entity1 else None,
                        'entity2_id': entity2.family_id if entity2 else None,
                        'entity1_name': entity1.name if entity1 else None,
                        'entity2_name': entity2.name if entity2 else None,
                        'version_index': version_index,
                        'total_versions': len(versions),
                        'embedding_preview': embedding_preview  # 添加embedding前5个值
                    }
                })
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
    
    def run(self, debug: bool = False, host: str = '0.0.0.0'):
        """
        启动 Web 服务器
        
        Args:
            debug: 是否开启调试模式
            host: 监听地址
        """
        # 获取embedding模型信息
        embedding_info = "未配置"
        if self.embedding_client.model:
            if self.embedding_client.model_path:
                embedding_info = f"本地模型: {self.embedding_client.model_path}"
            elif self.embedding_client.model_name:
                embedding_info = f"HuggingFace: {self.embedding_client.model_name}"
            else:
                embedding_info = "默认模型: all-MiniLM-L6-v2"
        else:
            embedding_info = "未安装sentence-transformers（将使用文本相似度搜索）"
        
        print(f"""
╔════════════════════════════════════════╗
║   时序记忆图谱可视化 Web 服务            ║


🌐 Web服务器已启动
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
访问地址: http://localhost:{self.port}
API地址:  http://localhost:{self.port}/api/graphs/data

📦 Embedding模型: {embedding_info}
📁 存储路径: {self.storage_path}

提示:
  1. 在浏览器中打开 http://localhost:{self.port}
  2. 图谱会自动加载并显示
  3. 点击"刷新图谱"按钮手动更新
  4. 点击节点或边查看详细信息
  5. 使用搜索功能进行语义搜索

按 Ctrl+C 停止服务器
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        """)
        
        try:
            self.app.run(host=host, port=self.port, debug=debug)
        except KeyboardInterrupt:
            print("\n\n👋 服务器已停止")
        except Exception as e:
            print(f"\n❌ 错误: {e}")


def main():
    """主函数。支持 --config 与 service_api 共用 service_config.json。"""
    import argparse
    from server.config import load_config, resolve_embedding_model

    parser = argparse.ArgumentParser(description='时序记忆图谱可视化 Web 服务')
    parser.add_argument('--config', type=str, default=None,
                       help='配置文件路径（与 service_api 共用 service_config.json 时，将使用其中 storage_path 与 embedding）')
    parser.add_argument('--storage', type=str, default='./graph',
                       help='图谱基础目录 (默认: ./graph，未使用 --config 时生效)')
    parser.add_argument('--graph-id', type=str, default='default',
                       help='要可视化的图谱 ID (默认: default，对应 ./graph/default/)')
    parser.add_argument('--port', type=int, default=5000,
                       help='服务器端口 (默认: 5000，与 service_api 不同端口可同时运行)')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                       help='监听地址 (默认: 0.0.0.0)')
    parser.add_argument('--debug', action='store_true',
                       help='开启调试模式')
    parser.add_argument('--embedding-model-path', type=str, default=None,
                       help='本地 embedding 模型路径（未使用 --config 时生效）')
    parser.add_argument('--embedding-model-name', type=str, default=None,
                       help='HuggingFace embedding 模型名称（例如: all-MiniLM-L6-v2）')
    parser.add_argument('--embedding-device', type=str, default='cpu',
                       help='计算设备 (默认: cpu，可为 cuda 或 cuda:0)')
    parser.add_argument('--embedding-use-local', action='store_true', default=True,
                       help='优先使用本地模型（默认: True）')
    parser.add_argument('--embedding-use-hf', action='store_true', default=False,
                       help='优先使用 HuggingFace 模型（与 --embedding-use-local 互斥）')

    args = parser.parse_args()

    if args.config:
        if not Path(args.config).exists():
            print(f"错误：配置文件不存在: {args.config}")
            return 1
        config = load_config(args.config)
        base_path = config.get('storage_path', args.storage)
        emb_cfg = config.get('embedding') or {}
        emb_path, emb_name, emb_use_local = resolve_embedding_model(emb_cfg)
        embedding_device = emb_cfg.get('device') or 'cpu'
        embedding_model_path = emb_path
        embedding_model_name = emb_name
        embedding_use_local = emb_use_local
    else:
        base_path = args.storage
        embedding_model_path = args.embedding_model_path
        embedding_model_name = args.embedding_model_name
        embedding_device = args.embedding_device
        embedding_use_local = args.embedding_use_local and not args.embedding_use_hf

    storage_path = str(Path(base_path) / args.graph_id)

    if not Path(storage_path).exists():
        print(f"错误：图谱路径不存在: {storage_path}")
        print(f"  基础目录: {base_path}，graph_id: {args.graph_id}")
        return 1

    server = GraphWebServer(
        storage_path=storage_path,
        port=args.port,
        embedding_model_path=embedding_model_path,
        embedding_model_name=embedding_model_name,
        embedding_device=embedding_device,
        embedding_use_local=embedding_use_local
    )
    server.run(debug=args.debug, host=args.host)

    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
