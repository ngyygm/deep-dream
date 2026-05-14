"""
Extracted from web.py — Graph data and search route handlers for GraphWebServer.
"""
from __future__ import annotations

import logging
from collections import deque
from datetime import datetime
from typing import Set

from flask import jsonify, request

from core.utils import normalize_entity_pair

logger = logging.getLogger(__name__)


# -- Hop color helper --------------------------------------------------------

_HOP_COLORS = [
    '#4A90E2',  # hop 0 (focus): blue
    '#E67E22',  # hop 1: orange
    '#27AE60',  # hop 2: green
    '#9B59B6',  # hop 3: purple
    '#E74C3C',  # hop 4: red
    '#F39C12',  # hop 5: yellow
    '#1ABC9C',  # hop 6: cyan
    '#34495E',  # hop 7: dark gray
]


def _get_hop_color(hop_level: int) -> str:
    return _HOP_COLORS[hop_level % len(_HOP_COLORS)]


# -- BFS shortest path (for focus mode) --------------------------------------

def _bfs_shortest_paths(start_node, edges_list):
    """BFS compute shortest-path length from start_node to all nodes."""
    graph: dict = {}
    for u, v in edges_list:
        graph.setdefault(u, []).append(v)
        graph.setdefault(v, []).append(u)

    distances = {start_node: 0}
    queue = deque([start_node])

    while queue:
        current = queue.popleft()
        for neighbor in graph.get(current, []):
            if neighbor not in distances:
                distances[neighbor] = distances[current] + 1
                queue.append(neighbor)

    return distances


# -- Route registration ------------------------------------------------------

def register_graph_routes(server):
    """Register graph data and search routes on *server.app*."""

    # =====================================================================
    # /api/graphs/data  — main graph data endpoint
    # =====================================================================
    @server.app.route('/api/graphs/data')
    def get_graph_data():
        """Fetch graph data API.

        Supported params:
        - limit_entities: max entities returned (default 100)
        - limit_edges_per_entity: max edges per entity (default 50)
        - time_point: ISO timestamp (optional)
        - storage_path: graph storage path (optional, triggers switch)
        - focus_family_id / focus_absolute_id: focus entity
        - hops: hop count in focus mode (default 1)
        """
        try:
            limit_entities = request.args.get('limit_entities', type=int, default=100)
            limit_edges_per_entity = request.args.get('limit_edges_per_entity', type=int, default=50)
            time_point_str = request.args.get('time_point')
            storage_path_param = request.args.get('storage_path')
            focus_family_id = request.args.get('focus_family_id')
            focus_absolute_id = request.args.get('focus_absolute_id')
            hops = request.args.get('hops', type=int, default=1)

            if storage_path_param and storage_path_param.strip():
                storage_path_param = storage_path_param.strip()
                try:
                    server._switch_storage_path(storage_path_param)
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

            if focus_family_id and focus_absolute_id:
                focus_entity = server.storage.get_entity_by_absolute_id(focus_absolute_id)
                if not focus_entity or focus_entity.family_id != focus_family_id:
                    return jsonify({
                        'success': False,
                        'error': f'未找到指定的实体版本: {focus_family_id}/{focus_absolute_id}'
                    }), 404

                entities = [focus_entity]
                focus_time_point = focus_entity.event_time
            else:
                if time_point:
                    entities = server.storage.get_all_entities_before_time(time_point, limit=limit_entities)
                else:
                    entities = server.storage.get_all_entities(limit=limit_entities)

                if not entities:
                    return jsonify({
                        'success': False,
                        'error': '没有找到实体数据'
                    })

            entity_absolute_ids = {entity.absolute_id for entity in entities}
            family_id_to_name = {}
            family_id_to_absolute_id = {}
            family_id_to_hop_level = {}

            # Batch-prefetch version counts (non-focus entities)
            non_focus_fids = [
                e.family_id for e in entities
                if hasattr(e, 'family_id') and e.family_id
                and not (focus_family_id and focus_family_id == e.family_id and focus_absolute_id)
            ]
            try:
                batch_version_counts = server.storage.get_entity_version_counts(non_focus_fids) or {}
            except Exception:
                batch_version_counts = {}

            # Build initial node data
            nodes = []
            for entity in entities:
                try:
                    if not entity or not hasattr(entity, 'family_id') or not hasattr(entity, 'absolute_id'):
                        logger.debug("跳过无效实体: %s", entity)
                        continue

                    family_id_to_name[entity.family_id] = entity.name
                    family_id_to_absolute_id[entity.family_id] = entity.absolute_id
                    if focus_family_id and focus_family_id == entity.family_id:
                        family_id_to_hop_level[entity.family_id] = 0

                    is_focus_entity = focus_family_id and focus_family_id == entity.family_id and focus_absolute_id
                    if is_focus_entity:
                        try:
                            versions = server.storage.get_entity_versions(entity.family_id)
                            version_count = len(versions) if versions else 0
                        except Exception as e:
                            logger.warning("获取实体版本失败 (family_id=%s): %s", entity.family_id, e)
                            versions = []
                            version_count = 1
                    else:
                        version_count = batch_version_counts.get(entity.family_id, 1) or 1

                    # Label with version info
                    if focus_family_id and focus_family_id == entity.family_id and focus_absolute_id:
                        try:
                            versions_sorted = sorted(
                                versions,
                                key=lambda v: v.processed_time if isinstance(v.processed_time, datetime) else datetime.fromisoformat(str(v.processed_time).replace("Z", "+00:00"))
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
                            logger.warning("处理版本索引时出错 (family_id=%s): %s", entity.family_id, e)
                            label = f"{entity.name} ({version_count}版本)" if version_count > 1 else entity.name
                    else:
                        label = f"{entity.name} ({version_count}版本)" if version_count > 1 else entity.name

                    hop_level = family_id_to_hop_level.get(entity.family_id, 0)
                    node_color = _get_hop_color(hop_level)

                    try:
                        event_time_str = entity.event_time.isoformat() if entity.event_time else None
                        processed_time_str = entity.processed_time.isoformat() if entity.processed_time else None
                    except Exception as e:
                        logger.warning("实体时间格式错误 (family_id=%s): %s", entity.family_id, e)
                        event_time_str = None
                        processed_time_str = None

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
                    logger.error("处理实体时发生错误 (entity=%s): %s", entity, e, exc_info=True)
                    continue

            # Collect edges
            edges = []
            edges_seen: Set[tuple] = set()
            all_related_family_ids: Set[str] = set()

            if focus_family_id and focus_absolute_id and hops > 0:
                # ===== Multi-hop focus mode =====
                final_edge_candidates = []
                graph_edges_for_bfs = []
                graph_nodes = {focus_family_id}

                current_level_entities = {focus_family_id: focus_absolute_id}
                processed_family_ids: Set[str] = set()

                for current_hop in range(1, hops + 1):
                    next_level_entities = {}

                    for fam_id, max_abs_id in current_level_entities.items():
                        if fam_id in processed_family_ids:
                            continue
                        processed_family_ids.add(fam_id)

                        entity_abs_ids = server.storage.get_entity_absolute_ids_up_to_version(fam_id, max_abs_id)
                        if not entity_abs_ids:
                            continue

                        entity_relations = server.storage.get_relations_by_entity_absolute_ids(entity_abs_ids, limit=None)

                        # Batch preload endpoints
                        all_rel_abs_ids = set()
                        for rel in entity_relations:
                            all_rel_abs_ids.add(rel.entity1_absolute_id)
                            all_rel_abs_ids.add(rel.entity2_absolute_id)
                        batch_fn = getattr(server.storage, 'get_entities_by_absolute_ids', None)
                        entity_by_abs = {}
                        if batch_fn and all_rel_abs_ids:
                            entity_by_abs = {e.absolute_id: e for e in batch_fn(list(all_rel_abs_ids)) if e}

                        # Degree map for sorting
                        focus_degree_map = {}
                        focus_end_fids = {e.family_id for e in entity_by_abs.values() if e}
                        if focus_end_fids:
                            focus_degree_map = server.storage.batch_get_entity_degrees(list(focus_end_fids))

                        relation_candidates = []
                        for relation in entity_relations:
                            entity1 = entity_by_abs.get(relation.entity1_absolute_id)
                            entity2 = entity_by_abs.get(relation.entity2_absolute_id)
                            if not entity1:
                                entity1 = server.storage.get_entity_by_absolute_id(relation.entity1_absolute_id)
                            if not entity2:
                                entity2 = server.storage.get_entity_by_absolute_id(relation.entity2_absolute_id)

                            if entity1 and entity2:
                                entity1_fid = entity1.family_id
                                entity2_fid = entity2.family_id
                                normalized_pair = normalize_entity_pair(entity1_fid, entity2_fid)
                                ne1 = normalized_pair[0]
                                ne2 = normalized_pair[1]
                                edge_key = (ne1, ne2, relation.absolute_id)
                                if edge_key not in edges_seen:
                                    other_entity = entity2 if relation.entity1_absolute_id in entity_abs_ids else entity1
                                    other_entity_fid = other_entity.family_id
                                    other_entity_abs_id = other_entity.absolute_id
                                    other_edge_count = focus_degree_map.get(other_entity_fid, 0)
                                    relation_candidates.append({
                                        'relation': relation, 'entity1': entity1, 'entity2': entity2,
                                        'ne1': ne1, 'ne2': ne2, 'edge_key': edge_key,
                                        'other_entity': other_entity, 'other_entity_fid': other_entity_fid,
                                        'other_entity_abs_id': other_entity_abs_id,
                                        'other_entity_edge_count': other_edge_count
                                    })

                        relation_candidates.sort(key=lambda x: x['other_entity_edge_count'], reverse=True)
                        if limit_edges_per_entity:
                            relation_candidates = relation_candidates[:limit_edges_per_entity]

                        for cand in relation_candidates:
                            edges_seen.add(cand['edge_key'])
                            final_edge_candidates.append(cand)
                            graph_edges_for_bfs.append((cand['ne1'], cand['ne2']))
                            graph_nodes.add(cand['ne1'])
                            graph_nodes.add(cand['ne2'])

                            for nid in (cand['ne1'], cand['ne2']):
                                if nid not in family_id_to_name:
                                    ent = cand['entity1'] if nid == cand['ne1'] else cand['entity2']
                                    family_id_to_name[nid] = ent.name
                                    family_id_to_absolute_id[nid] = ent.absolute_id
                                    all_related_family_ids.add(nid)

                            other_entity = cand['other_entity']
                            other_entity_fid = cand['other_entity_fid']
                            other_entity_abs_id = cand['other_entity_abs_id']

                            if current_hop < hops and other_entity_fid not in processed_family_ids:
                                if other_entity_fid in next_level_entities:
                                    existing_abs_id = next_level_entities[other_entity_fid]
                                    existing_entity = server.storage.get_entity_by_absolute_id(existing_abs_id)
                                    if existing_entity and other_entity.event_time > existing_entity.event_time:
                                        next_level_entities[other_entity_fid] = other_entity_abs_id
                                else:
                                    next_level_entities[other_entity_fid] = other_entity_abs_id

                    current_level_entities = next_level_entities
                    if not current_level_entities:
                        break

                # Compute shortest paths and set hop levels
                shortest_paths = _bfs_shortest_paths(focus_family_id, graph_edges_for_bfs)
                for fid in graph_nodes:
                    family_id_to_hop_level[fid] = shortest_paths.get(fid, 999)

                # Build final edges
                for cand in final_edge_candidates:
                    relation = cand['relation']
                    edge_label = ""
                    if relation.content:
                        edge_label = relation.content[:30] + "..." if len(relation.content) > 30 else relation.content

                    edges.append({
                        'from': cand['ne1'], 'to': cand['ne2'],
                        'label': edge_label, 'title': relation.content,
                        'content': relation.content,
                        'event_time': relation.event_time.isoformat() if relation.event_time else None,
                        'processed_time': relation.processed_time.isoformat() if relation.processed_time else None,
                        'family_id': relation.family_id, 'absolute_id': relation.absolute_id,
                        'color': '#888888', 'width': 2, 'arrows': ''
                    })
            else:
                # ===== Non-focus mode / hops=0 =====
                all_entity_relations = []
                for entity in entities:
                    max_version_absolute_id = focus_absolute_id if (focus_family_id and focus_family_id == entity.family_id) else None
                    effective_time_point = None if max_version_absolute_id else time_point
                    entity_relations = server.storage.get_entity_relations_by_family_id(
                        entity.family_id, limit=None, time_point=effective_time_point,
                        max_version_absolute_id=max_version_absolute_id
                    )
                    all_entity_relations.append((entity, entity_relations, effective_time_point))

                # Batch preload all endpoints
                all_rel_abs_ids = set()
                for _, rels, _ in all_entity_relations:
                    for rel in rels:
                        all_rel_abs_ids.add(rel.entity1_absolute_id)
                        all_rel_abs_ids.add(rel.entity2_absolute_id)
                batch_fn = getattr(server.storage, 'get_entities_by_absolute_ids', None)
                entity_by_abs = {}
                if batch_fn and all_rel_abs_ids:
                    entity_by_abs = {e.absolute_id: e for e in batch_fn(list(all_rel_abs_ids)) if e}

                # Degree map
                all_end_fids: Set[str] = set()
                for _, rels, _ in all_entity_relations:
                    for rel in rels:
                        e1 = entity_by_abs.get(rel.entity1_absolute_id)
                        e2 = entity_by_abs.get(rel.entity2_absolute_id)
                        if e1: all_end_fids.add(e1.family_id)
                        if e2: all_end_fids.add(e2.family_id)
                degree_map = {}
                if all_end_fids:
                    degree_map = server.storage.batch_get_entity_degrees(list(all_end_fids))

                for entity, entity_relations, effective_time_point in all_entity_relations:
                    relation_candidates = []
                    for relation in entity_relations:
                        entity1_temp = entity_by_abs.get(relation.entity1_absolute_id)
                        entity2_temp = entity_by_abs.get(relation.entity2_absolute_id)
                        if not entity1_temp:
                            entity1_temp = server.storage.get_entity_by_absolute_id(relation.entity1_absolute_id)
                        if not entity2_temp:
                            entity2_temp = server.storage.get_entity_by_absolute_id(relation.entity2_absolute_id)

                        if entity1_temp and entity2_temp:
                            effective_time_point_inner = focus_time_point if focus_family_id else time_point
                            if effective_time_point_inner:
                                entity1 = server.storage.get_entity_version_at_time(entity1_temp.family_id, effective_time_point_inner)
                                entity2 = server.storage.get_entity_version_at_time(entity2_temp.family_id, effective_time_point_inner)
                            else:
                                entity1 = entity1_temp
                                entity2 = entity2_temp

                            if entity1 and entity2:
                                entity1_fid = entity1.family_id
                                entity2_fid = entity2.family_id
                                normalized_pair = normalize_entity_pair(entity1_fid, entity2_fid)
                                ne1 = normalized_pair[0]
                                ne2 = normalized_pair[1]
                                edge_key = (ne1, ne2, relation.family_id)
                                if edge_key not in edges_seen:
                                    other_entity = entity2 if entity1_fid == entity.family_id else entity1
                                    other_entity_fid = other_entity.family_id
                                    other_edge_count = degree_map.get(other_entity_fid, 0)
                                    relation_candidates.append({
                                        'relation': relation, 'entity1': entity1, 'entity2': entity2,
                                        'ne1': ne1, 'ne2': ne2, 'edge_key': edge_key,
                                        'other_entity_edge_count': other_edge_count
                                    })

                    relation_candidates.sort(key=lambda x: x['other_entity_edge_count'], reverse=True)
                    if limit_edges_per_entity:
                        relation_candidates = relation_candidates[:limit_edges_per_entity]

                    for cand in relation_candidates:
                        relation = cand['relation']
                        entity1 = cand['entity1']
                        entity2 = cand['entity2']
                        ne1 = cand['ne1']
                        ne2 = cand['ne2']
                        edge_key = cand['edge_key']

                        edges_seen.add(edge_key)

                        if entity1.absolute_id not in entity_absolute_ids:
                            entity_absolute_ids.add(entity1.absolute_id)
                            all_related_family_ids.add(ne1)
                            family_id_to_name[ne1] = entity1.name
                            family_id_to_absolute_id[ne1] = entity1.absolute_id

                        if entity2.absolute_id not in entity_absolute_ids:
                            entity_absolute_ids.add(entity2.absolute_id)
                            all_related_family_ids.add(ne2)
                            family_id_to_name[ne2] = entity2.name
                            family_id_to_absolute_id[ne2] = entity2.absolute_id

                        if entity1.absolute_id in entity_absolute_ids or entity2.absolute_id in entity_absolute_ids:
                            edge_label = ""
                            if relation.content:
                                edge_label = relation.content[:30] + "..." if len(relation.content) > 30 else relation.content

                            edges.append({
                                'from': ne1, 'to': ne2,
                                'label': edge_label, 'title': relation.content,
                                'content': relation.content,
                                'event_time': relation.event_time.isoformat() if relation.event_time else None,
                                'processed_time': relation.processed_time.isoformat() if relation.processed_time else None,
                                'family_id': relation.family_id, 'absolute_id': relation.absolute_id,
                                'color': '#888888', 'width': 2, 'arrows': ''
                            })

            # Batch-prefetch related entity version counts
            existing_node_ids = {n['id'] for n in nodes}
            related_fids_to_prefetch = [fid for fid in all_related_family_ids if fid not in existing_node_ids]
            try:
                related_version_counts = server.storage.get_entity_version_counts(related_fids_to_prefetch) or {}
            except Exception:
                related_version_counts = {}

            # Batch-prefetch related entities
            related_abs_to_fetch = {
                family_id_to_absolute_id[fid]: fid
                for fid in related_fids_to_prefetch
                if family_id_to_absolute_id.get(fid)
            }
            related_entity_cache = {}
            if related_abs_to_fetch:
                try:
                    batch_fn = getattr(server.storage, 'get_entities_by_absolute_ids', None)
                    if batch_fn:
                        for e in batch_fn(list(related_abs_to_fetch)):
                            if e:
                                related_entity_cache[e.absolute_id] = e
                except Exception:
                    pass

            # Add related entity nodes
            for fid in all_related_family_ids:
                if fid not in existing_node_ids:
                    absolute_id = family_id_to_absolute_id.get(fid)
                    if absolute_id:
                        related_entity = related_entity_cache.get(absolute_id) or server.storage.get_entity_by_absolute_id(absolute_id)
                    else:
                        effective_time_point = focus_time_point if focus_family_id else time_point
                        if effective_time_point:
                            related_entity = server.storage.get_entity_version_at_time(fid, effective_time_point)
                        else:
                            related_entity = None

                    if related_entity:
                        need_version_index = focus_family_id and absolute_id
                        if need_version_index:
                            versions = server.storage.get_entity_versions(related_entity.family_id)
                            version_count = len(versions)
                        else:
                            version_count = related_version_counts.get(related_entity.family_id, 1) or 1

                        if focus_family_id and absolute_id:
                            versions_sorted = sorted(
                                versions,
                                key=lambda v: v.processed_time if isinstance(v.processed_time, datetime) else datetime.fromisoformat(str(v.processed_time).replace("Z", "+00:00"))
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

                        hop_level = family_id_to_hop_level.get(fid, 0)
                        node_color = _get_hop_color(hop_level)

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
                            'color': node_color,
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
            logger.error("/api/graphs/data 异常: %s: %s", type(e).__name__, e, exc_info=True)
            return jsonify({
                'success': False,
                'error': str(e),
                'error_type': type(e).__name__
            }), 500

    # =====================================================================
    # /api/graphs/config
    # =====================================================================
    @server.app.route('/api/graphs/config')
    def get_config():
        try:
            return jsonify({
                'success': True,
                'storage_path': server._current_storage_path
            })
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500

    # =====================================================================
    # /api/graphs/stats
    # =====================================================================
    @server.app.route('/api/graphs/stats')
    def get_stats():
        try:
            total_entities = server.storage.count_unique_entities()
            total_relations = server.storage.count_unique_relations()
            return jsonify({
                'success': True,
                'stats': {'total_entities': total_entities, 'total_relations': total_relations}
            })
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500

    # =====================================================================
    # /api/graphs/search  — semantic search
    # =====================================================================
    @server.app.route('/api/graphs/search', methods=['POST'])
    def search_graph():
        """Search graph API.

        JSON body:
        - query: natural language text
        - max_results: max entity+relation count (default 10)
        - storage_path: optional, triggers switch
        """
        try:
            data = request.get_json()
            if not data:
                return jsonify({'success': False, 'error': '请求数据格式错误，需要JSON格式'}), 400

            query = data.get('query', '').strip()
            if not query:
                return jsonify({'success': False, 'error': '查询文本不能为空'}), 400

            max_results = data.get('max_results', 10)
            storage_path_param = data.get('storage_path', '').strip() if data.get('storage_path') else None

            if storage_path_param:
                try:
                    server._switch_storage_path(storage_path_param)
                except Exception as e:
                    return jsonify({'success': False, 'error': f'切换存储路径失败: {str(e)}'}), 400

            try:
                logger.debug("搜索查询: %s, 最大结果数: %d", query, max_results)
                logger.debug("Embedding客户端可用: %s", server.embedding_client.is_available() if server.embedding_client else False)

                matched_entities = server.storage.search_entities_by_similarity(
                    query_name=query, query_content=query,
                    threshold=0.3, max_results=max_results,
                    content_snippet_length=100
                )

                matched_relations = server.storage.search_relations_by_similarity(
                    query_text=query, threshold=0.3, max_results=max_results
                )

                logger.debug("搜索完成，%d 实体, %d 关系", len(matched_entities), len(matched_relations))

                matched_entity_absolute_ids = {entity.absolute_id for entity in matched_entities}
                matched_relation_absolute_ids = {relation.absolute_id for relation in matched_relations}
                matched_family_ids = {entity.family_id for entity in matched_entities}
                matched_relation_ids = {relation.family_id for relation in matched_relations}

            except Exception as e:
                logger.error("搜索错误: %s", e, exc_info=True)
                return jsonify({'success': False, 'error': f'搜索过程中发生错误: {str(e)}'}), 500

            if not matched_entities and not matched_relations:
                return jsonify({
                    'success': True, 'nodes': [], 'edges': [],
                    'stats': {'total_entities': 0, 'total_relations': 0,
                              'matched_entities': 0, 'matched_relations': 0},
                    'query': query
                })

            # Collect entities (1-hop)
            entity_absolute_ids = set(matched_entity_absolute_ids)
            family_id_to_name = {}
            family_id_to_absolute_id = {}

            all_relations_for_preload = list(matched_relations)
            entity_relation_map = {}
            for entity in matched_entities:
                entity_rels = server.storage.get_entity_relations(entity.absolute_id, limit=None)
                entity_relation_map[entity.absolute_id] = entity_rels
                all_relations_for_preload.extend(entity_rels)

            # Batch preload endpoints
            all_rel_abs_ids = set()
            for rel in all_relations_for_preload:
                all_rel_abs_ids.add(rel.entity1_absolute_id)
                all_rel_abs_ids.add(rel.entity2_absolute_id)
            batch_fn = getattr(server.storage, 'get_entities_by_absolute_ids', None)
            abs_to_entity = {}
            if batch_fn and all_rel_abs_ids:
                abs_to_entity = {e.absolute_id: e for e in batch_fn(list(all_rel_abs_ids)) if e}

            # Endpoints from matched relations
            for relation in matched_relations:
                entity1 = abs_to_entity.get(relation.entity1_absolute_id) or server.storage.get_entity_by_absolute_id(relation.entity1_absolute_id)
                entity2 = abs_to_entity.get(relation.entity2_absolute_id) or server.storage.get_entity_by_absolute_id(relation.entity2_absolute_id)
                if entity1:
                    entity_absolute_ids.add(entity1.absolute_id)
                    family_id_to_name[entity1.family_id] = entity1.name
                    family_id_to_absolute_id[entity1.family_id] = entity1.absolute_id
                    abs_to_entity[entity1.absolute_id] = entity1
                if entity2:
                    entity_absolute_ids.add(entity2.absolute_id)
                    family_id_to_name[entity2.family_id] = entity2.name
                    family_id_to_absolute_id[entity2.family_id] = entity2.absolute_id
                    abs_to_entity[entity2.absolute_id] = entity2

            relation_absolute_ids = set(matched_relation_absolute_ids)
            edges_seen = set()

            # 1-hop from matched entities
            for entity in matched_entities:
                entity_relations = entity_relation_map.get(entity.absolute_id, [])
                for relation in entity_relations:
                    relation_absolute_ids.add(relation.absolute_id)
                    entity1 = abs_to_entity.get(relation.entity1_absolute_id)
                    entity2 = abs_to_entity.get(relation.entity2_absolute_id)
                    if entity1 and entity2:
                        e1_fid = entity1.family_id
                        e2_fid = entity2.family_id
                        edge_key = (e1_fid, e2_fid, relation.family_id)
                        if edge_key in edges_seen:
                            continue
                        edges_seen.add(edge_key)
                        if entity1.absolute_id not in entity_absolute_ids:
                            entity_absolute_ids.add(entity1.absolute_id)
                            family_id_to_name[e1_fid] = entity1.name
                            family_id_to_absolute_id[e1_fid] = entity1.absolute_id
                        if entity2.absolute_id not in entity_absolute_ids:
                            entity_absolute_ids.add(entity2.absolute_id)
                            family_id_to_name[e2_fid] = entity2.name
                            family_id_to_absolute_id[e2_fid] = entity2.absolute_id

            # Build nodes (dedup by family_id)
            nodes = []
            family_id_to_latest_absolute_id = {}

            unloaded_abs_ids = [aid for aid in entity_absolute_ids if aid not in abs_to_entity]
            if batch_fn and unloaded_abs_ids:
                for e in batch_fn(unloaded_abs_ids):
                    if e:
                        abs_to_entity[e.absolute_id] = e

            for entity_abs_id in entity_absolute_ids:
                entity = abs_to_entity.get(entity_abs_id) or server.storage.get_entity_by_absolute_id(entity_abs_id)
                if entity:
                    fid = entity.family_id
                    if fid not in family_id_to_latest_absolute_id:
                        family_id_to_latest_absolute_id[fid] = entity_abs_id
                    else:
                        existing_entity = abs_to_entity.get(family_id_to_latest_absolute_id[fid]) or server.storage.get_entity_by_absolute_id(family_id_to_latest_absolute_id[fid])
                        if existing_entity and entity.event_time > existing_entity.event_time:
                            family_id_to_latest_absolute_id[fid] = entity_abs_id

            try:
                search_version_counts = server.storage.get_entity_version_counts(list(family_id_to_latest_absolute_id)) or {}
            except Exception:
                search_version_counts = {}

            for fid, entity_abs_id in family_id_to_latest_absolute_id.items():
                entity = abs_to_entity.get(entity_abs_id) or server.storage.get_entity_by_absolute_id(entity_abs_id)
                if entity:
                    is_matched = entity.family_id in matched_family_ids
                    version_count = search_version_counts.get(entity.family_id, 1) or 1
                    label = f"{entity.name} ({version_count}版本)" if version_count > 1 else entity.name

                    nodes.append({
                        'id': entity.family_id, 'family_id': entity.family_id,
                        'absolute_id': entity.absolute_id, 'label': label,
                        'title': f"{entity.name}\n\n{entity.content[:100]}..." if len(entity.content) > 100 else f"{entity.name}\n\n{entity.content}",
                        'content': entity.content,
                        'event_time': entity.event_time.isoformat() if entity.event_time else None,
                        'processed_time': entity.processed_time.isoformat() if entity.processed_time else None,
                        'version_count': version_count,
                        'color': '#FF6B6B' if is_matched else '#97C2FC',
                        'shape': 'dot', 'size': 25 if is_matched else 20,
                        'font': {'color': 'white'}
                    })

            # Build edges
            edges = []
            edges_seen = set()

            for relation in matched_relations:
                entity1 = abs_to_entity.get(relation.entity1_absolute_id) or server.storage.get_entity_by_absolute_id(relation.entity1_absolute_id)
                entity2 = abs_to_entity.get(relation.entity2_absolute_id) or server.storage.get_entity_by_absolute_id(relation.entity2_absolute_id)
                if entity1 and entity2:
                    edge_key = (entity1.family_id, entity2.family_id, relation.family_id)
                    if edge_key not in edges_seen:
                        edges_seen.add(edge_key)
                        edge_label = ""
                        if relation.content:
                            edge_label = relation.content[:30] + "..." if len(relation.content) > 30 else relation.content
                        edges.append({
                            'from': entity1.family_id, 'to': entity2.family_id,
                            'label': edge_label, 'title': relation.content,
                            'content': relation.content,
                            'event_time': relation.event_time.isoformat() if relation.event_time else None,
                            'processed_time': relation.processed_time.isoformat() if relation.processed_time else None,
                            'family_id': relation.family_id, 'absolute_id': relation.absolute_id,
                            'color': '#FF6B6B', 'width': 3, 'arrows': ''
                        })

            for entity in matched_entities:
                entity_relations = entity_relation_map.get(entity.absolute_id, [])
                for relation in entity_relations:
                    entity1 = abs_to_entity.get(relation.entity1_absolute_id) or server.storage.get_entity_by_absolute_id(relation.entity1_absolute_id)
                    entity2 = abs_to_entity.get(relation.entity2_absolute_id) or server.storage.get_entity_by_absolute_id(relation.entity2_absolute_id)
                    if entity1 and entity2:
                        edge_key = (entity1.family_id, entity2.family_id, relation.family_id)
                        if edge_key not in edges_seen:
                            edges_seen.add(edge_key)
                            is_matched = relation.family_id in matched_relation_ids
                            edge_label = ""
                            if relation.content:
                                edge_label = relation.content[:30] + "..." if len(relation.content) > 30 else relation.content
                            edges.append({
                                'from': entity1.family_id, 'to': entity2.family_id,
                                'label': edge_label, 'title': relation.content,
                                'content': relation.content,
                                'event_time': relation.event_time.isoformat() if relation.event_time else None,
                                'processed_time': relation.processed_time.isoformat() if relation.processed_time else None,
                                'family_id': relation.family_id, 'absolute_id': relation.absolute_id,
                                'color': '#FF6B6B' if is_matched else '#97C2FC',
                                'width': 3 if is_matched else 2, 'arrows': ''
                            })

            return jsonify({
                'success': True, 'nodes': nodes, 'edges': edges,
                'stats': {
                    'total_entities': len(nodes), 'total_relations': len(edges),
                    'matched_entities': len(matched_entities),
                    'matched_relations': len(matched_relations)
                },
                'query': query
            })
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500

    # =====================================================================
    # /api/graphs/snapshot  — time-point snapshot
    # =====================================================================
    @server.app.route('/api/graphs/snapshot', methods=['POST'])
    def get_graph_snapshot():
        """Snapshot at a given time point."""
        try:
            data = request.get_json()
            if not data:
                return jsonify({'success': False, 'error': '请求数据格式错误，需要JSON格式'}), 400

            entity_versions = data.get('entity_versions', {})
            relation_versions = data.get('relation_versions', {})
            time_point_str = data.get('time_point')

            time_point = None
            if time_point_str:
                try:
                    time_point = datetime.fromisoformat(time_point_str)
                except (ValueError, TypeError):
                    pass

            # Batch-get entity info
            entity_abs_ids = list(entity_versions.values())
            entity_map = server.storage.get_entities_by_absolute_ids(entity_abs_ids) if entity_abs_ids else {}
            entity_fids = list(entity_versions)
            version_count_map = server.storage.get_entity_version_counts(entity_fids) if entity_fids else {}

            nodes_data = []
            for family_id, absolute_id in entity_versions.items():
                entity = entity_map.get(absolute_id)
                if entity:
                    version_count = version_count_map.get(family_id, 1)
                    label = f"{entity.name} ({version_count}版本)" if version_count > 1 else entity.name
                    nodes_data.append({
                        'id': family_id, 'family_id': family_id,
                        'absolute_id': absolute_id, 'label': label,
                        'name': entity.name, 'content': entity.content,
                        'event_time': entity.event_time.isoformat() if entity.event_time else None,
                        'processed_time': entity.processed_time.isoformat() if entity.processed_time else None,
                        'version_count': version_count
                    })

            # Batch-get relation info
            edges_data = []
            all_rel_entity_abs_ids = set()
            rel_versions_cache = {}
            for family_id, absolute_id in relation_versions.items():
                versions = server.storage.get_relation_versions(family_id)
                relation = next((r for r in versions if r.absolute_id == absolute_id), None)
                if relation:
                    rel_versions_cache[family_id] = relation
                    all_rel_entity_abs_ids.add(relation.entity1_absolute_id)
                    all_rel_entity_abs_ids.add(relation.entity2_absolute_id)

            rel_entity_map = server.storage.get_entities_by_absolute_ids(list(all_rel_entity_abs_ids)) if all_rel_entity_abs_ids else {}

            for family_id, absolute_id in relation_versions.items():
                relation = rel_versions_cache.get(family_id)
                if relation:
                    entity1 = rel_entity_map.get(relation.entity1_absolute_id)
                    entity2 = rel_entity_map.get(relation.entity2_absolute_id)
                    if entity1 and entity2:
                        edges_data.append({
                            'family_id': family_id, 'absolute_id': absolute_id,
                            'from': entity1.family_id, 'to': entity2.family_id,
                            'content': relation.content,
                            'event_time': relation.event_time.isoformat() if relation.event_time else None,
                            'processed_time': relation.processed_time.isoformat() if relation.processed_time else None
                        })

            return jsonify({'success': True, 'nodes': nodes_data, 'edges': edges_data})
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500
