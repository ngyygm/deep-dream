"""
Extracted from web.py — Graph data and search route handlers for GraphWebServer.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Set

from flask import jsonify, request

from core.server.web_graph_data import (
    batch_preload_entities,
    batch_preload_version_counts,
    batch_preload_degrees,
    build_entity_nodes_with_version_index,
    build_edge_dict,
    build_focus_edges,
    build_related_entity_nodes,
    build_search_edges_from_entity_relations,
    build_search_edges_from_matched_relations,
    build_search_nodes,
    build_snapshot_edges,
    build_snapshot_nodes,
    collect_focus_edges,
    collect_non_focus_edges,
    collect_relation_endpoint_abs_ids,
    preload_related_entities,
)

logger = logging.getLogger(__name__)


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

            # Batch-prefetch version counts (non-focus entities)
            non_focus_fids = [
                e.family_id for e in entities
                if hasattr(e, 'family_id') and e.family_id
                and not (focus_family_id and focus_family_id == e.family_id and focus_absolute_id)
            ]
            batch_version_counts = batch_preload_version_counts(server.storage, non_focus_fids)

            # Build initial node data
            nodes, family_id_to_name, family_id_to_absolute_id, family_id_to_hop_level = \
                build_entity_nodes_with_version_index(
                    server.storage, entities, batch_version_counts,
                    focus_family_id=focus_family_id, focus_absolute_id=focus_absolute_id,
                )

            # Collect edges
            all_related_family_ids: Set[str] = set()
            edges_seen: Set[tuple] = set()

            if focus_family_id and focus_absolute_id and hops > 0:
                # ===== Multi-hop focus mode =====
                (final_edge_candidates, graph_edges_for_bfs, all_related_family_ids,
                 focus_fid_to_name, focus_fid_to_abs_id, family_id_to_hop_level) = \
                    collect_focus_edges(
                        server.storage, focus_family_id, focus_absolute_id,
                        hops, limit_edges_per_entity, edges_seen,
                    )
                # Merge maps from focus edge collection
                family_id_to_name.update(focus_fid_to_name)
                family_id_to_absolute_id.update(focus_fid_to_abs_id)
                edges = build_focus_edges(final_edge_candidates)
            else:
                # ===== Non-focus mode / hops=0 =====
                edges, all_related_family_ids, nf_fid_to_name, nf_fid_to_abs_id = \
                    collect_non_focus_edges(
                        server.storage, entities,
                        focus_family_id, focus_absolute_id,
                        focus_time_point if focus_family_id else None, time_point,
                        limit_edges_per_entity, entity_absolute_ids,
                    )
                family_id_to_name.update(nf_fid_to_name)
                family_id_to_absolute_id.update(nf_fid_to_abs_id)

            # Preload and build related entity nodes
            existing_node_ids = {n['id'] for n in nodes}
            related_version_counts, related_entity_cache = preload_related_entities(
                server.storage, all_related_family_ids, existing_node_ids, family_id_to_absolute_id,
            )
            related_nodes, _ = build_related_entity_nodes(
                server.storage, all_related_family_ids, existing_node_ids,
                family_id_to_absolute_id, family_id_to_hop_level,
                related_version_counts, related_entity_cache,
                focus_family_id=focus_family_id,
                focus_time_point=focus_time_point if focus_family_id else None,
                time_point=time_point,
            )
            nodes.extend(related_nodes)

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
                    query,
                    threshold=0.3, max_results=max_results,
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
            all_rel_abs_ids = collect_relation_endpoint_abs_ids(all_relations_for_preload)
            abs_to_entity = batch_preload_entities(server.storage, all_rel_abs_ids)

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

            # Build nodes
            nodes, _ = build_search_nodes(
                server.storage, entity_absolute_ids, abs_to_entity, matched_family_ids,
            )

            # Build edges
            matched_edges, edges_seen, abs_to_entity = build_search_edges_from_matched_relations(
                matched_relations, abs_to_entity, server.storage,
            )
            hop_edges = build_search_edges_from_entity_relations(
                matched_entities, entity_relation_map, abs_to_entity,
                server.storage, matched_relation_ids, edges_seen,
            )
            edges = matched_edges + hop_edges

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

            nodes_data = build_snapshot_nodes(server.storage, entity_versions)
            edges_data = build_snapshot_edges(server.storage, relation_versions)

            return jsonify({'success': True, 'nodes': nodes_data, 'edges': edges_data})
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500
