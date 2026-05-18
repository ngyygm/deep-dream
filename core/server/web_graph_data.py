"""
Data formatting and query-building helpers for web_graph route handlers.

Extracted from web_graph.py — contains node/edge construction, batch preloading,
hop color mapping, and BFS shortest-path computation.
"""
from __future__ import annotations

import logging
from collections import deque
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

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


def get_hop_color(hop_level: int) -> str:
    """Return a hex color for a given hop level."""
    return _HOP_COLORS[hop_level % len(_HOP_COLORS)]


# -- BFS shortest path (for focus mode) --------------------------------------

def bfs_shortest_paths(start_node: str, edges_list: List[Tuple[str, str]]) -> Dict[str, int]:
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


# -- Batch preloading helpers ------------------------------------------------

def batch_preload_entities(storage, absolute_ids: Set[str]) -> Dict[str, Any]:
    """Batch-fetch entities by absolute IDs. Returns {absolute_id: entity}."""
    batch_fn = getattr(storage, 'get_entities_by_absolute_ids', None)
    if batch_fn and absolute_ids:
        return {e.absolute_id: e for e in batch_fn(list(absolute_ids)) if e}
    return {}


def batch_preload_version_counts(storage, family_ids: List[str]) -> Dict[str, int]:
    """Batch-fetch version counts for a list of family_ids."""
    if not family_ids:
        return {}
    try:
        return storage.get_entity_version_counts(family_ids) or {}
    except Exception:
        return {}


def batch_preload_degrees(storage, family_ids: List[str]) -> Dict[str, int]:
    """Batch-fetch entity degree counts."""
    if not family_ids:
        return {}
    try:
        return storage.batch_get_entity_degrees(family_ids)
    except Exception:
        return {}


def collect_relation_endpoint_abs_ids(relations) -> Set[str]:
    """Collect all entity1/entity2 absolute IDs from a list of relations."""
    ids: Set[str] = set()
    for rel in relations:
        ids.add(rel.entity1_absolute_id)
        ids.add(rel.entity2_absolute_id)
    return ids


# -- Node/edge building helpers ----------------------------------------------

def build_entity_label(entity_name: str, version_count: int,
                       current_version_index: Optional[int] = None) -> str:
    """Build a display label for an entity with optional version info."""
    if current_version_index is not None and version_count > 1:
        return f"{entity_name} ({current_version_index}/{version_count}版本)"
    if version_count > 1:
        return f"{entity_name} ({version_count}版本)"
    return entity_name


def build_node_dict(
    entity,
    version_count: int = 1,
    hop_level: int = 0,
    current_version_index: Optional[int] = None,
    color: Optional[str] = None,
    size: int = 20,
) -> Dict[str, Any]:
    """Build a vis-network node dict from an entity object."""
    name = getattr(entity, 'name', None) or '未知实体'
    content = getattr(entity, 'content', None) or ''
    family_id = getattr(entity, 'family_id', None)
    absolute_id = getattr(entity, 'absolute_id', None)

    label = build_entity_label(name, version_count, current_version_index)
    node_color = color if color is not None else get_hop_color(hop_level)

    try:
        event_time_str = entity.event_time.isoformat() if entity.event_time else None
        processed_time_str = entity.processed_time.isoformat() if entity.processed_time else None
    except Exception as e:
        logger.warning("实体时间格式错误 (family_id=%s): %s", family_id, e)
        event_time_str = None
        processed_time_str = None

    title = f"{name}\n\n{content[:100]}..." if len(content) > 100 else f"{name}\n\n{content}"

    return {
        'id': family_id,
        'family_id': family_id,
        'absolute_id': absolute_id,
        'label': label,
        'title': title,
        'content': content,
        'event_time': event_time_str,
        'processed_time': processed_time_str,
        'version_count': version_count,
        'color': node_color,
        'shape': 'dot',
        'size': size,
        'font': {'color': 'white'}
    }


def build_edge_dict(
    from_id: str,
    to_id: str,
    relation,
    color: str = '#888888',
    width: int = 2,
    arrows: str = '',
) -> Dict[str, Any]:
    """Build a vis-network edge dict from a relation object."""
    content = getattr(relation, 'content', None) or ''
    edge_label = ""
    if content:
        edge_label = content[:30] + "..." if len(content) > 30 else content

    return {
        'from': from_id,
        'to': to_id,
        'label': edge_label,
        'title': content,
        'content': content,
        'event_time': relation.event_time.isoformat() if relation.event_time else None,
        'processed_time': relation.processed_time.isoformat() if relation.processed_time else None,
        'family_id': relation.family_id,
        'absolute_id': relation.absolute_id,
        'color': color,
        'width': width,
        'arrows': arrows,
    }


def build_entity_nodes_with_version_index(
    storage,
    entities,
    batch_version_counts: Dict[str, int],
    focus_family_id: Optional[str] = None,
    focus_absolute_id: Optional[str] = None,
) -> Tuple[List[Dict], Dict[str, str], Dict[str, str], Dict[str, int]]:
    """
    Build node dicts for a list of entities, with version index info for focus entities.

    Returns (nodes, family_id_to_name, family_id_to_absolute_id, family_id_to_hop_level).
    """
    family_id_to_name: Dict[str, str] = {}
    family_id_to_absolute_id: Dict[str, str] = {}
    family_id_to_hop_level: Dict[str, int] = {}

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
                    versions = storage.get_entity_versions(entity.family_id)
                    version_count = len(versions) if versions else 0
                except Exception as e:
                    logger.warning("获取实体版本失败 (family_id=%s): %s", entity.family_id, e)
                    versions = []
                    version_count = 1
            else:
                version_count = batch_version_counts.get(entity.family_id, 1) or 1

            # Compute version index for focus entity
            current_version_index = None
            if is_focus_entity and version_count > 1:
                try:
                    versions_sorted = sorted(
                        versions,
                        key=lambda v: v.processed_time if isinstance(v.processed_time, datetime)
                        else datetime.fromisoformat(str(v.processed_time).replace("Z", "+00:00"))
                    )
                    for idx, v in enumerate(versions_sorted, 1):
                        if v.absolute_id == focus_absolute_id:
                            current_version_index = idx
                            break
                except Exception as e:
                    logger.warning("处理版本索引时出错 (family_id=%s): %s", entity.family_id, e)

            hop_level = family_id_to_hop_level.get(entity.family_id, 0)
            nodes.append(build_node_dict(
                entity,
                version_count=version_count,
                hop_level=hop_level,
                current_version_index=current_version_index,
            ))
        except Exception as e:
            logger.error("处理实体时发生错误 (entity=%s): %s", entity, e, exc_info=True)
            continue

    return nodes, family_id_to_name, family_id_to_absolute_id, family_id_to_hop_level


def build_related_entity_nodes(
    storage,
    all_related_family_ids: Set[str],
    existing_node_ids: Set[str],
    family_id_to_absolute_id: Dict[str, str],
    family_id_to_hop_level: Dict[str, int],
    related_version_counts: Dict[str, int],
    related_entity_cache: Dict[str, Any],
    focus_family_id: Optional[str] = None,
    focus_time_point=None,
    time_point=None,
) -> Tuple[List[Dict], Dict[str, str]]:
    """
    Build node dicts for related entities not already in the node set.

    Returns (new_nodes, updated family_id_to_name).
    """
    family_id_to_name: Dict[str, str] = {}
    new_nodes = []

    for fid in all_related_family_ids:
        if fid in existing_node_ids:
            continue
        absolute_id = family_id_to_absolute_id.get(fid)
        if absolute_id:
            related_entity = related_entity_cache.get(absolute_id) or storage.get_entity_by_absolute_id(absolute_id)
        else:
            effective_time_point = focus_time_point if focus_family_id else time_point
            if effective_time_point:
                related_entity = storage.get_entity_version_at_time(fid, effective_time_point)
            else:
                related_entity = None

        if not related_entity:
            continue

        need_version_index = focus_family_id and absolute_id
        if need_version_index:
            versions = storage.get_entity_versions(related_entity.family_id)
            version_count = len(versions)
        else:
            version_count = related_version_counts.get(related_entity.family_id, 1) or 1

        current_version_index = None
        if focus_family_id and absolute_id and version_count > 1:
            versions_sorted = sorted(
                versions,
                key=lambda v: v.processed_time if isinstance(v.processed_time, datetime)
                else datetime.fromisoformat(str(v.processed_time).replace("Z", "+00:00"))
            )
            for idx, v in enumerate(versions_sorted, 1):
                if v.absolute_id == related_entity.absolute_id:
                    current_version_index = idx
                    break

        hop_level = family_id_to_hop_level.get(fid, 0)
        new_nodes.append(build_node_dict(
            related_entity,
            version_count=version_count,
            hop_level=hop_level,
            current_version_index=current_version_index,
        ))

    return new_nodes, family_id_to_name


# -- Multi-hop focus mode helpers -------------------------------------------

def collect_focus_edges(
    storage,
    focus_family_id: str,
    focus_absolute_id: str,
    hops: int,
    limit_edges_per_entity: Optional[int],
    edges_seen: Set[tuple],
) -> Tuple[List[Dict], List[Tuple[str, str]], Set[str], Dict[str, str], Dict[str, str], Dict[str, int]]:
    """
    Collect edges in multi-hop focus mode.

    Returns (final_edge_candidates, graph_edges_for_bfs, all_related_family_ids,
             family_id_to_name, family_id_to_absolute_id, family_id_to_hop_level).
    """
    final_edge_candidates = []
    graph_edges_for_bfs = []
    graph_nodes = {focus_family_id}
    all_related_family_ids: Set[str] = set()
    family_id_to_name: Dict[str, str] = {}
    family_id_to_absolute_id: Dict[str, str] = {}
    family_id_to_hop_level: Dict[str, int] = {}

    current_level_entities = {focus_family_id: focus_absolute_id}
    processed_family_ids: Set[str] = set()

    for current_hop in range(1, hops + 1):
        next_level_entities = {}

        for fam_id, max_abs_id in current_level_entities.items():
            if fam_id in processed_family_ids:
                continue
            processed_family_ids.add(fam_id)

            entity_abs_ids = storage.get_entity_absolute_ids_up_to_version(fam_id, max_abs_id)
            if not entity_abs_ids:
                continue

            entity_relations = storage.get_relations_by_entity_absolute_ids(entity_abs_ids, limit=None)

            # Batch preload endpoints
            all_rel_abs_ids = collect_relation_endpoint_abs_ids(entity_relations)
            entity_by_abs = batch_preload_entities(storage, all_rel_abs_ids)

            # Degree map for sorting
            focus_end_fids = {e.family_id for e in entity_by_abs.values() if e}
            focus_degree_map = batch_preload_degrees(storage, list(focus_end_fids))

            relation_candidates = []
            for relation in entity_relations:
                entity1 = entity_by_abs.get(relation.entity1_absolute_id)
                entity2 = entity_by_abs.get(relation.entity2_absolute_id)
                if not entity1:
                    entity1 = storage.get_entity_by_absolute_id(relation.entity1_absolute_id)
                if not entity2:
                    entity2 = storage.get_entity_by_absolute_id(relation.entity2_absolute_id)

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
                        existing_entity = storage.get_entity_by_absolute_id(existing_abs_id)
                        if existing_entity and other_entity.event_time > existing_entity.event_time:
                            next_level_entities[other_entity_fid] = other_entity_abs_id
                    else:
                        next_level_entities[other_entity_fid] = other_entity_abs_id

        current_level_entities = next_level_entities
        if not current_level_entities:
            break

    # Compute shortest paths and set hop levels
    shortest_paths = bfs_shortest_paths(focus_family_id, graph_edges_for_bfs)
    for fid in graph_nodes:
        family_id_to_hop_level[fid] = shortest_paths.get(fid, 999)

    return (final_edge_candidates, graph_edges_for_bfs, all_related_family_ids,
            family_id_to_name, family_id_to_absolute_id, family_id_to_hop_level)


def build_focus_edges(final_edge_candidates: List[Dict]) -> List[Dict]:
    """Convert focus-mode edge candidates to edge dicts."""
    edges = []
    for cand in final_edge_candidates:
        relation = cand['relation']
        edges.append(build_edge_dict(cand['ne1'], cand['ne2'], relation))
    return edges


def collect_non_focus_edges(
    storage,
    entities,
    focus_family_id: Optional[str],
    focus_absolute_id: Optional[str],
    focus_time_point,
    time_point,
    limit_edges_per_entity: Optional[int],
    entity_absolute_ids: Set[str],
) -> Tuple[List[Dict], Set[str], Dict[str, str], Dict[str, str]]:
    """
    Collect edges in non-focus / hops=0 mode.

    Returns (edges, all_related_family_ids, family_id_to_name, family_id_to_absolute_id).
    """
    edges = []
    edges_seen: Set[tuple] = set()
    all_related_family_ids: Set[str] = set()
    family_id_to_name: Dict[str, str] = {}
    family_id_to_absolute_id: Dict[str, str] = {}

    all_entity_relations = []
    for entity in entities:
        max_version_absolute_id = focus_absolute_id if (focus_family_id and focus_family_id == entity.family_id) else None
        effective_time_point = None if max_version_absolute_id else time_point
        entity_relations = storage.get_entity_relations_by_family_id(
            entity.family_id, limit=None, time_point=effective_time_point,
            max_version_absolute_id=max_version_absolute_id
        )
        all_entity_relations.append((entity, entity_relations, effective_time_point))

    # Batch preload all endpoints
    all_rel_abs_ids: Set[str] = set()
    for _, rels, _ in all_entity_relations:
        all_rel_abs_ids.update(collect_relation_endpoint_abs_ids(rels))
    entity_by_abs = batch_preload_entities(storage, all_rel_abs_ids)

    # Degree map
    all_end_fids: Set[str] = set()
    for _, rels, _ in all_entity_relations:
        for rel in rels:
            e1 = entity_by_abs.get(rel.entity1_absolute_id)
            e2 = entity_by_abs.get(rel.entity2_absolute_id)
            if e1: all_end_fids.add(e1.family_id)
            if e2: all_end_fids.add(e2.family_id)
    degree_map = batch_preload_degrees(storage, list(all_end_fids))

    for entity, entity_relations, effective_time_point in all_entity_relations:
        relation_candidates = []
        for relation in entity_relations:
            entity1_temp = entity_by_abs.get(relation.entity1_absolute_id)
            entity2_temp = entity_by_abs.get(relation.entity2_absolute_id)
            if not entity1_temp:
                entity1_temp = storage.get_entity_by_absolute_id(relation.entity1_absolute_id)
            if not entity2_temp:
                entity2_temp = storage.get_entity_by_absolute_id(relation.entity2_absolute_id)

            if entity1_temp and entity2_temp:
                effective_time_point_inner = focus_time_point if focus_family_id else time_point
                if effective_time_point_inner:
                    entity1 = storage.get_entity_version_at_time(entity1_temp.family_id, effective_time_point_inner)
                    entity2 = storage.get_entity_version_at_time(entity2_temp.family_id, effective_time_point_inner)
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
                edges.append(build_edge_dict(ne1, ne2, relation))

    return edges, all_related_family_ids, family_id_to_name, family_id_to_absolute_id


# -- Related entity preloading -----------------------------------------------

def preload_related_entities(
    storage,
    all_related_family_ids: Set[str],
    existing_node_ids: Set[str],
    family_id_to_absolute_id: Dict[str, str],
) -> Tuple[Dict[str, int], Dict[str, Any]]:
    """
    Batch-prefetch version counts and entity objects for related entities.

    Returns (related_version_counts, related_entity_cache).
    """
    related_fids_to_prefetch = [fid for fid in all_related_family_ids if fid not in existing_node_ids]
    related_version_counts = batch_preload_version_counts(storage, related_fids_to_prefetch)

    related_abs_to_fetch = {
        family_id_to_absolute_id[fid]: fid
        for fid in related_fids_to_prefetch
        if family_id_to_absolute_id.get(fid)
    }
    related_entity_cache: Dict[str, Any] = {}
    if related_abs_to_fetch:
        try:
            batch_fn = getattr(storage, 'get_entities_by_absolute_ids', None)
            if batch_fn:
                for e in batch_fn(list(related_abs_to_fetch)):
                    if e:
                        related_entity_cache[e.absolute_id] = e
        except Exception:
            pass

    return related_version_counts, related_entity_cache


# -- Search-mode node/edge building ------------------------------------------

def build_search_nodes(
    storage,
    entity_absolute_ids: Set[str],
    abs_to_entity: Dict[str, Any],
    matched_family_ids: Set[str],
) -> Tuple[List[Dict], Dict[str, str]]:
    """
    Build node dicts for search results, deduplicating by family_id
    (keeping the latest entity per family).

    Returns (nodes, family_id_to_latest_absolute_id).
    """
    family_id_to_latest_absolute_id: Dict[str, str] = {}

    # Ensure all entities loaded
    batch_fn = getattr(storage, 'get_entities_by_absolute_ids', None)
    unloaded_abs_ids = [aid for aid in entity_absolute_ids if aid not in abs_to_entity]
    if batch_fn and unloaded_abs_ids:
        for e in batch_fn(unloaded_abs_ids):
            if e:
                abs_to_entity[e.absolute_id] = e

    # Dedup: keep latest per family
    for entity_abs_id in entity_absolute_ids:
        entity = abs_to_entity.get(entity_abs_id) or storage.get_entity_by_absolute_id(entity_abs_id)
        if entity:
            fid = entity.family_id
            if fid not in family_id_to_latest_absolute_id:
                family_id_to_latest_absolute_id[fid] = entity_abs_id
            else:
                existing_entity = abs_to_entity.get(family_id_to_latest_absolute_id[fid]) or \
                    storage.get_entity_by_absolute_id(family_id_to_latest_absolute_id[fid])
                if existing_entity and entity.event_time > existing_entity.event_time:
                    family_id_to_latest_absolute_id[fid] = entity_abs_id

    search_version_counts = batch_preload_version_counts(storage, list(family_id_to_latest_absolute_id))

    nodes = []
    for fid, entity_abs_id in family_id_to_latest_absolute_id.items():
        entity = abs_to_entity.get(entity_abs_id) or storage.get_entity_by_absolute_id(entity_abs_id)
        if entity:
            is_matched = entity.family_id in matched_family_ids
            version_count = search_version_counts.get(entity.family_id, 1) or 1
            nodes.append(build_node_dict(
                entity,
                version_count=version_count,
                color='#FF6B6B' if is_matched else '#97C2FC',
                size=25 if is_matched else 20,
            ))

    return nodes, family_id_to_latest_absolute_id


def build_search_edges_from_matched_relations(
    matched_relations,
    abs_to_entity: Dict[str, Any],
    storage,
) -> Tuple[List[Dict], Set[tuple], Dict[str, Any]]:
    """
    Build edge dicts from matched relations (search mode).

    Returns (edges, edges_seen, updated abs_to_entity).
    """
    edges = []
    edges_seen: Set[tuple] = set()

    for relation in matched_relations:
        entity1 = abs_to_entity.get(relation.entity1_absolute_id) or \
            storage.get_entity_by_absolute_id(relation.entity1_absolute_id)
        entity2 = abs_to_entity.get(relation.entity2_absolute_id) or \
            storage.get_entity_by_absolute_id(relation.entity2_absolute_id)
        if entity1 and entity2:
            edge_key = (entity1.family_id, entity2.family_id, relation.family_id)
            if edge_key not in edges_seen:
                edges_seen.add(edge_key)
                edges.append(build_edge_dict(
                    entity1.family_id, entity2.family_id, relation,
                    color='#FF6B6B', width=3,
                ))

    return edges, edges_seen, abs_to_entity


def build_search_edges_from_entity_relations(
    matched_entities,
    entity_relation_map: Dict[str, list],
    abs_to_entity: Dict[str, Any],
    storage,
    matched_relation_ids: Set[str],
    edges_seen: Set[tuple],
) -> List[Dict]:
    """Build edge dicts from entity 1-hop relations (search mode)."""
    edges = []

    for entity in matched_entities:
        entity_relations = entity_relation_map.get(entity.absolute_id, [])
        for relation in entity_relations:
            entity1 = abs_to_entity.get(relation.entity1_absolute_id) or \
                storage.get_entity_by_absolute_id(relation.entity1_absolute_id)
            entity2 = abs_to_entity.get(relation.entity2_absolute_id) or \
                storage.get_entity_by_absolute_id(relation.entity2_absolute_id)
            if entity1 and entity2:
                edge_key = (entity1.family_id, entity2.family_id, relation.family_id)
                if edge_key not in edges_seen:
                    edges_seen.add(edge_key)
                    is_matched = relation.family_id in matched_relation_ids
                    edges.append(build_edge_dict(
                        entity1.family_id, entity2.family_id, relation,
                        color='#FF6B6B' if is_matched else '#97C2FC',
                        width=3 if is_matched else 2,
                    ))

    return edges


# -- Snapshot helpers ---------------------------------------------------------

def build_snapshot_nodes(
    storage,
    entity_versions: Dict[str, str],
) -> List[Dict]:
    """Build node data for a snapshot request."""
    entity_abs_ids = list(entity_versions.values())
    entity_map = storage.get_entities_by_absolute_ids(entity_abs_ids) if entity_abs_ids else {}
    entity_fids = list(entity_versions)
    version_count_map = batch_preload_version_counts(storage, entity_fids)

    nodes_data = []
    for family_id, absolute_id in entity_versions.items():
        entity = entity_map.get(absolute_id)
        if entity:
            version_count = version_count_map.get(family_id, 1)
            label = build_entity_label(entity.name, version_count)
            nodes_data.append({
                'id': family_id, 'family_id': family_id,
                'absolute_id': absolute_id, 'label': label,
                'name': entity.name, 'content': entity.content,
                'event_time': entity.event_time.isoformat() if entity.event_time else None,
                'processed_time': entity.processed_time.isoformat() if entity.processed_time else None,
                'version_count': version_count,
            })
    return nodes_data


def build_snapshot_edges(
    storage,
    relation_versions: Dict[str, str],
) -> List[Dict]:
    """Build edge data for a snapshot request."""
    edges_data = []
    all_rel_entity_abs_ids: Set[str] = set()
    rel_versions_cache = {}

    for family_id, absolute_id in relation_versions.items():
        versions = storage.get_relation_versions(family_id)
        relation = next((r for r in versions if r.absolute_id == absolute_id), None)
        if relation:
            rel_versions_cache[family_id] = relation
            all_rel_entity_abs_ids.add(relation.entity1_absolute_id)
            all_rel_entity_abs_ids.add(relation.entity2_absolute_id)

    rel_entity_map = storage.get_entities_by_absolute_ids(list(all_rel_entity_abs_ids)) \
        if all_rel_entity_abs_ids else {}

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
                    'processed_time': relation.processed_time.isoformat() if relation.processed_time else None,
                })

    return edges_data
