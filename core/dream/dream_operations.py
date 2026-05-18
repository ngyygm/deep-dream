"""梦境操作步骤 — 从 orchestrator.py 提取的独立 dream 操作实现。

包含图探索、关联发现、LLM 判断、语义预过滤、梦境记录保存等步骤。
通过 Mixin 模式注入 DreamOrchestrator，保持主编排流程在 orchestrator.py 中。
"""

import logging
from collections import defaultdict
from concurrent.futures import as_completed
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from core.llm.json_repair import parse_json_response
from core.llm.prompts import (
    JUDGE_AND_GENERATE_RELATION_SYSTEM_PROMPT,
    JUDGE_AND_GENERATE_RELATION_DISCOVERY_PROMPT,
)
from core.dream._types import DreamConfig, _trunc

logger = logging.getLogger(__name__)


class DreamOperationsMixin:
    """Dream 操作步骤 Mixin — 提供图探索、关联发现、保存等实例方法。

    由 DreamOrchestrator 继承，期望子类提供：
    - self.storage
    - self.llm_client
    - self._history (DreamHistory)
    - self._pool (ThreadPoolExecutor)
    - self._searcher (GraphTraversalSearcher)
    - self.config (DreamConfig)
    """

    # ------------------------------------------------------------------
    # Step 2: 图探索
    # ------------------------------------------------------------------

    def _explore_graph(
        self,
        seeds: List[Dict[str, Any]],
        config: DreamConfig,
    ) -> tuple:
        """BFS 扩展获取邻居实体及已有关系。

        Returns:
            (entity_lookup, seen_ids, explored_list, relation_context)
            relation_context: family_id -> list of "neighbor_name — relation_snippet"
        """
        seed_family_ids = [s["family_id"] for s in seeds if s.get("family_id")]

        try:
            bfs_entities, bfs_relations, _ = self._searcher.bfs_expand_with_relations(
                seed_family_ids,
                max_depth=config.max_depth,
                max_nodes=config.max_explore_entities,
            )
        except Exception as exc:
            logger.warning("Dream: BFS遍历失败: %s", exc)
            bfs_entities, bfs_relations = [], []

        # 构建 entity_lookup + abs/fid/name 映射（单次遍历）
        entity_lookup: Dict[str, Dict[str, str]] = {}
        seen_ids: Set[str] = set()
        abs_to_fid: Dict[str, str] = {}
        fid_to_name: Dict[str, str] = {}

        for ent in bfs_entities:
            fid = ent.family_id
            abs_id = ent.absolute_id
            name = ent.name
            if fid:
                entity_lookup[fid] = {
                    "family_id": fid,
                    "name": name,
                    "content": _trunc(ent.content or '', 500),
                }
                seen_ids.add(fid)
            if abs_id:
                abs_to_fid[abs_id] = fid or ''
            if fid:
                fid_to_name[fid] = name

        # 补充种子自身
        for s in seeds:
            fid = s.get("family_id")
            if fid and fid not in entity_lookup:
                entity_lookup[fid] = {
                    "family_id": fid,
                    "name": s.get("name", ""),
                    "content": _trunc(s.get("content") or "", 500),
                }
                seen_ids.add(fid)
                fid_to_name[fid] = s.get("name", "")

        # 构建关系上下文: family_id -> ["neighbor_name — relation_snippet"]
        # 同时构建邻接表: family_id -> set of neighbor family_ids
        relation_context: Dict[str, List[str]] = defaultdict(list)
        adjacency: Dict[str, set] = defaultdict(set)
        for rel in bfs_relations:
            e1_abs = rel.entity1_absolute_id
            e2_abs = rel.entity2_absolute_id
            e1_fid = abs_to_fid.get(e1_abs)
            e2_fid = abs_to_fid.get(e2_abs)
            content_snippet = (rel.content or '')[:80]
            e1_name = fid_to_name.get(e1_fid, '') if e1_fid else ''
            e2_name = fid_to_name.get(e2_fid, '') if e2_fid else ''

            if e1_fid:
                relation_context[e1_fid].append(
                    f"{e2_name or e2_abs[:12]} — {content_snippet}"
                )
                if e2_fid:
                    adjacency[e1_fid].add(e2_fid)
            if e2_fid:
                relation_context[e2_fid].append(
                    f"{e1_name or e1_abs[:12]} — {content_snippet}"
                )
                if e1_fid:
                    adjacency[e2_fid].add(e1_fid)

        # 为每个种子构建 explored 信息
        explored: List[Dict[str, Any]] = []
        for seed in seeds:
            fid = seed.get("family_id")
            if not fid:
                continue
            # Use adjacency dict for O(1) neighbor lookup per seed
            neighbor_fids = adjacency.get(fid, set())
            neighbor_data = []
            for nfid in neighbor_fids:
                info = entity_lookup.get(nfid)
                if info:
                    neighbor_data.append({
                        "family_id": nfid,
                        "name": info["name"],
                        "content": _trunc(info["content"], 200),
                    })

            # Fallback for isolated seeds: use embedding similarity to find neighbors
            if not neighbor_data and hasattr(self.storage, 'search_entities_by_similarity'):
                try:
                    seed_name = seed.get("name", "")
                    seed_content = _trunc(seed.get("content") or "", 200)
                    sim_results = self.storage.search_entities_by_similarity(
                        query_name=seed_name,
                        query_content=seed_content,
                        max_results=config.max_neighbors_per_seed,
                    )
                    for ent in sim_results:
                        efid = ent.family_id
                        if efid and efid != fid:
                            info = {
                                "family_id": efid,
                                "name": ent.name,
                                "content": _trunc(ent.content or '', 200),
                            }
                            neighbor_data.append(info)
                            if efid not in entity_lookup:
                                entity_lookup[efid] = {
                                    "family_id": efid,
                                    "name": info["name"],
                                    "content": info["content"],
                                }
                                seen_ids.add(efid)
                    if neighbor_data:
                        logger.info("Dream: 孤立种子 %s 通过embedding找到 %d 个候选邻居", seed_name, len(neighbor_data))
                except Exception as exc:
                    logger.warning("Dream: embedding fallback failed for seed %s: %s", fid, exc)

            explored.append({
                "seed": {"family_id": fid, "name": seed.get("name", "")},
                "neighbors": neighbor_data[:20],
                "neighbor_count": len(neighbor_data),
            })

        return entity_lookup, seen_ids, explored, relation_context

    # ------------------------------------------------------------------
    # Step 3: 关联发现（并发 LLM 判断）
    # ------------------------------------------------------------------

    def _discover_relations(
        self,
        seeds: List[Dict[str, Any]],
        explored: List[Dict[str, Any]],
        entity_lookup: Dict[str, Dict[str, str]],
        cycle_id: str,
        config: DreamConfig,
        relation_context: Optional[Dict[str, List[str]]] = None,
    ) -> tuple:
        """并发发现实体间的隐含关联。

        Generates two kinds of candidate pairs:
        1. Cross-neighbor pairs: entities from *different* seeds' neighborhoods
           that may not yet be connected (primary discovery mechanism).
        2. Seed-to-neighbor pairs: only included when discovery_mode is true,
           as these usually already have relations.

        Returns:
            (relations_created, pairs_checked)
        """
        # 收集所有待检查的配对（跳过历史已检查的）
        # Strategy: cross-neighbor pairs (A's neighbor ↔ B's neighbor) instead of
        # seed-to-own-neighbor (which always get filtered by existing-pairs check).
        pairs: List[tuple] = []
        skipped_by_history = 0
        involved_fids_set: set = set()
        _history = self._history
        _seen_pair_set: set = set()

        # Build per-seed neighbor lists
        seed_neighbor_map: Dict[str, List[Dict]] = {}
        for exp in explored:
            seed_info = exp["seed"]
            seed_fid = seed_info["family_id"]
            neighbors = exp["neighbors"][:config.max_neighbors_per_seed]
            seed_neighbor_map[seed_fid] = neighbors

        seed_fids = list(seed_neighbor_map.keys())

        # Generate seed-to-seed pairs — especially valuable for cross_community
        # where seeds come from different communities and may have hidden connections
        for i in range(len(seed_fids)):
            for j in range(i + 1, len(seed_fids)):
                fid_i, fid_j = seed_fids[i], seed_fids[j]
                pair_key = (min(fid_i, fid_j), max(fid_i, fid_j))
                if pair_key in _seen_pair_set:
                    continue
                _seen_pair_set.add(pair_key)
                if _history.was_checked(fid_i, fid_j):
                    skipped_by_history += 1
                    continue
                name_i = entity_lookup.get(fid_i, {}).get("name", fid_i)
                name_j = entity_lookup.get(fid_j, {}).get("name", fid_j)
                pairs.append((fid_i, name_i, fid_j, name_j))
                involved_fids_set.add(fid_i)
                involved_fids_set.add(fid_j)

        # Generate cross-seed pairs: neighbor of seed_i ↔ neighbor of seed_j
        for i in range(len(seed_fids)):
            for j in range(i + 1, len(seed_fids)):
                fid_i, fid_j = seed_fids[i], seed_fids[j]
                for nb_i in seed_neighbor_map[fid_i]:
                    for nb_j in seed_neighbor_map[fid_j]:
                        nfid_i = nb_i["family_id"]
                        nfid_j = nb_j["family_id"]
                        if nfid_i == nfid_j:
                            continue
                        pair_key = (min(nfid_i, nfid_j), max(nfid_i, nfid_j))
                        if pair_key in _seen_pair_set:
                            continue
                        _seen_pair_set.add(pair_key)
                        if _history.was_checked(nfid_i, nfid_j):
                            skipped_by_history += 1
                            continue
                        pairs.append((nfid_i, nb_i["name"], nfid_j, nb_j["name"]))
                        involved_fids_set.add(nfid_i)
                        involved_fids_set.add(nfid_j)

        # Also add seed-to-neighbor from DIFFERENT seeds' neighborhoods
        for i in range(len(seed_fids)):
            for j in range(len(seed_fids)):
                if i == j:
                    continue
                fid_i = seed_fids[i]
                for nb in seed_neighbor_map[seed_fids[j]]:
                    nfid = nb["family_id"]
                    if fid_i == nfid:
                        continue
                    pair_key = (min(fid_i, nfid), max(fid_i, nfid))
                    if pair_key in _seen_pair_set:
                        continue
                    _seen_pair_set.add(pair_key)
                    if _history.was_checked(fid_i, nfid):
                        skipped_by_history += 1
                        continue
                    seed_name = entity_lookup.get(fid_i, {}).get("name", fid_i)
                    pairs.append((fid_i, seed_name, nfid, nb["name"]))
                    involved_fids_set.add(fid_i)
                    involved_fids_set.add(nfid)


        # Intra-neighborhood "friend-of-friend" pairs within same seed
        # Entities that share a neighbor but aren't directly connected
        for fid in seed_fids:
            nbs = seed_neighbor_map[fid]
            for a in range(len(nbs)):
                for b in range(a + 1, len(nbs)):
                    nfid_a, nfid_b = nbs[a]["family_id"], nbs[b]["family_id"]
                    if nfid_a == nfid_b:
                        continue
                    pair_key = (min(nfid_a, nfid_b), max(nfid_a, nfid_b))
                    if pair_key in _seen_pair_set:
                        continue
                    _seen_pair_set.add(pair_key)
                    if _history.was_checked(nfid_a, nfid_b):
                        skipped_by_history += 1
                        continue
                    pairs.append((nfid_a, nbs[a]["name"], nfid_b, nbs[b]["name"]))
                    involved_fids_set.add(nfid_a)
                    involved_fids_set.add(nfid_b)

        if skipped_by_history:
            logger.info("Dream: 跳过 %d 对历史已检查的配对", skipped_by_history)

        if not pairs:
            return [], 0

        # 语义预过滤：跳过语义相似度过低的配对
        pairs = self._prefilter_pairs_by_similarity(pairs, entity_lookup, config, involved_fids_set)

        if not pairs:
            return [], 0

        # 批量预取所有配对的已有关系，避免 _judge_pair 中逐对查询
        try:
            pair_keys = [(p[0], p[2]) for p in pairs]
            existing_map = self.storage.get_relations_by_entity_pairs(pair_keys)
            existing_pairs = {k for k, v in existing_map.items() if v}
        except Exception as exc:
            logger.debug("Dream: 批量关系预取失败，回退逐对查询: %s", exc)
            existing_pairs = None

        relations_created: List[Dict[str, Any]] = []
        pairs_checked = 0
        pair_errors = 0

        # Batch pre-fetch all entities involved in pairs (avoid N+1 in save_dream_relation)
        _all_pair_fids = involved_fids_set  # already computed above (superset of filtered pairs' fids)
        _entity_lookup = {}
        if _all_pair_fids:
            try:
                _entity_lookup = self.storage.get_entities_by_family_ids(list(_all_pair_fids))
            except Exception:
                pass

        # Backfill entity_lookup for any fids missing from BFS-built lookup
        _missing_fids = involved_fids_set - set(entity_lookup.keys())
        if _missing_fids:
            try:
                _backfill = self.storage.get_entities_by_family_ids(list(_missing_fids))
                for fid, ent in _backfill.items():
                    if fid not in entity_lookup:
                        entity_lookup[fid] = {
                            "family_id": fid,
                            "name": ent.name if hasattr(ent, "name") else ent.get("name", ""),
                            "content": _trunc(
                                (ent.content or "") if hasattr(ent, "content") else ent.get("content", ""),
                                500,
                            ),
                        }
            except Exception:
                pass

        # 使用共享线程池并发判断
        _pool = self._pool
        futures = {}
        for pair in pairs:
            seed_fid, seed_name, nb_fid, nb_name = pair
            future = _pool.submit(
                self._judge_pair,
                seed_fid, seed_name, nb_fid, nb_name, config,
                entity_lookup, existing_pairs, relation_context,
            )
            futures[future] = pair

        early_break = False
        for future in as_completed(futures):
            pair = futures[future]
            seed_fid, seed_name, nb_fid, nb_name = pair
            # Always mark checked to prevent re-checking in future cycles
            self._history.mark_checked(seed_fid, nb_fid, cycle_id)
            pairs_checked += 1

            if early_break:
                # Already hit max_relations — just drain futures and mark history
                continue
            if len(relations_created) >= config.max_relations:
                early_break = True
                continue

            try:
                result = future.result()
                if result is None:
                    continue

                # 保存 dream relation
                confidence = result["confidence"]
                if confidence < config.min_confidence:
                    continue

                reasoning = f"梦境发现：{seed_name} 与 {nb_name} 存在潜在关联（策略: {config.strategy}）"

                save_result = self.storage.save_dream_relation(
                    entity1_id=seed_fid,
                    entity2_id=nb_fid,
                    content=result["content"],
                    confidence=confidence,
                    reasoning=reasoning,
                    dream_cycle_id=cycle_id,
                )
                relations_created.append({
                    "entity1_id": seed_fid,
                    "entity1_name": seed_name,
                    "entity2_id": nb_fid,
                    "entity2_name": nb_name,
                    "content": result["content"],
                    "confidence": confidence,
                    "result": save_result,
                })
            except Exception as exc:
                pair_errors += 1
                logger.warning("Dream: 检查关系 %s↔%s 时出错: %s", pair[0], pair[2], exc)

        return relations_created, pairs_checked, pair_errors

    def _prefilter_pairs_by_similarity(
        self,
        pairs: List[tuple],
        entity_lookup: Dict[str, Dict[str, str]],
        config: DreamConfig,
        involved_fids: set = None,
    ) -> List[tuple]:
        """基于 embedding 余弦相似度预过滤配对，跳过语义不相关的配对。

        当 min_pair_similarity > 0 且 embedding 客户端可用时，
        批量计算所有实体的 embedding 并过滤低相似度配对。
        无法获取 embedding 的实体保留（不过滤）。
        """
        if config.min_pair_similarity <= 0:
            return pairs

        ec = getattr(self.storage, 'embedding_client', None)
        if not ec or not getattr(ec, 'is_available', lambda: False)():
            return pairs

        # Use pre-computed fids if provided, otherwise collect from pairs
        if involved_fids is None:
            involved_fids = set()
            for seed_fid, _, nb_fid, _ in pairs:
                involved_fids.add(seed_fid)
                involved_fids.add(nb_fid)

        # 批量计算 embedding
        texts = []
        fid_list = []
        for fid in involved_fids:
            info = entity_lookup.get(fid)
            if info:
                text = f"{info.get('name', '')}: {info.get('content', '')}"
                texts.append(text)
                fid_list.append(fid)

        if not texts:
            return pairs

        try:
            embeddings = ec.encode(texts)
            if embeddings is None:
                return pairs
            fid_to_emb: Dict[str, np.ndarray] = {}
            emb_matrix = np.array(embeddings[:len(fid_list)], dtype=np.float32)
            norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-9)
            emb_normed = emb_matrix / norms
            for i, fid in enumerate(fid_list):
                fid_to_emb[fid] = emb_normed[i]
        except Exception as exc:
            logger.warning("Dream: embedding 预计算失败，跳过语义过滤: %s", exc)
            return pairs

        # 过滤配对
        filtered = []
        for pair in pairs:
            seed_fid, _, nb_fid, _ = pair
            e1 = fid_to_emb.get(seed_fid)
            e2 = fid_to_emb.get(nb_fid)
            if e1 is not None and e2 is not None:
                similarity = float(np.dot(e1, e2))
                if similarity < config.min_pair_similarity:
                    continue
            # 无 embedding 的实体保留
            filtered.append(pair)

        if len(filtered) < len(pairs):
            logger.info(
                "Dream: 语义预过滤 %d→%d 对 (阈值=%.2f)",
                len(pairs), len(filtered), config.min_pair_similarity,
            )

        return filtered

    def _judge_pair(
        self,
        seed_fid: str,
        seed_name: str,
        nb_fid: str,
        nb_name: str,
        config: DreamConfig,
        entity_lookup: Optional[Dict[str, Dict[str, str]]] = None,
        existing_pairs: Optional[Set[Tuple[str, str]]] = None,
        relation_context: Optional[Dict[str, List[str]]] = None,
    ) -> Optional[Dict[str, Any]]:
        """判断一对实体是否存在隐含关联。

        Returns:
            None 表示无关联，dict 包含 content 和 confidence 表示有关联。
        """
        # 检查是否已有关系（优先使用批量预取结果）
        if existing_pairs is not None:
            pair_key = (seed_fid, nb_fid)
            rev_key = (nb_fid, seed_fid)
            if pair_key in existing_pairs or rev_key in existing_pairs:
                return None
        else:
            try:
                existing = self.storage.get_relations_by_entities(seed_fid, nb_fid)
                if existing:
                    return None
            except Exception as exc:
                logger.debug("Dream: existing relation check failed for %s↔%s: %s", seed_fid, nb_fid, exc)

        # 优先从 entity_lookup 获取实体详情，避免重复 DB 查询
        if entity_lookup:
            seed_info = entity_lookup.get(seed_fid)
            nb_info = entity_lookup.get(nb_fid)
            if not seed_info or not nb_info:
                return None
            seed_name = seed_info.get("name", seed_name)
            seed_content = seed_info.get("content", "")
            nb_name = nb_info.get("name", nb_name)
            nb_content = nb_info.get("content", "")
        else:
            seed_entity = self.storage.get_entity_by_family_id(seed_fid)
            nb_entity = self.storage.get_entity_by_family_id(nb_fid)
            if not seed_entity or not nb_entity:
                return None
            seed_name = seed_entity.name
            seed_content = _trunc(seed_entity.content or "", 500)
            nb_name = nb_entity.name
            nb_content = _trunc(nb_entity.content or "", 500)

        # 构建图拓扑上下文（每个实体已有的关系）
        topology_lines = []
        if relation_context:
            seed_rels = relation_context.get(seed_fid, [])
            nb_rels = relation_context.get(nb_fid, [])
            if seed_rels:
                topology_lines.append(
                    f"实体A的已知关联:\n" + "\n".join(f"  - {r}" for r in seed_rels[:8])
                )
            if nb_rels:
                topology_lines.append(
                    f"实体B的已知关联:\n" + "\n".join(f"  - {r}" for r in nb_rels[:8])
                )
        topology_block = ("\n\n".join(topology_lines) + "\n\n") if topology_lines else ""

        # LLM 判断 + 生成（单次调用）
        _system_prompt = (JUDGE_AND_GENERATE_RELATION_DISCOVERY_PROMPT
                          if config.discovery_mode
                          else JUDGE_AND_GENERATE_RELATION_SYSTEM_PROMPT)
        judge_messages = [
            {"role": "system", "content": _system_prompt},
            {"role": "user", "content": (
                f"实体A: {seed_name}\n描述: {seed_content}\n\n"
                f"实体B: {nb_name}\n描述: {nb_content}\n\n"
                f"{topology_block}"
                "判断这两个实体之间是否存在明确的、有意义的关联。如果存在，同时生成关系描述。"
            )},
        ]
        judge_obj, _ = self.llm_client.call_llm_until_json_parses(
            judge_messages,
            parse_fn=parse_json_response,
            json_parse_retries=1,
            timeout=config.llm_timeout,
        )
        if not judge_obj.get("need_create", False):
            return None

        judge_confidence = float(judge_obj.get("confidence", 0.5))
        rel_content = (judge_obj.get("content") or "").strip()
        if not rel_content or len(rel_content) < 10:
            return None

        return {
            "content": rel_content,
            "confidence": max(0.1, min(1.0, judge_confidence)),
        }

    # ------------------------------------------------------------------
    # Step 4: 保存梦境记录
    # ------------------------------------------------------------------

    def _save_episode(
        self,
        cycle_id: str,
        cycle_summary: str,
        seen_ids: Set[str],
        relations_created: List[Dict[str, Any]],
        config: DreamConfig,
        start_time: float = None,
        end_time: float = None,
    ) -> None:
        """保存梦境周期记录。先写 DreamLog（确保状态可查），再写 Episode。"""
        from datetime import datetime as _dt
        from types import SimpleNamespace
        st = _dt.fromtimestamp(start_time) if start_time else None
        et = _dt.fromtimestamp(end_time) if end_time else None

        # Step A: DreamLog — must succeed so dream_status/dream_logs work
        try:
            now = _dt.now()
            report = SimpleNamespace(
                cycle_id=cycle_id,
                graph_id=self.storage._graph_id,
                start_time=st or now,
                end_time=et or now,
                status="completed",
                narrative=cycle_summary[:2000],
                insights=[],
                new_connections=[
                    r.get("result", {}).get("family_id", "")
                    for r in relations_created if r.get("result")
                ],
                consolidations=[],
                strategy=config.strategy,
                entities_examined=min(len(seen_ids), 50),
                relations_created=len(relations_created),
                episode_ids=[],
            )
            self.storage.save_dream_log(report)
        except Exception as exc:
            logger.error("Dream: save_dream_log 失败: %s", exc)

        # Step B: Episode + mentions — best-effort, non-blocking
        try:
            self.storage.save_dream_episode(
                content=cycle_summary,
                entities_examined=list(seen_ids)[:50],
                relations_created=[
                    r.get("result", {}).get("family_id", "")
                    for r in relations_created if r.get("result")
                ],
                strategy_used=config.strategy,
                dream_cycle_id=cycle_id,
                relations_created_count=len(relations_created),
                cycle_start_time=st,
                cycle_end_time=et,
            )
        except Exception as exc:
            logger.warning("Dream: save_dream_episode 失败（DreamLog已保存）: %s", exc)
