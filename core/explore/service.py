"""One implementation of Deep-Dream exploration for CLI, agents, and evals."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from core.cli._helpers import (
    concept_source_evidence,
    evidence_cards,
    expand_query_terms,
    relation_evidence,
    search_document_terms,
)


def _dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "__dict__"):
        return dict(vars(value))
    return {}


@dataclass(frozen=True)
class ExploreOptions:
    role: str | None = None
    limit: int = 20
    threshold: float = 0.2
    file_limit: int = 20
    per_term_file_limit: int = 5
    semantic_queries: int = 5
    min_semantic_score: float = 0.0
    evidence_limit: int = 12
    neighbor_seeds: int = 3
    neighbor_limit: int = 50
    neighbor_evidence_limit: int = 20
    depth: int = 1
    relation_seed_count: int = 5
    relation_pair_limit: int = 8
    relation_evidence_limit: int = 10


class _ScopeFilter:
    """Resolve every result back to active source documents before exposure."""

    def __init__(self, storage: Any, allowed_document_ids: Iterable[str] | None):
        self.storage = storage
        self.allowed = set(allowed_document_ids or [])
        self.enabled = allowed_document_ids is not None
        self._episode_docs: dict[str, str] = {}
        self._version_docs: dict[str, str] = {}

    def episode_allowed(self, episode_id: str) -> bool:
        if not self.enabled:
            return True
        if not episode_id:
            return False
        if episode_id not in self._episode_docs:
            row = self.storage._conn().execute(
                "SELECT document_id FROM episodes WHERE episode_id = ? AND status = 'active'",
                (episode_id,),
            ).fetchone()
            self._episode_docs[episode_id] = str(row[0]) if row else ""
        return self._episode_docs[episode_id] in self.allowed

    def version_allowed(self, version_id: str) -> bool:
        if not self.enabled:
            return True
        if not version_id:
            return False
        if version_id not in self._version_docs:
            row = self.storage._conn().execute(
                "SELECT document_id FROM document_versions WHERE document_version_id = ? AND status = 'active'",
                (version_id,),
            ).fetchone()
            self._version_docs[version_id] = str(row[0]) if row else ""
        return self._version_docs[version_id] in self.allowed

    def evidence(self, row: dict[str, Any]) -> bool:
        episode_id = str(row.get("episode_version_id") or row.get("episode_id") or "")
        if episode_id:
            return self.episode_allowed(episode_id)
        version_id = str(row.get("document_version_id") or "")
        return self.version_allowed(version_id) if version_id else not self.enabled

    def file_hit(self, row: dict[str, Any]) -> bool:
        doc = row.get("document") or {}
        return self.version_allowed(str(doc.get("document_version_id") or ""))


def explore_memory(
    storage: Any,
    question: str,
    *,
    explicit_terms: str | Iterable[str] | None = None,
    expand_query: bool = True,
    options: ExploreOptions | None = None,
    allowed_document_ids: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Run raw-file, episode, semantic, provenance, graph, and relation search.

    ``allowed_document_ids`` is security-sensitive: when provided, every
    observation returned to an agent must trace to one of those active source
    documents.
    """
    opts = options or ExploreOptions()
    if explicit_terms is not None and not isinstance(explicit_terms, str):
        explicit_terms = ",".join(str(term) for term in explicit_terms if str(term).strip())
    query_terms = expand_query_terms(question, explicit_terms)
    if not expand_query:
        query_terms = query_terms[:1]
    scope = _ScopeFilter(storage, allowed_document_ids)

    # Specific phrases should be searched before high-frequency names; the
    # original full question remains first for exact sentence matches.
    file_terms = query_terms[:1] + sorted(
        query_terms[1:],
        key=lambda row: (-len(str(row.get("term") or "").split()), -len(str(row.get("term") or ""))),
    )
    file_hits = [
        row for row in search_document_terms(
            storage, file_terms,
            per_term_limit=opts.per_term_file_limit,
            total_limit=opts.file_limit,
        )
        if scope.file_hit(row)
    ]

    # Literal episode FTS is a separate channel from semantic concept search.
    episode_hits: list[dict[str, Any]] = []
    episode_seen: set[str] = set()
    for term_info in query_terms[:opts.semantic_queries]:
        for raw in storage.search_concepts_by_bm25(term_info["term"], limit=opts.limit):
            row = _dict(raw)
            episode_id = str(row.get("episode_id") or "")
            if not episode_id or episode_id in episode_seen or not scope.episode_allowed(episode_id):
                continue
            episode_seen.add(episode_id)
            row["matched_query"] = term_info["term"]
            row["query_source"] = term_info.get("source", "expanded")
            episode_hits.append(row)
            if len(episode_hits) >= opts.limit:
                break
        if len(episode_hits) >= opts.limit:
            break

    semantic_results: list[dict[str, Any]] = []
    semantic_seen: set[str] = set()
    for term_info in query_terms[:opts.semantic_queries]:
        semantic = storage.agent_semantic_search(
            term_info["term"], role=opts.role, top_k=opts.limit, threshold=opts.threshold,
        )
        for raw in semantic.get("results", []):
            row = _dict(raw)
            score = row.get("score")
            if score is not None and float(score or 0.0) < opts.min_semantic_score:
                continue
            family_id = str(row.get("family_id") or "")
            if not family_id or family_id in semantic_seen:
                continue
            semantic_seen.add(family_id)
            row["matched_query"] = term_info["term"]
            row["query_source"] = term_info.get("source", "expanded")
            semantic_results.append(row)
            if len(semantic_results) >= opts.limit:
                break
        if len(semantic_results) >= opts.limit:
            break
    semantic_results.sort(key=lambda row: float(row.get("score") or 0.0), reverse=True)

    concept_ids = [str(row.get("family_id")) for row in semantic_results if row.get("family_id")]
    source_evidence = [
        row for row in concept_source_evidence(storage, concept_ids, limit=max(opts.limit * 5, 50))
        if scope.evidence(row)
    ][:opts.limit]
    if scope.enabled:
        sourced = {str(row.get("target_family_id") or "") for row in source_evidence}
        semantic_results = [row for row in semantic_results if str(row.get("family_id") or "") in sourced]
        concept_ids = [str(row.get("family_id")) for row in semantic_results]

    neighbors: list[dict[str, Any]] = []
    neighbor_seen: set[str] = set()
    for family_id in concept_ids[:opts.neighbor_seeds]:
        try:
            raw_neighbors = storage.get_concept_neighbors(
                family_id, max_depth=opts.depth, max_results=opts.neighbor_limit,
            )
        except Exception:
            continue
        for raw in raw_neighbors:
            row = _dict(raw)
            neighbor_id = str(row.get("family_id") or row.get("target_family_id") or "")
            if not neighbor_id or neighbor_id in neighbor_seen:
                continue
            if scope.enabled:
                proof = concept_source_evidence(storage, [neighbor_id], limit=20)
                if not any(scope.evidence(item) for item in proof):
                    continue
            neighbor_seen.add(neighbor_id)
            neighbors.append(row)
            if len(neighbors) >= opts.neighbor_limit:
                break
        if len(neighbors) >= opts.neighbor_limit:
            break

    neighbor_ids = [
        str(row.get("family_id") or row.get("target_family_id") or "")
        for row in neighbors
    ]
    neighbor_evidence = [
        row
        for row in concept_source_evidence(
            storage, neighbor_ids, limit=max(opts.neighbor_evidence_limit * 5, 50)
        )
        if scope.evidence(row)
    ][:opts.neighbor_evidence_limit]

    relation_samples: list[dict[str, Any]] = []
    relation_pairs: list[tuple[str, str]] = []
    for index, left in enumerate(concept_ids[:opts.relation_seed_count]):
        for right in concept_ids[index + 1:opts.relation_seed_count]:
            if left != right:
                relation_pairs.append((left, right))
    for left, right in relation_pairs[:opts.relation_pair_limit]:
        for raw in relation_evidence(storage, left, right, limit=opts.relation_evidence_limit):
            row = _dict(raw)
            if not scope.evidence(row):
                continue
            row["query_pair"] = [left, right]
            relation_samples.append(row)
            if len(relation_samples) >= opts.relation_evidence_limit:
                break
        if len(relation_samples) >= opts.relation_evidence_limit:
            break

    cards = evidence_cards(file_hits, source_evidence, query_terms, limit=opts.evidence_limit)
    return {
        "question": question,
        "query_terms": query_terms,
        "file_hits": file_hits,
        "episode_hits": episode_hits,
        "semantic_hits": semantic_results,
        "semantic_total": len(semantic_results),
        "episode_ids": [str(row.get("episode_id") or "") for row in episode_hits],
        "source_evidence": source_evidence,
        "evidence_cards": cards,
        "neighbors": neighbors,
        "neighbor_evidence": neighbor_evidence,
        "relation_evidence": relation_samples,
        "depth": opts.depth,
        "coverage": {
            "file_hits": len(file_hits),
            "episode_hits": len(episode_hits),
            "semantic_hits": len(semantic_results),
            "source_evidence": len(source_evidence),
            "evidence_cards": len(cards),
            "neighbors": len(neighbors),
            "neighbor_evidence": len(neighbor_evidence),
            "relation_evidence": len(relation_samples),
            "relation_pairs_checked": min(len(relation_pairs), opts.relation_pair_limit),
        },
    }
