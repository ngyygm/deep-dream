"""Benchmark adapter over the shared Deep-Dream exploration service.

``legacy`` preserves the original session-level RRF behavior. ``hybrid-v2``
uses the same candidate generators but ranks source turns deterministically,
expands only the selected turn neighbourhoods, and enforces an evidence budget.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Any, Iterable

import numpy as np

from core.explore import ExploreOptions, explore_memory

from .datasets import MemorySession


HYBRID_V2_WEIGHTS = {
    "rrf": 0.40,
    "semantic": 0.35,
    "lexical": 0.20,
    "phrase": 0.05,
}
HYBRID_V2_MMR_LAMBDA = 0.85
HYBRID_V2_SESSION_AGGREGATION = (1.0, 0.15, 0.05)
HYBRID_V2_SESSION_RANK_FUSION = {"turn_aggregate": 0.10, "legacy_guard": 0.90}
HYBRID_V2_TURN_RANK_FUSION = {"hybrid_features": 0.35, "legacy_guard": 0.65}

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "but", "by",
    "can", "could", "did", "do", "does", "for", "from", "had", "has", "have",
    "he", "her", "hers", "him", "his", "how", "i", "if", "in", "into", "is",
    "it", "its", "likely", "may", "might", "my", "of", "on", "or", "our", "she",
    "should", "that", "the", "their", "them", "they", "this", "to", "was", "were",
    "what", "when", "where", "which", "who", "why", "will", "with", "would", "you",
    "your",
}

_REFERENTIAL_BRIDGE_PHRASES = ("home country", "home town", "hometown")


def _tokens(text: str) -> set[str]:
    return {
        token.lower() for token in re.findall(r"[\w']+", text or "", flags=re.UNICODE)
        if len(token) > 1
    }


def _stem_token(token: str) -> str:
    token = token.lower().strip("'")
    if token.endswith("'s"):
        token = token[:-2]
    if len(token) > 4 and token.endswith("ved"):
        token = token[:-1]
    elif len(token) > 5 and token.endswith("ing"):
        token = token[:-3]
    elif len(token) > 4 and token.endswith("ed"):
        token = token[:-2]
    elif (
        len(token) > 4
        and token.endswith("s")
        and not token.endswith(("ss", "us", "is"))
    ):
        token = token[:-1]
    return token


def _content_tokens(text: str) -> list[str]:
    return [
        _stem_token(token) for token in re.findall(r"[\w']+", text or "", flags=re.UNICODE)
        if len(token) > 2 and token.lower() not in _STOPWORDS
    ]


def _phrase_normalize(text: str) -> str:
    return " ".join(_content_tokens(text))


def extract_query_terms(question: str, *, minimum: int = 3, maximum: int = 8) -> list[dict[str, str]]:
    """Deterministically derive benchmark-safe terms without category leakage."""
    question = re.sub(r"\s+", " ", question or "").strip()
    if not question:
        return []
    candidates: list[tuple[str, str]] = [(question, "original")]

    # Quoted strings, named spans, and date expressions are the most precise.
    for value in re.findall(r'["“]([^"”]{2,80})["”]', question):
        candidates.append((value.strip(), "quoted"))
    for value in re.findall(
        r"\b(?:Dr\.\s+)?[A-Z][\w'.-]*(?:\s+(?:[A-Z][\w'.-]*|of|the|and)){0,3}", question
    ):
        value = value.strip().rstrip("?.!,")
        parts = value.split()
        if len(parts) > 1 and parts[0].casefold() in {
            "did", "does", "do", "is", "was", "were", "are", "has", "have", "had",
            "can", "could", "would", "when", "what", "where", "which", "who", "how", "why",
        }:
            value = " ".join(parts[1:])
        if value.lower() not in {"what", "when", "where", "which", "who", "how", "why"}:
            candidates.append((value, "named"))
    for value in re.findall(
        r"\b(?:yesterday|today|tomorrow|last\s+(?:week|month|year|monday|tuesday|wednesday|thursday|friday|saturday|sunday)|"
        r"\d{1,2}(?:st|nd|rd|th)?\s+[A-Z][a-z]+(?:\s+\d{4})?)\b",
        question,
        flags=re.IGNORECASE,
    ):
        candidates.append((value.strip(), "temporal"))

    content = _content_tokens(question)
    # Preserve a few contiguous content phrases before falling back to unigrams.
    for size in (3, 2):
        for index in range(max(0, len(content) - size + 1)):
            candidates.append((" ".join(content[index:index + size]), "phrase"))
    for token in content:
        candidates.append((token, "keyword"))

    seen: set[str] = set()
    result: list[dict[str, str]] = []
    for value, source in candidates:
        normalized = re.sub(r"\s+", " ", value).strip()
        key = normalized.casefold()
        if not normalized or key in seen:
            continue
        seen.add(key)
        result.append({"term": normalized, "source": source})
        if len(result) >= maximum:
            break
    # Short questions legitimately have fewer than three distinct useful terms.
    if len(result) < minimum:
        for token in re.findall(r"[\w']+", question, flags=re.UNICODE):
            key = token.casefold()
            if len(token) > 1 and key not in seen:
                seen.add(key)
                result.append({"term": token, "source": "fallback"})
                if len(result) >= min(minimum, maximum):
                    break
    return result


def _turn_rows(text: str) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for line in (text or "").splitlines():
        match = re.match(r"^\[([^\]]+)\]\s*(.*)$", line.strip())
        if match:
            rows.append((match.group(1), match.group(2)))
    return rows


def _estimated_tokens(text: str) -> int:
    return max(1, math.ceil(len((text or "").encode("utf-8")) / 4))


def _minmax(values: dict[str, float]) -> dict[str, float]:
    if not values:
        return {}
    low, high = min(values.values()), max(values.values())
    if high <= low:
        return {key: (1.0 if high > 0 else 0.0) for key in values}
    return {key: (value - low) / (high - low) for key, value in values.items()}


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 0:
        return 0.0
    return float(np.dot(left, right) / denominator)


ALL_RETRIEVAL_CHANNELS: tuple[str, ...] = (
    "raw-document",
    "episode-bm25",
    "semantic-provenance",
    "graph-neighbor",
    "relation-evidence",
)


@dataclass(frozen=True)
class HybridRetrievalConfig:
    candidate_k: int = 30
    context_k: int = 5
    evidence_token_budget: int = 1600
    neighbor_turns: int = 1
    # X7 provenance ablation: which retrieval channels may contribute
    # candidates. ``None`` (default) means all five channels, preserving
    # backward compatibility with every pre-existing caller. A tuple of
    # channel names restricts the candidate set to those channels only, so
    # each arm of the source-only -> overlay -> evidence-gate ablation
    # toggles a channel independently. The chosen set is part of the
    # fingerprint below, so each arm's profile hash differs.
    enabled_channels: tuple[str, ...] | None = None

    def resolved_channels(self) -> tuple[str, ...]:
        return self.enabled_channels if self.enabled_channels else ALL_RETRIEVAL_CHANNELS

    def payload(self) -> dict[str, Any]:
        return {
            "profile": "hybrid-v2",
            "candidate_k": self.candidate_k,
            "context_k": self.context_k,
            "evidence_token_budget": self.evidence_token_budget,
            "neighbor_turns": self.neighbor_turns,
            "enabled_channels": list(self.resolved_channels()),
            "weights": HYBRID_V2_WEIGHTS,
            "mmr_lambda": HYBRID_V2_MMR_LAMBDA,
            "session_aggregation": list(HYBRID_V2_SESSION_AGGREGATION),
            "session_rank_fusion": HYBRID_V2_SESSION_RANK_FUSION,
            "turn_rank_fusion": HYBRID_V2_TURN_RANK_FUSION,
            "referential_bridge_phrases": list(_REFERENTIAL_BRIDGE_PHRASES),
        }

    def fingerprint(self) -> str:
        return hashlib.sha256(
            json.dumps(self.payload(), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


class UnifiedRetriever:
    """Source-grounded legacy RRF and deterministic turn-level hybrid retrieval."""

    CHANNEL_WEIGHTS = {
        "raw-document": 1.00,
        "episode-bm25": 0.95,
        "semantic-provenance": 0.80,
        "graph-neighbor": 0.55,
        "relation-evidence": 0.70,
    }

    def __init__(
        self,
        storage: Any,
        document_to_session: dict[str, str],
        sessions: list[MemorySession],
        *,
        allowed_document_ids: Iterable[str] | None = None,
    ):
        self.storage = storage
        self.document_to_session = dict(document_to_session)
        self.session_to_document = {value: key for key, value in document_to_session.items()}
        self.sessions = {session.session_id: session for session in sessions}
        self.allowed_document_ids = set(allowed_document_ids or document_to_session)
        self._episode_cache: dict[str, dict[str, Any]] = {}
        self._session_turns = {
            session.session_id: _turn_rows(session.text) for session in sessions
        }
        self._turn_to_session = {
            turn_id: session_id
            for session_id, rows in self._session_turns.items()
            for turn_id, _ in rows
        }

    def _episode(self, episode_id: str) -> dict[str, Any]:
        if not episode_id:
            return {}
        if episode_id not in self._episode_cache:
            row = self.storage._conn().execute(
                """SELECT episode_id, document_id, source_text, memory_text
                   FROM episodes WHERE episode_id = ? AND status = 'active'""",
                (episode_id,),
            ).fetchone()
            self._episode_cache[episode_id] = dict(row) if row else {}
        return self._episode_cache[episode_id]

    def _document_for_version(self, version_id: str) -> str:
        if not version_id:
            return ""
        row = self.storage._conn().execute(
            "SELECT document_id FROM document_versions WHERE document_version_id = ? AND status = 'active'",
            (version_id,),
        ).fetchone()
        return str(row[0]) if row else ""

    def _source_evidence(
        self,
        *,
        channel: str,
        episode_id: str = "",
        document_id: str = "",
        source_text: str = "",
        query: str,
        hints: Iterable[str] = (),
        rank: int,
        include_unmatched_turns: bool = False,
    ) -> list[dict[str, Any]]:
        episode = self._episode(episode_id) if episode_id else {}
        document_id = document_id or str(episode.get("document_id") or "")
        if document_id not in self.allowed_document_ids:
            return []
        session_id = self.document_to_session.get(document_id, "")
        if not session_id or session_id not in self.sessions:
            return []
        source_text = source_text or str(episode.get("source_text") or "")
        needle = _tokens(" ".join([query, *[str(hint) for hint in hints]]))
        candidates: list[tuple[float, str, str]] = []
        allowed_turns = set(self.sessions[session_id].turn_ids)
        for turn_id, turn_text in _turn_rows(source_text):
            if turn_id not in allowed_turns:
                continue
            overlap = len(needle & _tokens(turn_text))
            if overlap or include_unmatched_turns:
                candidates.append((float(overlap), turn_id, turn_text))
        candidates.sort(key=lambda row: (-row[0], row[1]))

        if not candidates:
            evidence_id = hashlib.sha256(
                f"{channel}\0{episode_id}\0{session_id}\0{rank}".encode()
            ).hexdigest()[:20]
            return [{
                "evidence_id": evidence_id,
                "session_id": session_id,
                "turn_id": "",
                "episode_id": episode_id,
                "source_text": source_text,
                "raw_text": source_text,
                "retrieval_channel": channel,
                "channel_rank": rank,
            }]

        rows = []
        for overlap, turn_id, turn_text in candidates:
            evidence_id = hashlib.sha256(
                f"{channel}\0{episode_id}\0{session_id}\0{turn_id}".encode()
            ).hexdigest()[:20]
            rows.append({
                "evidence_id": evidence_id,
                "session_id": session_id,
                "turn_id": turn_id,
                "episode_id": episode_id,
                "source_text": source_text,
                "raw_text": turn_text,
                "retrieval_channel": channel,
                "channel_rank": rank,
                "match_score": overlap,
            })
        return rows

    def _candidate_channels(
        self,
        query: str,
        *,
        terms: list[dict[str, str]] | Iterable[str] | str | None,
        limit: int,
        threshold: float,
        include_unmatched_turns: bool,
        enabled_channels: tuple[str, ...] | None = None,
    ) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
        # X7 ablation: when an explicit channel set is supplied, a channel
        # absent from it contributes no candidates. ``None`` keeps the
        # historic all-five-channel behaviour.
        active = tuple(enabled_channels) if enabled_channels else ALL_RETRIEVAL_CHANNELS
        explicit_terms: Any = terms
        if isinstance(terms, list) and (not terms or isinstance(terms[0], dict)):
            explicit_terms = [row["term"] for row in terms[1:]]
        payload = explore_memory(
            self.storage,
            query,
            explicit_terms=explicit_terms,
            options=ExploreOptions(
                limit=max(limit * 2, 20),
                threshold=threshold,
                file_limit=max(limit * 2, 20),
                evidence_limit=max(limit * 2, 20),
                semantic_queries=8,
                neighbor_limit=max(limit * 2, 20),
                neighbor_evidence_limit=max(limit * 2, 20),
                relation_evidence_limit=max(limit, 10),
            ),
            allowed_document_ids=self.allowed_document_ids,
        )

        by_channel: dict[str, list[dict[str, Any]]] = defaultdict(list)
        if "raw-document" in active:
            for rank, hit in enumerate(payload.get("file_hits") or [], 1):
                doc = hit.get("document") or {}
                document_id = self._document_for_version(str(doc.get("document_version_id") or ""))
                by_channel["raw-document"].extend(self._source_evidence(
                    channel="raw-document", document_id=document_id,
                    source_text=str(hit.get("text") or ""), query=query,
                    hints=[str(hit.get("matched_term") or "")], rank=rank,
                    include_unmatched_turns=include_unmatched_turns,
                ))
        if "episode-bm25" in active:
            for rank, hit in enumerate(payload.get("episode_hits") or [], 1):
                by_channel["episode-bm25"].extend(self._source_evidence(
                    channel="episode-bm25", episode_id=str(hit.get("episode_id") or ""),
                    source_text=str(hit.get("source_text") or ""), query=query,
                    hints=[str(hit.get("matched_query") or "")], rank=rank,
                    include_unmatched_turns=include_unmatched_turns,
                ))
        if "semantic-provenance" in active:
            for rank, hit in enumerate(payload.get("source_evidence") or [], 1):
                by_channel["semantic-provenance"].extend(self._source_evidence(
                    channel="semantic-provenance",
                    episode_id=str(hit.get("episode_version_id") or ""),
                    source_text=str(hit.get("source_text") or ""), query=query,
                    hints=[str(hit.get("target_name") or "")], rank=rank,
                    include_unmatched_turns=include_unmatched_turns,
                ))
        if "graph-neighbor" in active:
            for rank, hit in enumerate(payload.get("neighbor_evidence") or [], 1):
                by_channel["graph-neighbor"].extend(self._source_evidence(
                    channel="graph-neighbor",
                    episode_id=str(hit.get("episode_version_id") or ""),
                    source_text=str(hit.get("source_text") or ""), query=query,
                    hints=[str(hit.get("target_name") or "")], rank=rank,
                    include_unmatched_turns=include_unmatched_turns,
                ))
        if "relation-evidence" in active:
            for rank, hit in enumerate(payload.get("relation_evidence") or [], 1):
                by_channel["relation-evidence"].extend(self._source_evidence(
                    channel="relation-evidence",
                    episode_id=str(hit.get("episode_version_id") or ""),
                    source_text=str(hit.get("source_text") or ""), query=query,
                    hints=[str(hit.get("entity1_name") or ""), str(hit.get("entity2_name") or "")],
                    rank=rank, include_unmatched_turns=include_unmatched_turns,
                ))
        return dict(by_channel), payload

    def _legacy_explore(
        self,
        query: str,
        *,
        terms: Iterable[str] | str | None,
        limit: int,
        threshold: float,
        enabled_channels: tuple[str, ...] | None = None,
    ) -> dict[str, Any]:
        by_channel, payload = self._candidate_channels(
            query, terms=terms, limit=limit, threshold=threshold,
            include_unmatched_turns=False, enabled_channels=enabled_channels,
        )
        # ``legacy`` predates graph-neighbour provenance. Keep its channel set
        # byte-for-byte reproducible even though the shared explorer now emits it.
        by_channel.pop("graph-neighbor", None)
        session_scores: dict[str, float] = defaultdict(float)
        turn_scores: dict[str, float] = defaultdict(float)
        session_evidence: dict[str, list[dict[str, Any]]] = defaultdict(list)
        seen_evidence: set[str] = set()
        for channel, rows in by_channel.items():
            weight = self.CHANNEL_WEIGHTS[channel]
            seen_session_in_channel: set[str] = set()
            seen_turn_in_channel: set[str] = set()
            for row in rows:
                session_id = row["session_id"]
                channel_rank = int(row.get("channel_rank") or 1)
                if session_id not in seen_session_in_channel:
                    session_scores[session_id] += weight / (60 + channel_rank)
                    seen_session_in_channel.add(session_id)
                turn_id = str(row.get("turn_id") or "")
                if turn_id and turn_id not in seen_turn_in_channel:
                    turn_scores[turn_id] += weight / (60 + channel_rank)
                    seen_turn_in_channel.add(turn_id)
                if row["evidence_id"] not in seen_evidence:
                    seen_evidence.add(row["evidence_id"])
                    session_evidence[session_id].append(row)

        ranked_sessions = sorted(session_scores, key=lambda sid: (-session_scores[sid], sid))
        ranked_turns = sorted(turn_scores, key=lambda tid: (-turn_scores[tid], tid))
        contexts = []
        for session_id in ranked_sessions[:limit]:
            session = self.sessions[session_id]
            matched = [turn_id for turn_id in ranked_turns if turn_id in set(session.turn_ids)]
            contexts.append({
                "session_id": session_id,
                "timestamp": session.timestamp,
                "text": session.text,
                "turn_ids": session.turn_ids,
                "matched_turn_ids": matched,
                "score": session_scores[session_id],
                "evidence": session_evidence[session_id],
            })
        return {
            "contexts": contexts,
            "ranked_session_ids": ranked_sessions,
            "ranked_turn_ids": ranked_turns,
            "evidence": [row for sid in ranked_sessions for row in session_evidence[sid]],
            "channel_evidence": by_channel,
            "explore": payload,
            "retrieval_profile": "legacy",
        }

    def _embeddings(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        client = getattr(self.storage, "embedding_client", None)
        if client is None:
            return np.zeros((len(texts), 1), dtype=np.float32)
        try:
            encoded = client.encode(texts)
            # Degraded clients (e.g. sentence-transformers absent) return None for
            # every text; ``np.asarray(None)`` is a 0-d array that would later
            # trip ``len()`` at the ``query_vector = vectors[0] if len(vectors)``
            # guard. Normalize away both degenerate shapes here.
            if encoded is None:
                return np.zeros((len(texts), 1), dtype=np.float32)
            values = np.asarray(encoded, dtype=np.float32)
            if values.ndim == 0:
                values = values.reshape(1, 1)
            elif values.ndim == 1:
                values = values.reshape(1, -1)
            return values
        except Exception:
            return np.zeros((len(texts), 1), dtype=np.float32)

    def _hybrid_explore(
        self,
        query: str,
        *,
        threshold: float,
        config: HybridRetrievalConfig,
    ) -> dict[str, Any]:
        query_terms = extract_query_terms(query)
        legacy_guard = self._legacy_explore(
            query,
            terms=None,
            limit=max(config.candidate_k, config.context_k),
            threshold=threshold,
            # Keep the guard backfill on the same channel budget as the main
            # explorer so an X7 arm that excludes a channel does not leak
            # that channel's turns back in via the source-grounded guard.
            enabled_channels=config.enabled_channels,
        )
        by_channel, payload = self._candidate_channels(
            query, terms=query_terms, limit=max(config.candidate_k * 4, 120),
            threshold=threshold, include_unmatched_turns=True,
            enabled_channels=config.enabled_channels,
        )

        candidates: dict[str, dict[str, Any]] = {}
        channel_seen: dict[str, set[str]] = defaultdict(set)
        for channel, rows in by_channel.items():
            channel_weight = self.CHANNEL_WEIGHTS[channel]
            for row in rows:
                turn_id = str(row.get("turn_id") or "")
                session_id = str(row.get("session_id") or "")
                if not turn_id or self._turn_to_session.get(turn_id) != session_id:
                    continue
                candidate = candidates.setdefault(turn_id, {
                    "turn_id": turn_id,
                    "session_id": session_id,
                    "raw_text": str(row.get("raw_text") or ""),
                    "rrf_raw": 0.0,
                    "channels": [],
                    "evidence": [],
                })
                if turn_id not in channel_seen[channel]:
                    overlap = float(row.get("match_score") or 0.0)
                    match_factor = 0.2 + 0.4 * min(2.0, overlap)
                    candidate["rrf_raw"] += (
                        channel_weight * match_factor / (60 + int(row.get("channel_rank") or 1))
                    )
                    candidate["channels"].append(channel)
                    channel_seen[channel].add(turn_id)
                candidate["evidence"].append(row)

        legacy_turn_rank = list(legacy_guard.get("ranked_turn_ids") or [])
        legacy_evidence_by_turn: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for rows in (legacy_guard.get("channel_evidence") or {}).values():
            for row in rows:
                turn_id = str(row.get("turn_id") or "")
                if turn_id:
                    legacy_evidence_by_turn[turn_id].append(row)
        # Query expansion can occasionally suppress a turn found by the exact
        # legacy query. Keep those source-grounded turns eligible for v2 scoring.
        for turn_id in legacy_turn_rank[: max(config.candidate_k * 2, 60)]:
            if turn_id in candidates:
                continue
            session_id = self._turn_to_session.get(turn_id, "")
            if not session_id:
                continue
            raw_text = next(
                (text for candidate_id, text in self._session_turns[session_id]
                 if candidate_id == turn_id),
                "",
            )
            candidates[turn_id] = {
                "turn_id": turn_id,
                "session_id": session_id,
                "raw_text": raw_text,
                "rrf_raw": 0.0,
                "channels": ["legacy-guard"],
                "evidence": legacy_evidence_by_turn.get(turn_id, []),
            }

        if not candidates:
            return {
                "contexts": [], "ranked_session_ids": [], "ranked_turn_ids": [],
                "evidence": [], "channel_evidence": by_channel, "explore": payload,
                "retrieval_profile": "hybrid-v2", "query_terms": query_terms,
                "turn_scores": [], "budget": {"limit": config.evidence_token_budget, "used": 0},
                "profile": {**config.payload(), "sha256": config.fingerprint()},
            }

        turn_ids = sorted(candidates)
        texts = [candidates[turn_id]["raw_text"] for turn_id in turn_ids]
        vectors = self._embeddings([query, *texts])
        query_vector = vectors[0] if len(vectors) else np.zeros(1, dtype=np.float32)
        candidate_vectors = vectors[1:] if len(vectors) > 1 else np.zeros((len(texts), 1), dtype=np.float32)
        vector_by_turn = {
            turn_id: candidate_vectors[index] for index, turn_id in enumerate(turn_ids)
        }
        rrf_normalized = _minmax({turn_id: candidates[turn_id]["rrf_raw"] for turn_id in turn_ids})
        question_tokens = set(_content_tokens(query))
        phrase_terms = [
            _phrase_normalize(row["term"]) for row in query_terms[1:]
            if len(row["term"].split()) > 1
        ]
        for turn_id in turn_ids:
            candidate = candidates[turn_id]
            turn_tokens = _tokens(candidate["raw_text"])
            lexical = len(question_tokens & turn_tokens) / max(1, len(question_tokens))
            semantic = max(0.0, min(1.0, (_cosine(query_vector, vector_by_turn[turn_id]) + 1) / 2))
            normalized_turn = _phrase_normalize(candidate["raw_text"])
            unique_phrases = list(dict.fromkeys(value for value in phrase_terms if value))
            phrase = (
                sum(value in normalized_turn for value in unique_phrases) / len(unique_phrases)
                if unique_phrases else 0.0
            )
            features = {
                "rrf": rrf_normalized.get(turn_id, 0.0) * (0.25 + 0.75 * lexical),
                "semantic": semantic,
                "lexical": lexical,
                "phrase": phrase,
            }
            candidate["features"] = features
            candidate["feature_score"] = sum(
                HYBRID_V2_WEIGHTS[key] * features[key] for key in HYBRID_V2_WEIGHTS
            )

        feature_ranked = sorted(
            candidates.values(), key=lambda row: (-row["feature_score"], row["turn_id"])
        )
        feature_positions = {
            row["turn_id"]: rank for rank, row in enumerate(feature_ranked, 1)
        }
        legacy_turn_positions = {
            turn_id: rank for rank, turn_id in enumerate(legacy_turn_rank, 1)
        }
        for candidate in candidates.values():
            turn_id = candidate["turn_id"]
            candidate["score"] = (
                HYBRID_V2_TURN_RANK_FUSION["hybrid_features"]
                / (60 + feature_positions[turn_id])
                + (
                    HYBRID_V2_TURN_RANK_FUSION["legacy_guard"]
                    / (60 + legacy_turn_positions[turn_id])
                    if turn_id in legacy_turn_positions else 0.0
                )
            )
            candidate["feature_rank"] = feature_positions[turn_id]
            candidate["legacy_rank"] = legacy_turn_positions.get(turn_id)

        ranked_candidates = sorted(candidates.values(), key=lambda row: (-row["score"], row["turn_id"]))
        ranked_candidates = ranked_candidates[:config.candidate_k]
        ranked_turn_ids = [row["turn_id"] for row in ranked_candidates]

        turns_by_session: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for candidate in ranked_candidates:
            turns_by_session[candidate["session_id"]].append(candidate)
        session_scores: dict[str, float] = {}
        session_vectors: dict[str, np.ndarray] = {}
        for session_id, rows in turns_by_session.items():
            rows.sort(key=lambda row: (-row["score"], row["turn_id"]))
            session_scores[session_id] = sum(
                weight * rows[index]["score"]
                for index, weight in enumerate(HYBRID_V2_SESSION_AGGREGATION)
                if index < len(rows)
            )
            available = [vector_by_turn[row["turn_id"]] for row in rows[:3]]
            session_vectors[session_id] = np.mean(available, axis=0) if available else np.zeros(1)
        turn_aggregate_scores = dict(session_scores)
        hybrid_rank = sorted(session_scores, key=lambda sid: (-session_scores[sid], sid))
        legacy_rank = list(legacy_guard.get("ranked_session_ids") or [])
        hybrid_positions = {session_id: rank for rank, session_id in enumerate(hybrid_rank, 1)}
        legacy_positions = {session_id: rank for rank, session_id in enumerate(legacy_rank, 1)}
        # A rank-only guard prevents a highly literal later turn from evicting a
        # session that the established baseline reliably retrieved. It does not
        # affect turn ranking and never introduces a session outside the v2 pool.
        session_scores = {
            session_id: (
                HYBRID_V2_SESSION_RANK_FUSION["turn_aggregate"]
                / (60 + hybrid_positions[session_id])
                + (
                    HYBRID_V2_SESSION_RANK_FUSION["legacy_guard"]
                    / (60 + legacy_positions[session_id])
                    if session_id in legacy_positions else 0.0
                )
            )
            for session_id in session_scores
        }
        relevance = _minmax(session_scores)
        remaining = set(session_scores)
        selected_sessions: list[str] = []
        mmr_scores: dict[str, float] = {}
        while remaining and len(selected_sessions) < config.context_k:
            best: tuple[float, str] | None = None
            for session_id in sorted(remaining):
                redundancy = max(
                    (_cosine(session_vectors[session_id], session_vectors[chosen]) + 1) / 2
                    for chosen in selected_sessions
                ) if selected_sessions else 0.0
                mmr = HYBRID_V2_MMR_LAMBDA * relevance.get(session_id, 0.0) - (
                    1 - HYBRID_V2_MMR_LAMBDA
                ) * redundancy
                if best is None or mmr > best[0]:
                    best = (mmr, session_id)
            assert best is not None
            mmr_scores[best[1]] = best[0]
            selected_sessions.append(best[1])
            remaining.remove(best[1])

        ranked_session_ids = sorted(session_scores, key=lambda sid: (-session_scores[sid], sid))
        # Resolve explicit indirect-place references without an LLM. For example,
        # a highly ranked turn may say "moved from my home country" while another
        # source turn grounds "home country" as Sweden. These bridge turns are
        # genuine phrase matches and therefore count as anchors, not neighbours.
        named_tokens = {
            token.casefold()
            for row in query_terms if row.get("source") == "named"
            for token in _content_tokens(row.get("term") or "")
        }
        active_bridges = {
            phrase
            for candidate in ranked_candidates
            for phrase in _REFERENTIAL_BRIDGE_PHRASES
            if phrase in candidate["raw_text"].casefold()
        }
        bridge_turn_ids: list[str] = []
        if active_bridges:
            for session_id in selected_sessions:
                for turn_id, turn_text in self._session_turns[session_id]:
                    normalized = turn_text.casefold()
                    if not any(phrase in normalized for phrase in active_bridges):
                        continue
                    if named_tokens and not (named_tokens & set(_content_tokens(turn_text))):
                        continue
                    bridge_turn_ids.append(turn_id)
            bridge_turn_ids = list(dict.fromkeys(bridge_turn_ids))
            ranked_turn_ids = bridge_turn_ids + [
                turn_id for turn_id in ranked_turn_ids if turn_id not in set(bridge_turn_ids)
            ]
        selected_turns: dict[str, set[str]] = {session_id: set() for session_id in selected_sessions}
        anchors: dict[str, list[str]] = {}
        used_tokens = 0
        budget_exceeded_for_anchors = False
        # First guarantee the best anchor for every selected session.
        for session_id in selected_sessions:
            anchor = turns_by_session[session_id][0]["turn_id"]
            anchors[session_id] = list(dict.fromkeys([
                *[turn_id for turn_id in bridge_turn_ids if self._turn_to_session.get(turn_id) == session_id],
                *[row["turn_id"] for row in turns_by_session[session_id]],
            ]))
            selected_turns[session_id].add(anchor)
            line = next((text for tid, text in self._session_turns[session_id] if tid == anchor), "")
            used_tokens += _estimated_tokens(f"[{anchor}] {line}")
        if used_tokens > config.evidence_token_budget:
            budget_exceeded_for_anchors = True

        # Add remaining anchors and neighbours in global score order at turn boundaries.
        additions: list[tuple[float, str, str]] = []
        for turn_id in bridge_turn_ids:
            session_id = self._turn_to_session.get(turn_id, "")
            if session_id in selected_turns:
                additions.append((2.0, session_id, turn_id))
        for session_id in selected_sessions:
            rows = self._session_turns[session_id]
            positions = {turn_id: index for index, (turn_id, _) in enumerate(rows)}
            for candidate in turns_by_session[session_id]:
                anchor = candidate["turn_id"]
                additions.append((candidate["score"] + 1.0, session_id, anchor))
                position = positions.get(anchor, -1)
                if position >= 0:
                    for offset in range(1, config.neighbor_turns + 1):
                        for neighbor_index in (position - offset, position + offset):
                            if 0 <= neighbor_index < len(rows):
                                additions.append((candidate["score"] - offset * 0.01, session_id, rows[neighbor_index][0]))
        for _, session_id, turn_id in sorted(additions, key=lambda row: (-row[0], row[1], row[2])):
            if turn_id in selected_turns[session_id]:
                continue
            text = next((value for tid, value in self._session_turns[session_id] if tid == turn_id), "")
            cost = _estimated_tokens(f"[{turn_id}] {text}")
            if used_tokens + cost > config.evidence_token_budget:
                continue
            selected_turns[session_id].add(turn_id)
            used_tokens += cost

        contexts = []
        for session_id in selected_sessions:
            rows = [
                (turn_id, text) for turn_id, text in self._session_turns[session_id]
                if turn_id in selected_turns[session_id]
            ]
            included_ids = [turn_id for turn_id, _ in rows]
            matched_ids = [turn_id for turn_id in anchors[session_id] if turn_id in included_ids]
            evidence = [
                evidence
                for candidate in turns_by_session[session_id]
                for evidence in candidate["evidence"]
            ]
            contexts.append({
                "session_id": session_id,
                "timestamp": self.sessions[session_id].timestamp,
                "text": "\n".join(f"[{turn_id}] {text}" for turn_id, text in rows),
                "turn_ids": included_ids,
                "matched_turn_ids": matched_ids,
                "score": session_scores[session_id],
                "mmr_score": mmr_scores[session_id],
                "evidence": evidence,
            })

        audit_scores = [{
            "turn_id": row["turn_id"],
            "session_id": row["session_id"],
            "score": round(float(row["score"]), 8),
            "feature_score": round(float(row["feature_score"]), 8),
            "feature_rank": row["feature_rank"],
            "legacy_rank": row["legacy_rank"],
            "features": {key: round(float(value), 8) for key, value in row["features"].items()},
            "rrf_raw": round(float(row["rrf_raw"]), 8),
            "channels": row["channels"],
        } for row in ranked_candidates]
        return {
            "contexts": contexts,
            "ranked_session_ids": ranked_session_ids,
            "ranked_turn_ids": ranked_turn_ids,
            "evidence": [row for context in contexts for row in context["evidence"]],
            "channel_evidence": by_channel,
            "explore": payload,
            "retrieval_profile": "hybrid-v2",
            "query_terms": query_terms,
            "turn_scores": audit_scores,
            "session_scores": [
                {
                    "session_id": session_id,
                    "score": round(float(session_scores[session_id]), 8),
                    "turn_aggregate_score": round(float(turn_aggregate_scores[session_id]), 8),
                    "hybrid_rank": hybrid_positions[session_id],
                    "legacy_rank": legacy_positions.get(session_id),
                }
                for session_id in ranked_session_ids
            ],
            "referential_bridges": {
                "phrases": sorted(active_bridges),
                "turn_ids": bridge_turn_ids,
            },
            "budget": {
                "limit": config.evidence_token_budget,
                "used": used_tokens,
                "estimator": "utf8-bytes/4-ceil",
                "anchor_overflow": budget_exceeded_for_anchors,
            },
            "profile": {**config.payload(), "sha256": config.fingerprint()},
        }

    def explore(
        self,
        query: str,
        *,
        terms: Iterable[str] | str | None = None,
        limit: int = 20,
        threshold: float = 0.3,
        retrieval_profile: str = "legacy",
        candidate_k: int = 30,
        context_k: int = 5,
        evidence_token_budget: int = 1600,
        neighbor_turns: int = 1,
        enabled_channels: tuple[str, ...] | None = None,
    ) -> dict[str, Any]:
        if retrieval_profile == "legacy":
            return self._legacy_explore(
                query, terms=terms, limit=limit, threshold=threshold,
                enabled_channels=enabled_channels,
            )
        if retrieval_profile != "hybrid-v2":
            raise ValueError("retrieval_profile must be legacy or hybrid-v2")
        config = HybridRetrievalConfig(
            candidate_k=max(1, int(candidate_k)),
            context_k=max(1, int(context_k)),
            evidence_token_budget=max(1, int(evidence_token_budget)),
            neighbor_turns=max(0, int(neighbor_turns)),
            enabled_channels=enabled_channels,
        )
        return self._hybrid_explore(query, threshold=threshold, config=config)

    def search(self, question: str, *, top_k: int, threshold: float = 0.3) -> list[dict[str, Any]]:
        """Backward-compatible fixed one-shot baseline API."""
        return self.explore(question, limit=top_k, threshold=threshold)["contexts"]


DeepDreamRetriever = UnifiedRetriever
