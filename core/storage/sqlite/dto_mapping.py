"""V1.5 table rows → legacy Entity/Relation/Episode DTO mapping."""
from datetime import datetime, timezone
from typing import Optional

from core.models import Entity, Episode, Relation


def _parse_dt(value) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except Exception:
            return None
    return None


def _now():
    return datetime.now(timezone.utc)


def observation_to_entity(
    family_row: dict,
    obs_row: dict,
    *,
    embedding_blob: Optional[bytes] = None,
    version_seq: int = 1,
) -> Entity:
    """Map V1.5 entity_families + entity_observations rows → Entity DTO."""
    now = _now()
    return Entity(
        absolute_id=obs_row["entity_id"],
        family_id=family_row["entity_family_id"],
        name=obs_row.get("name") or family_row.get("canonical_name", ""),
        content=obs_row.get("content") or family_row.get("canonical_content", ""),
        event_time=_parse_dt(obs_row.get("processed_at")) or now,
        processed_time=_parse_dt(obs_row.get("processed_at")) or now,
        episode_id=obs_row.get("episode_id", ""),
        source_document="",
        embedding=embedding_blob,
        version_seq=version_seq,
    )


def assertion_to_relation(
    family_row: dict,
    assert_row: dict,
    *,
    subject_entity_id: str = "",
    object_entity_id: str = "",
    embedding_blob: Optional[bytes] = None,
    version_seq: int = 1,
) -> Relation:
    """Map V1.5 relation_families + relation_assertions rows → Relation DTO."""
    now = _now()
    return Relation(
        absolute_id=assert_row["relation_id"],
        family_id=family_row["relation_family_id"],
        entity1_absolute_id=subject_entity_id or assert_row.get("subject_entity_id", ""),
        entity2_absolute_id=object_entity_id or assert_row.get("object_entity_id", ""),
        content=assert_row.get("content") or family_row.get("canonical_content", ""),
        event_time=_parse_dt(assert_row.get("processed_at")) or now,
        processed_time=_parse_dt(assert_row.get("processed_at")) or now,
        episode_id=assert_row.get("episode_id", ""),
        source_document="",
        entity1_family_id=assert_row.get("subject_entity_family_id", ""),
        entity2_family_id=assert_row.get("object_entity_family_id", ""),
        embedding=embedding_blob,
        version_seq=version_seq,
    )


def episode_row_to_dto(episode_row: dict) -> Episode:
    """Map V1.5 episodes row → Episode DTO."""
    now = _now()
    return Episode(
        absolute_id=episode_row["episode_id"],
        content=episode_row.get("memory_text") or episode_row.get("source_text", ""),
        event_time=_parse_dt(episode_row.get("event_time")) or now,
        source_document="",
        processed_time=_parse_dt(episode_row.get("processed_at")) or now,
        activity_type=episode_row.get("activity_type"),
        episode_type=episode_row.get("episode_type"),
    )
