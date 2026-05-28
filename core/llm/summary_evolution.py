"""SummaryEvolutionMixin - deprecated, summary field removed from concept_version."""
from __future__ import annotations
from typing import Optional
from ..models import Entity


class SummaryEvolutionMixin:
    """No-op stub: summary field has been removed from the data model."""

    async def evolve_entity_summary(self, entity: Entity, old_version: Optional[Entity] = None) -> str:
        return ""
