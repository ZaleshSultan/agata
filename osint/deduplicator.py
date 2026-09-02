"""Entity deduplication by normalized type/value, not by speculative identity."""

from __future__ import annotations

from osint.models import Entity
from osint.normalizer import normalize


class Deduplicator:
    @staticmethod
    def find_existing(entities: dict[str, Entity], candidate: Entity) -> Entity | None:
        needle = normalize(candidate.value, candidate.type)
        for entity in entities.values():
            if entity.type is candidate.type and normalize(entity.value, entity.type) == needle:
                return entity
        return None

    def merge(self, entities: dict[str, Entity], candidate: Entity) -> tuple[Entity, bool]:
        existing = self.find_existing(entities, candidate)
        if existing is None:
            entities[candidate.id] = candidate; return candidate, True
        existing.attributes.update({key: value for key, value in candidate.attributes.items() if value})
        existing.evidence_ids = list(dict.fromkeys([*existing.evidence_ids, *candidate.evidence_ids]))
        return existing, False
