"""Deterministic playbooks and bounded pivots; no LLM controls collection."""

from __future__ import annotations

from osint.collectors.base import CollectorSpec
from osint.models import EntityType, Pivot


class Planner:
    def __init__(self, registry: dict[str, CollectorSpec]) -> None: self._registry = registry

    def initial_collectors(self, entity_type: EntityType, depth: str) -> list[str]:
        playbooks = {
            EntityType.USERNAME: ["username_discovery", "github", "web"],
            EntityType.EMAIL: ["email", "web"],
            EntityType.DOMAIN: ["domain", "web"],
            EntityType.WEBSITE: ["domain", "web"],
            EntityType.PHONE: ["web"],
            EntityType.IMAGE: ["image_metadata", "image_visual"],
        }
        names = playbooks.get(entity_type, ["web"])
        if depth == "quick": names = names[:2]
        return [name for name in names if name in self._registry and self._registry[name].enabled]

    def pivots(self, entity_id: str, entity_type: EntityType, depth: str) -> list[Pivot]:
        if depth != "deep": return []
        # Only source-type pivots; no recursive profile crawling.
        if entity_type is EntityType.SOCIAL and "web" in self._registry:
            return [Pivot(entity_id, "web", 70, "Public profile URL may contain corroborating links.", 0.35)]
        return []
