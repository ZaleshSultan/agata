"""Small, public username discovery set; deliberately not mass enumeration."""

from __future__ import annotations

from time import perf_counter

from osint.context import InvestigationContext
from osint.models import CollectorOutcome, Entity, EntityType, Evidence
from osint.collectors.base import CollectorSpec


class UsernameCollector:
    spec = CollectorSpec("username_discovery", (EntityType.USERNAME,), priority=10)
    _PLATFORMS = {
        "GitHub": "https://github.com/{username}",
        "Reddit": "https://www.reddit.com/user/{username}",
        "X": "https://x.com/{username}",
    }

    async def collect(self, context: InvestigationContext, entity_id: str) -> CollectorOutcome:
        started, subject = perf_counter(), context.entities[entity_id]
        items = [
            Entity(EntityType.SOCIAL, template.format(username=subject.value), label=platform,
                   attributes={"platform": platform, "username": subject.value})
            for platform, template in self._PLATFORMS.items()
        ]
        evidence = [Evidence(
            claim=f"Candidate public {item.label} profile URL generated from supplied username.",
            source_url=item.value, source=self.spec.name,
        ) for item in items]
        for item, proof in zip(items, evidence, strict=True): item.evidence_ids.append(proof.id)
        return CollectorOutcome(self.spec.name, True, items, evidence, elapsed_ms=int((perf_counter() - started) * 1000))
