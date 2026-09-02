"""Optional X integration boundary; no scraping or login bypass is included."""

from __future__ import annotations

from time import perf_counter

from osint.collectors.base import CollectorSpec
from osint.context import InvestigationContext
from osint.models import CollectorOutcome, EntityType


class XCollector:
    spec = CollectorSpec("x", (EntityType.USERNAME, EntityType.SOCIAL), priority=50, requires_api_key=True, enabled=False)
    async def collect(self, context: InvestigationContext, entity_id: str) -> CollectorOutcome:
        started = perf_counter()
        return CollectorOutcome(self.spec.name, True, errors=["X collector is disabled until a lawful configured provider is supplied."], elapsed_ms=int((perf_counter()-started)*1000))
