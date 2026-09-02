"""Email collector deliberately avoids account enumeration and breach lookups."""

from __future__ import annotations

from time import perf_counter

from osint.collectors.base import CollectorSpec
from osint.context import InvestigationContext
from osint.models import CollectorOutcome, EntityType


class EmailCollector:
    spec = CollectorSpec("email", (EntityType.EMAIL,), priority=30)
    async def collect(self, context: InvestigationContext, entity_id: str) -> CollectorOutcome:
        started = perf_counter()
        return CollectorOutcome(self.spec.name, True, errors=["Email account enumeration and breach lookup are intentionally not implemented."], elapsed_ms=int((perf_counter()-started)*1000))
