"""Email collector deliberately avoids account enumeration and breach lookups."""

from __future__ import annotations

from time import perf_counter

from osint.collectors.base import CollectorSpec
from osint.context import InvestigationContext
from osint.models import CollectorOutcome, Entity, EntityType, Evidence


class EmailCollector:
    spec = CollectorSpec("email", (EntityType.EMAIL,), priority=30)
    async def collect(self, context: InvestigationContext, entity_id: str) -> CollectorOutcome:
        started = perf_counter(); email = context.entities[entity_id].value
        domain = email.rsplit("@", 1)[-1]
        entity = Entity(EntityType.DOMAIN, domain, label=domain, attributes={"derived_from": "public_email"})
        proof = Evidence("Domain derived locally from supplied email address.", source=self.spec.name)
        entity.evidence_ids.append(proof.id)
        return CollectorOutcome(self.spec.name, True, [entity], [proof], errors=["No account enumeration, breach lookup, or mailbox access was performed."], elapsed_ms=int((perf_counter()-started)*1000))
