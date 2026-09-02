"""Passive domain normalization; active DNS/WHOIS can be registered later."""

from __future__ import annotations

from time import perf_counter
from urllib.parse import urlsplit

from osint.collectors.base import CollectorSpec
from osint.context import InvestigationContext
from osint.models import CollectorOutcome, Entity, EntityType, Evidence


class DomainCollector:
    spec = CollectorSpec("domain", (EntityType.DOMAIN, EntityType.WEBSITE), priority=20)
    async def collect(self, context: InvestigationContext, entity_id: str) -> CollectorOutcome:
        started, subject = perf_counter(), context.entities[entity_id]
        domain = urlsplit(subject.value if "://" in subject.value else f"https://{subject.value}").netloc or subject.value
        entity = Entity(EntityType.DOMAIN, domain, label=domain)
        proof = Evidence("Domain was derived from the supplied public URL.", source= self.spec.name)
        entity.evidence_ids.append(proof.id)
        return CollectorOutcome(self.spec.name, True, [entity], [proof], elapsed_ms=int((perf_counter()-started)*1000))
