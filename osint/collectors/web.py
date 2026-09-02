"""Web footprint collector reusing Agata's shared browser layer."""

from __future__ import annotations

from time import perf_counter

from browser import BrowserManager, Extractor, Navigation
from osint.collectors.base import CollectorSpec
from osint.context import InvestigationContext
from osint.models import CollectorOutcome, Entity, EntityType, Evidence


class WebCollector:
    spec = CollectorSpec("web", (EntityType.USERNAME, EntityType.EMAIL, EntityType.DOMAIN, EntityType.WEBSITE), priority=40)

    def __init__(self, manager: BrowserManager | None = None) -> None:
        self._manager = manager

    async def collect(self, context: InvestigationContext, entity_id: str) -> CollectorOutcome:
        started, subject = perf_counter(), context.entities[entity_id]
        cache_key = f"web:{subject.type.value}:{subject.value}"
        cached = context.cache.get(cache_key)
        if cached is not None: return cached
        own_manager = self._manager is None
        manager = self._manager or BrowserManager()
        try:
            context.budget.consume_request(pages=1)
            result = await Navigation(manager).search(subject.value)
            if not result.success or result.page is None:
                outcome = CollectorOutcome(self.spec.name, False, errors=[result.error or "Browser search failed."], elapsed_ms=int((perf_counter()-started)*1000))
            else:
                extracted = await Extractor().extract_links(result.page)
                links = (extracted.links or [])[:5] if extracted.success else []
                entities = [Entity(EntityType.WEBSITE, link["url"], label=link.get("text", ""), attributes={"discovered_from": result.url}) for link in links]
                proof = Evidence("Public web search results page visited.", result.url, source=self.spec.name)
                for item in entities: item.evidence_ids.append(proof.id)
                outcome = CollectorOutcome(self.spec.name, extracted.success, entities, [proof], errors=([extracted.error] if extracted.error else []), requests=1, elapsed_ms=int((perf_counter()-started)*1000))
        except Exception as exc:
            outcome = CollectorOutcome(self.spec.name, False, errors=[f"Web collector failed: {exc}"], elapsed_ms=int((perf_counter()-started)*1000))
        finally:
            if own_manager: await manager.close()
        context.cache.set(cache_key, outcome); return outcome
