"""Independent deterministic OSINT orchestration entry point."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import config
from osint.collectors import DomainCollector, EmailCollector, GitHubCollector, UsernameCollector, WebCollector, XCollector
from osint.collectors.base import Collector
from osint.context import InvestigationContext
from osint.correlator import Correlator
from osint.executor import Executor
from osint.image_intel.metadata import extract_metadata
from osint.image_intel.visual import analyze_visual_clues
from osint.limits import InvestigationLimits
from osint.models import CollectorOutcome, Entity, EntityType, Evidence
from osint.normalizer import infer_target_type, normalize
from osint.planner import Planner
from osint.providers.ai import configured_provider
from osint.scheduler import Scheduler
from osint.storage import InvestigationStorage


class OSINTEngine:
    """A lawful-public-source engine that can be used with no LLM or Agent."""

    def __init__(self, collectors: Mapping[str, Collector] | None = None, storage: InvestigationStorage | None = None) -> None:
        supplied = collectors or {item.spec.name: item for item in (UsernameCollector(), GitHubCollector(), WebCollector(), DomainCollector(), EmailCollector(), XCollector())}
        self.collectors = dict(supplied); self.storage = storage or InvestigationStorage()
        self.planner = Planner({name: collector.spec for name, collector in self.collectors.items()})

    def provider_registry(self) -> list[dict[str, Any]]:
        """Introspection for CLI/API without exposing implementation instances."""
        return [{"name": spec.name, "supported_entity_types": [kind.value for kind in spec.entity_types], "cost": spec.cost, "priority": spec.priority, "enabled": spec.enabled, "requires_api_key": spec.requires_api_key} for spec in (collector.spec for collector in self.collectors.values())]

    async def investigate(self, target: str, target_type: str = "auto", image: bytes | None = None, depth: str = "standard", options: dict[str, Any] | None = None) -> dict[str, Any]:
        if not config.OSINT_ENABLED: return {"success": False, "error": "OSINT is disabled by OSINT_ENABLED."}
        if depth not in {"quick", "standard", "deep"}: return {"success": False, "error": "depth must be quick, standard, or deep."}
        inferred = infer_target_type(target) if target_type == "auto" else EntityType(target_type)
        context = InvestigationContext(target=target, target_type=inferred.value, depth=depth, limits=self._limits(options), options=options or {})
        root = Entity(inferred, normalize(target, inferred), label=target); context.add_entity(root)
        if image is not None:
            image_entity = Entity(EntityType.IMAGE, "provided-image", label="Provided investigation image"); context.add_entity(image_entity)
            await self._apply_image(context, image_entity, image)
        executor, scheduler = Executor(), Scheduler(context.limits.max_parallel_tasks)
        names = self.planner.initial_collectors(inferred, depth)
        outcomes = await scheduler.run([lambda name=name: self.collectors[name].collect(context, root.id) for name in names])
        new_ids: list[str] = []
        for outcome in outcomes: new_ids.extend(executor.apply(context, outcome))
        # Deep investigations may enrich only one level of public social links.
        if depth == "deep":
            jobs = []
            for entity_id in new_ids:
                entity = context.entities[entity_id]
                for pivot in self.planner.pivots(entity_id, entity.type, depth):
                    if pivot.collector in self.collectors: jobs.append(lambda pivot=pivot: self.collectors[pivot.collector].collect(context, pivot.entity_id))
            for outcome in await scheduler.run(jobs[:context.limits.max_parallel_tasks]): executor.apply(context, outcome)
        for relation in Correlator().correlate(context): context.add_relation(relation)
        folder = self.storage.save(context)
        return self._result(context, folder)

    @staticmethod
    def _limits(options: dict[str, Any] | None) -> InvestigationLimits:
        values = dict(options or {}); allowed = {key: values[key] for key in InvestigationLimits.__dataclass_fields__ if key in values}
        return InvestigationLimits(**allowed)

    @staticmethod
    async def _apply_image(context: InvestigationContext, image_entity: Entity, image: bytes) -> None:
        try:
            metadata, evidence, observations = extract_metadata(image, image_entity.id)
            image_entity.attributes["metadata"] = metadata
            for item in evidence: context.add_evidence(item)
            for item in observations: context.add_observation(item)
        except Exception as exc:
            context.errors.append(f"Image metadata extraction failed: {exc}")
        try:
            evidence, observations = await analyze_visual_clues(image, image_entity.id, configured_provider())
            for item in evidence: context.add_evidence(item)
            for item in observations: context.add_observation(item)
        except Exception as exc:
            context.errors.append(f"Image visual analysis failed: {exc}")

    @staticmethod
    def _result(context: InvestigationContext, folder: Path) -> dict[str, Any]:
        return {"success": True, "investigation_id": context.investigation_id, "target": context.target, "target_type": context.target_type, "report_path": str(folder / "report.md"), "entities": [item.to_dict() for item in context.entities.values()], "relations": [item.to_dict() for item in context.relations.values()], "evidence": [item.to_dict() for item in context.evidence.values()], "metrics": {"runtime_seconds": round(context.budget.elapsed_seconds, 3), "requests": context.budget.requests, "cache_hits": context.cache.hits, "entities_discovered": len(context.entities), "provider_timings_ms": context.timings_ms}, "errors": context.errors}


_DEFAULT_ENGINE: OSINTEngine | None = None
def get_engine() -> OSINTEngine:
    global _DEFAULT_ENGINE
    if _DEFAULT_ENGINE is None: _DEFAULT_ENGINE = OSINTEngine()
    return _DEFAULT_ENGINE
