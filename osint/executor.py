"""Applies successful collector output to context through normalization/deduplication."""

from __future__ import annotations

from osint.context import InvestigationContext
from osint.deduplicator import Deduplicator
from osint.models import CollectorOutcome


class Executor:
    def __init__(self) -> None: self._deduplicator = Deduplicator()
    def apply(self, context: InvestigationContext, outcome: CollectorOutcome) -> list[str]:
        context.executed_collectors.append(outcome.collector)
        context.timings_ms[outcome.collector] = outcome.elapsed_ms
        context.errors.extend(outcome.errors)
        for evidence in outcome.evidence: context.add_evidence(evidence)
        ids: list[str] = []
        for entity in outcome.entities:
            merged, created = self._deduplicator.merge(context.entities, entity)
            if created: ids.append(merged.id)
        for observation in outcome.observations:
            # Collector-generated entities may be deduplicated, so keep an
            # observation only when its source entity survived the merge.
            if observation.subject_id in context.entities: context.add_observation(observation)
        return ids
