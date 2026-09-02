"""Mutable state shared by deterministic collectors, never by the Agata LLM."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from osint.cache import InvestigationCache
from osint.limits import Budget, InvestigationLimits
from osint.models import Entity, Evidence, Observation, Pivot, Relation


@dataclass
class InvestigationContext:
    target: str
    target_type: str
    depth: str
    limits: InvestigationLimits
    options: dict[str, Any] = field(default_factory=dict)
    investigation_id: str = field(default_factory=lambda: uuid4().hex)
    entities: dict[str, Entity] = field(default_factory=dict)
    evidence: dict[str, Evidence] = field(default_factory=dict)
    observations: dict[str, Observation] = field(default_factory=dict)
    relations: dict[str, Relation] = field(default_factory=dict)
    visited_urls: set[str] = field(default_factory=set)
    executed_collectors: list[str] = field(default_factory=list)
    pending_pivots: list[Pivot] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    timings_ms: dict[str, int] = field(default_factory=dict)
    cache: InvestigationCache = field(default_factory=InvestigationCache)
    budget: Budget = field(init=False)

    def __post_init__(self) -> None:
        self.budget = Budget(self.limits)

    def add_entity(self, entity: Entity) -> Entity:
        self.budget.check_time()
        if len(self.entities) >= self.limits.max_entities:
            return entity
        self.entities[entity.id] = entity
        return entity

    def add_evidence(self, evidence: Evidence) -> Evidence:
        self.evidence[evidence.id] = evidence; return evidence

    def add_observation(self, observation: Observation) -> Observation:
        self.observations[observation.id] = observation; return observation

    def add_relation(self, relation: Relation) -> Relation:
        self.relations[relation.id] = relation; return relation
