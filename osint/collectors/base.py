"""Collector contract and registry metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from osint.context import InvestigationContext
from osint.models import CollectorOutcome, EntityType


@dataclass(frozen=True, slots=True)
class CollectorSpec:
    name: str
    entity_types: tuple[EntityType, ...]
    cost: int = 1
    priority: int = 50
    requires_api_key: bool = False
    enabled: bool = True


class Collector(Protocol):
    spec: CollectorSpec

    async def collect(self, context: InvestigationContext, entity_id: str) -> CollectorOutcome: ...
