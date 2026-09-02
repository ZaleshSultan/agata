"""Provider-neutral records used throughout the independent OSINT engine."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any
from uuid import uuid4


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class EntityType(StrEnum):
    PERSON = "person"
    USERNAME = "username"
    EMAIL = "email"
    PHONE = "phone"
    SOCIAL = "social"
    WEBSITE = "website"
    DOMAIN = "domain"
    IP = "ip"
    ORGANIZATION = "organization"
    LOCATION = "location"
    IMAGE = "image"
    EVENT = "event"


class AssertionStatus(StrEnum):
    OBSERVED = "observed"
    INFERRED = "inferred"
    POSSIBLE = "possible"
    UNVERIFIED = "unverified"


@dataclass(slots=True)
class Evidence:
    claim: str
    source_url: str = ""
    excerpt: str = ""
    source: str = ""
    collected_at: str = field(default_factory=utcnow)
    status: AssertionStatus = AssertionStatus.OBSERVED
    id: str = field(default_factory=lambda: f"ev_{uuid4().hex[:12]}")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self); data["status"] = self.status.value; return data


@dataclass(slots=True)
class Entity:
    type: EntityType
    value: str
    label: str = ""
    attributes: dict[str, Any] = field(default_factory=dict)
    evidence_ids: list[str] = field(default_factory=list)
    id: str = field(default_factory=lambda: f"ent_{uuid4().hex[:12]}")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self); data["type"] = self.type.value; return data


@dataclass(slots=True)
class Observation:
    subject_id: str
    claim: str
    status: AssertionStatus = AssertionStatus.OBSERVED
    confidence: float = 0.5
    evidence_ids: list[str] = field(default_factory=list)
    source: str = ""
    timestamp: str = field(default_factory=utcnow)
    id: str = field(default_factory=lambda: f"obs_{uuid4().hex[:12]}")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self); data["status"] = self.status.value; return data


@dataclass(slots=True)
class Relation:
    source_id: str
    target_id: str
    type: str
    confidence: float
    evidence_ids: list[str] = field(default_factory=list)
    status: AssertionStatus = AssertionStatus.INFERRED
    reason: str = ""
    timestamp: str = field(default_factory=utcnow)
    id: str = field(default_factory=lambda: f"rel_{uuid4().hex[:12]}")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self); data["status"] = self.status.value; return data


@dataclass(slots=True)
class Pivot:
    entity_id: str
    collector: str
    priority: int
    reason: str
    confidence: float
    cost: int = 1


@dataclass(slots=True)
class CollectorOutcome:
    collector: str
    success: bool
    entities: list[Entity] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    observations: list[Observation] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    requests: int = 0
    cache_hits: int = 0
    elapsed_ms: int = 0
