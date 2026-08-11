"""Typed data used by the Memory v2 boundary."""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class MemoryType(StrEnum):
    FACT = "FACT"
    PREFERENCE = "PREFERENCE"
    PERSON = "PERSON"
    PROJECT = "PROJECT"
    GOAL = "GOAL"
    EVENT = "EVENT"
    DECISION = "DECISION"
    ROUTINE = "ROUTINE"
    RELATIONSHIP = "RELATIONSHIP"
    TEMPORARY = "TEMPORARY"


class MemoryStatus(StrEnum):
    ACTIVE = "ACTIVE"
    SUPERSEDED = "SUPERSEDED"
    EXPIRED = "EXPIRED"
    ARCHIVED = "ARCHIVED"


class MemorySource(StrEnum):
    CONVERSATION = "CONVERSATION"
    USER_EXPLICIT = "USER_EXPLICIT"
    CONSOLIDATION = "CONSOLIDATION"
    SYSTEM = "SYSTEM"
    IMPORT = "IMPORT"


@dataclass(slots=True)
class MemoryCandidate:
    """Untrusted proposal from a rule or LLM extractor; never persisted directly."""
    content: str
    memory_type: MemoryType
    importance: float
    confidence: float
    source: MemorySource = MemorySource.CONVERSATION
    subject: str | None = None
    entities: list[str] = field(default_factory=list)
    expires_at: str | None = None
    temporary: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Memory:
    id: int | None
    content: str
    memory_type: MemoryType = MemoryType.FACT
    importance: float = 0.5
    confidence: float = 0.7
    status: MemoryStatus = MemoryStatus.ACTIVE
    source: MemorySource = MemorySource.CONVERSATION
    created_at: str | None = None
    updated_at: str | None = None
    last_recalled_at: str | None = None
    expires_at: str | None = None
    consolidated_at: str | None = None
    supersedes_id: int | None = None
    embedding: list[float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    similarity: float | None = None
    score: float | None = None
