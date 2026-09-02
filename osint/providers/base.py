"""Optional external AI boundary; no collector delegates network mechanics to AI."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(slots=True)
class AIAnalysis:
    text: str
    structured: dict[str, Any] = field(default_factory=dict)


class AIProvider(Protocol):
    name: str
    async def analyze_image(self, image: bytes, prompt: str) -> AIAnalysis: ...
