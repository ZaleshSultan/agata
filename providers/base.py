"""Small provider contracts used by reusable components, not OSINT orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(slots=True)
class VisionAnalysis:
    text: str
    structured: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class VisionProvider(Protocol):
    """Contract for a future Gemini/OpenRouter/local vision adapter."""

    async def analyze_image(self, image: bytes, prompt: str) -> VisionAnalysis:
        """Return a provider-neutral textual and structured image analysis."""
