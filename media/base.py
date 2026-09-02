"""Base interfaces and data contracts for media providers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass(slots=True)
class MediaResult:
    url: str
    title: str
    duration: str = ""
    position: int = 0
    score: float = 0.0


@dataclass(slots=True)
class ActionResult:
    success: bool
    action: str = "play_content"
    platform: str = ""
    query: str = ""
    url: str = ""
    title: str = ""
    stage: str = ""
    reason: str = ""
    extra: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        d: dict[str, object] = {
            "success": self.success,
            "action": self.action,
        }
        if self.platform:
            d["platform"] = self.platform
        if self.query:
            d["query"] = self.query
        if self.url:
            d["url"] = self.url
        if self.title:
            d["title"] = self.title
        if self.stage:
            d["stage"] = self.stage
        if self.reason:
            d["reason"] = self.reason
        return d


@runtime_checkable
class MediaProvider(Protocol):
    """Protocol for platform-specific media search and playback."""

    async def search(self, query: str) -> list[MediaResult]:
        """Search platform for media matching query, returning ranked results."""
        ...

    async def play(self, result: MediaResult) -> ActionResult:
        """Open and verify playback for the given media item."""
        ...
