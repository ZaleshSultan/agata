"""Base interfaces and data contracts for media providers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(slots=True)
class PlaybackStatus:
    """Detailed live playback state from the browser DOM player."""

    navigation: bool = False
    element_found: bool = False
    play_requested: bool = False
    playing: bool = False
    muted: bool = False
    current_time: float = 0.0
    duration: float = 0.0
    verification: str = "pending"  # "opened", "play_requested", "started", "verified", "playback_verified_audio_unknown", "failed"
    error: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "navigation": self.navigation,
            "element_found": self.element_found,
            "play_requested": self.play_requested,
            "playing": self.playing,
            "muted": self.muted,
            "current_time": self.current_time,
            "duration": self.duration,
            "verification": self.verification,
            "error": self.error,
        }


@dataclass(slots=True)
class MediaResult:
    url: str
    title: str
    duration: str = ""
    position: int = 0
    score: float = 0.0
    channel: str = ""
    description: str = ""
    verified_channel: bool = False
    content_type: str = "general"


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
    verification: str = ""
    playback_status: PlaybackStatus | None = None
    extra: dict[str, Any] = field(default_factory=dict)

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
        if self.verification:
            d["verification"] = self.verification
        if self.playback_status:
            d["playback_status"] = self.playback_status.as_dict()
        if self.extra:
            d["extra"] = self.extra
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

