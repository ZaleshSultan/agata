"""Deterministic parser for media transport commands (stop, pause, resume, next, previous)."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(slots=True)
class MediaCommand:
    action: str  # "stop" | "pause" | "resume" | "next" | "previous" | "play_pause"
    raw_text: str = ""


# Dedicated patterns for transport commands to ensure zero-LLM priority routing
_STOP_EXACT = re.compile(
    r"^\s*(?:стоп|останови|остановить|поставь\s+на\s+стоп|на\s+стоп|заглуши|остановись)\s*[.!]?\s*$",
    re.IGNORECASE,
)

_PAUSE_EXACT = re.compile(
    r"^\s*(?:пауза|на\s+паузу|поставь\s+на\s+паузу|приостанови|приостановить|паузу|сделай\s+паузу)\s*[.!]?\s*$",
    re.IGNORECASE,
)

_RESUME_EXACT = re.compile(
    r"^\s*(?:продолжи|возобнови|продолжай|продолжить|возобновить|играй|играй\s+дальше)\s*[.!]?\s*$",
    re.IGNORECASE,
)

_NEXT_EXACT = re.compile(
    r"^\s*(?:следующ(?:ий|ая|ее|ую)?(?:\s+(?:песн\w*|трек\w*|композиц\w*))?|переключи(?:\s+(?:трек\w*|песн\w*))?|скип|пропусти(?:\s+трек)?)\s*[.!]?\s*$",
    re.IGNORECASE,
)

_PREV_EXACT = re.compile(
    r"^\s*(?:предыдущ(?:ий|ая|ее|ую)?(?:\s+(?:песн\w*|трек\w*|композиц\w*))?|(?:назад|верни)(?:\s+(?:трек\w*|песн\w*))?)\s*[.!]?\s*$",
    re.IGNORECASE,
)

# Inline transport patterns
_INLINE_STOP = re.compile(
    r"\b(?:поставь\s+на\s+стоп|останови\s+(?:музыку|воспроизведение|трек|песню|видео)|выключи\s+(?:музыку|воспроизведение|звук))\b",
    re.IGNORECASE,
)

_INLINE_PAUSE = re.compile(
    r"\b(?:поставь\s+на\s+паузу|сделай\s+паузу|пауза\s+в\s+музыке)\b",
    re.IGNORECASE,
)

_INLINE_RESUME = re.compile(
    r"\b(?:возобнови\s+(?:музыку|воспроизведение|трек|песню)|продолжи\s+(?:музыку|воспроизведение|трек|песню)|продолжай\s+играть)\b",
    re.IGNORECASE,
)

_INLINE_NEXT = re.compile(
    r"\b(?:включи\s+следующ(?:ий|ую)|переключи\s+на\s+следующ(?:ий|ую)|следующ(?:ий|ая|ую)\s+(?:трек|песн\w*))\b",
    re.IGNORECASE,
)

_INLINE_PREV = re.compile(
    r"\b(?:включи\s+предыдущ(?:ий|ую)|переключи\s+на\s+предыдущ(?:ий|ую)|предыдущ(?:ий|ая|ую)\s+(?:трек|песн\w*))\b",
    re.IGNORECASE,
)


def parse_media_command(text: str) -> MediaCommand | None:
    """Parse user text for media transport commands.

    Always checked before content play to prevent 'поставь на стоп' from
    triggering content search.
    """
    if not text:
        return None

    cleaned = " ".join(text.casefold().strip().rstrip(".,!?:;\"'").split())
    if not cleaned:
        return None

    # Strip conversational prefix if present
    cleaned = re.sub(r"^(?:агата|пожалуйста)[\s,]+", "", cleaned, flags=re.IGNORECASE).strip()

    # 1. Exact phrase matches (highest priority)
    if _STOP_EXACT.fullmatch(cleaned):
        return MediaCommand(action="stop", raw_text=text)

    if _PAUSE_EXACT.fullmatch(cleaned):
        return MediaCommand(action="pause", raw_text=text)

    if _RESUME_EXACT.fullmatch(cleaned):
        return MediaCommand(action="resume", raw_text=text)

    if _NEXT_EXACT.fullmatch(cleaned):
        return MediaCommand(action="next", raw_text=text)

    if _PREV_EXACT.fullmatch(cleaned):
        return MediaCommand(action="previous", raw_text=text)

    # 2. Inline phrase matches
    if _INLINE_STOP.search(cleaned):
        return MediaCommand(action="stop", raw_text=text)

    if _INLINE_PAUSE.search(cleaned):
        return MediaCommand(action="pause", raw_text=text)

    if _INLINE_RESUME.search(cleaned):
        return MediaCommand(action="resume", raw_text=text)

    if _INLINE_NEXT.search(cleaned):
        return MediaCommand(action="next", raw_text=text)

    if _INLINE_PREV.search(cleaned):
        return MediaCommand(action="previous", raw_text=text)

    return None
