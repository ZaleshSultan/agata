"""Conservative, dependency-free extraction of durable user memories."""

import re

from memory.models import Memory, MemorySource, MemoryType

_EXPLICIT = re.compile(r"^\s*(?:запомни(?:,?\s*что)?|помни(?:,?\s*что)?|remember(?:\s+that)?)\s+(.+?)\s*[.!]*$", re.I)
_DURABLE = re.compile(r"(?:я\s+(?:(?:больше\s+)?не\s+)?(?:работаю|занимаюсь|хочу|предпочитаю|люблю)|мой\s+(?:проект|план|цель)|i\s+(?:work|want|prefer|am working))", re.I)


def extract_candidates(user_message: str, assistant_response: str = "", current_context: str = "") -> list[Memory]:
    """Return only high-signal candidates; questions and model guesses are ignored."""
    text = user_message.strip()
    explicit = _EXPLICIT.match(text)
    if explicit:
        return [Memory(None, explicit.group(1).strip(), MemoryType.FACT, .95, .98, source=MemorySource.USER_EXPLICIT, metadata={"extraction": "explicit"})]
    if "?" in text or len(text) < 20 or not _DURABLE.search(text):
        return []
    lowered = text.lower()
    memory_type = MemoryType.PROJECT if any(word in lowered for word in ("проект", "работаю", "занимаюсь", "agata", "агат")) else MemoryType.GOAL
    return [Memory(None, text.rstrip(".!"), memory_type, .75 if memory_type is MemoryType.PROJECT else .7, .95, source=MemorySource.CONVERSATION, metadata={"extraction": "rule_based"})]
