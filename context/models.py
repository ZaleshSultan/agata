"""Structured, bounded context passed from ContextEngine to prompt builders."""

from dataclasses import dataclass


@dataclass(slots=True)
class ContextBundle:
    recent_conversation: list[dict]
    working_memory: str
    relevant_memories: str
    current_datetime: str
    current_state: str = ""
