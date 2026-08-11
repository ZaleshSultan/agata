"""Strict JSON candidate extraction.  This module never accesses SQLite."""

import json
import logging
from typing import Any

from memory.models import MemoryCandidate, MemorySource, MemoryType

logger = logging.getLogger(__name__)


class ExtractionValidationError(ValueError):
    pass


class LLMMemoryExtractor:
    def __init__(self, llm=None) -> None:
        self._llm = llm

    def _client(self):
        if self._llm is None:
            from llm import LocalLLM
            self._llm = LocalLLM()
        return self._llm

    @staticmethod
    def _parse(payload: str) -> list[MemoryCandidate]:
        try:
            data = json.loads(payload)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ExtractionValidationError("invalid JSON") from exc
        if not isinstance(data, dict) or set(data) - {"memories"} or not isinstance(data.get("memories"), list):
            raise ExtractionValidationError("invalid response schema")
        candidates: list[MemoryCandidate] = []
        for item in data["memories"]:
            if not isinstance(item, dict):
                continue
            allowed = {"content", "memory_type", "importance", "confidence", "subject", "entities", "temporary", "expires_at"}
            if set(item) - allowed:
                continue
            content, type_name = item.get("content"), item.get("memory_type")
            importance, confidence = item.get("importance"), item.get("confidence")
            entities = item.get("entities", [])
            if not isinstance(content, str) or not content.strip() or not isinstance(type_name, str):
                continue
            if not isinstance(importance, (int, float)) or not 0 <= importance <= 1 or not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
                continue
            if not isinstance(entities, list) or not all(isinstance(entity, str) for entity in entities):
                continue
            try:
                memory_type = MemoryType(type_name)
            except ValueError:
                continue
            temporary = item.get("temporary", False)
            if not isinstance(temporary, bool):
                continue
            candidates.append(MemoryCandidate(
                content=content.strip(), memory_type=memory_type, importance=float(importance), confidence=float(confidence),
                subject=item.get("subject") if isinstance(item.get("subject"), str) else None,
                entities=entities, temporary=temporary,
                expires_at=item.get("expires_at") if isinstance(item.get("expires_at"), str) else None,
                metadata={"extraction": "llm"},
            ))
        return candidates

    def extract(self, user_message: str, assistant_response: str, context: dict[str, str]) -> list[MemoryCandidate]:
        prompt = f'''Return ONLY JSON: {{"memories": [...]}}. Extract durable facts stated by the USER, not assistant suggestions.
Save explicit memories, stable preferences, projects, goals, decisions, relationships and meaningful events. Do not save questions, jokes, temporary details, or assistant-only claims.
Allowed memory_type: {", ".join(item.value for item in MemoryType)}.
Each memory needs content, memory_type, importance (0..1), confidence (0..1), subject, entities (list), temporary (bool).
<user_message>{user_message}</user_message>
<assistant_response>{assistant_response}</assistant_response>
<relevant_memories>{context.get("relevant_memories", "")}</relevant_memories>
<working_memory>{context.get("working_memory", "")}</working_memory>
<current_datetime>{context.get("current_datetime", "")}</current_datetime>'''
        response = self._client().chat([
            {"role": "system", "content": "You are a conservative memory candidate extractor. Output JSON only."},
            {"role": "user", "content": prompt},
        ], tools=None, think=False)
        if response.get("type") != "text":
            raise ExtractionValidationError("non-text extractor response")
        return self._parse(response.get("content", ""))
