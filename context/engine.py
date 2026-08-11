"""Bounded cognitive context assembly; no SQLite or embedding details live here."""

import logging
from datetime import datetime

import config
from context.models import ContextBundle
from memory.repository import user_timezone

logger = logging.getLogger(__name__)


class ContextEngine:
    def __init__(self, memory_manager) -> None:
        self.memory = memory_manager

    def build(self, user_message: str, conversation_history: list[dict]) -> ContextBundle:
        try:
            memories = self.memory.recall(user_message, top_k=config.MEMORY_RECALL_LIMIT)
            chunks, used = [], 0
            for memory in memories:
                line = f"- [{memory.memory_type}] {memory.content}"
                if used + len(line) > config.MEMORY_CONTEXT_CHAR_BUDGET:
                    break
                chunks.append(line); used += len(line)
            history = conversation_history[-config.MEMORY_RECENT_CONVERSATION_TURNS * 2:]
            bundle = ContextBundle(history, self.memory.get_working_context(), "\n".join(chunks), datetime.now(user_timezone()).isoformat())
            logger.info("[CONTEXT] memories=%d working_memory_chars=%d history_messages=%d chars=%d", len(memories), len(bundle.working_memory), len(history), used)
            return bundle
        except Exception as exc:
            logger.warning("[CONTEXT] build failed: %s", exc)
            return ContextBundle([], "", "", datetime.now(user_timezone()).isoformat())
