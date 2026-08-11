"""Backward-compatible memory tools backed by MemoryManager."""

import logging

from memory import MemoryManager
from memory.models import MemorySource

logger = logging.getLogger(__name__)
_memory: MemoryManager | None = None


def _get_memory() -> MemoryManager:
    global _memory
    if _memory is None:
        _memory = MemoryManager()
    return _memory


def write_diary(entry: str) -> str:
    """Compatibility tool: explicitly persist a user fact or event."""
    try:
        memory = _get_memory().remember(entry, importance=.95, confidence=.98, source=MemorySource.USER_EXPLICIT)
        return f"Записано в память (id={memory.id}): {memory.content[:100]}"
    except Exception as exc:
        logger.exception("[MEMORY] write_diary failed")
        return f"Ошибка записи в память: {exc}"


def search_diary(query: str) -> str:
    """Compatibility tool: explicit lookup through MemoryManager.recall()."""
    try:
        results = _get_memory().recall(query, top_k=5)
        if not results:
            return "В памяти ничего не найдено по этому запросу."
        return "\n".join(f"{i}. [{m.created_at}] (score: {m.score:.3f}) {m.content}" for i, m in enumerate(results, 1))
    except Exception as exc:
        logger.exception("[MEMORY] search_diary failed")
        return f"Ошибка поиска в памяти: {exc}"


WRITE_DIARY_DEFINITION = {
    "function": {"name": "write_diary", "description": "Сохранить важный факт в долгосрочную память.", "parameters": {"type": "object", "properties": {"entry": {"type": "string"}}, "required": ["entry"]}},
    "requires_confirmation": False,
}
SEARCH_DIARY_DEFINITION = {
    "function": {"name": "search_diary", "description": "Явно найти сохранённые воспоминания.", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}},
    "requires_confirmation": False,
}
