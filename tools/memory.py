"""
Инструменты памяти — дневник Агаты.

write_diary: сохранить новый факт/событие в долгосрочную память.
search_diary: семантический поиск по прошлым записям.

Побочные эффекты:
- write_diary: запись в SQLite + запрос эмбеддинга через Ollama.
- search_diary: чтение SQLite + запрос эмбеддинга через Ollama.
"""

import logging

from llm import LocalLLM
from memory import diary

logger = logging.getLogger(__name__)

# Ленивая инициализация LLM-клиента для эмбеддингов
_llm: LocalLLM | None = None


def _get_llm() -> LocalLLM:
    """Получить или создать экземпляр LLM-клиента."""
    global _llm
    if _llm is None:
        _llm = LocalLLM()
    return _llm


def write_diary(entry: str) -> str:
    """
    Записать важный факт или событие в дневник.

    Args:
        entry: Текст записи для дневника.

    Returns:
        Подтверждение записи.
    """
    try:
        embedding = _get_llm().embed(entry)
        entry_id = diary.add_entry(text=entry, embedding=embedding)
        result = f"Записано в дневник (id={entry_id}): {entry[:100]}"
        logger.info(result)
        return result
    except Exception as exc:
        error_msg = f"Ошибка записи в дневник: {exc}"
        logger.error(error_msg)
        return error_msg


def search_diary(query: str) -> str:
    """
    Поиск по дневнику — найти релевантные прошлые записи.

    Args:
        query: Поисковый запрос (вопрос, тема, ключевые слова).

    Returns:
        Найденные записи или сообщение "ничего не найдено".
    """
    try:
        query_embedding = _get_llm().embed(query)
        results = diary.search(query_embedding=query_embedding, top_k=5)
        if not results:
            return "В дневнике ничего не найдено по этому запросу."

        lines: list[str] = []
        for i, r in enumerate(results, 1):
            lines.append(
                f"{i}. [{r['created_at']}] (схожесть: {r['similarity']}) {r['text']}"
            )
        result = "\n".join(lines)
        logger.info("Найдено %d записей по запросу '%s'", len(results), query[:50])
        return result
    except Exception as exc:
        error_msg = f"Ошибка поиска в дневнике: {exc}"
        logger.error(error_msg)
        return error_msg


# ── JSON Schema для tool calling ──────────────────────────────────────────

WRITE_DIARY_DEFINITION: dict = {
    "function": {
        "name": "write_diary",
        "description": (
            "Записать важный факт, событие или информацию о пользователе в дневник "
            "(долгосрочную память). Используй, когда пользователь сообщает что-то, "
            "что стоит запомнить: предпочтения, задачи, события, факты о себе."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "entry": {
                    "type": "string",
                    "description": "Текст записи для дневника. Формулируй кратко и фактологично.",
                },
            },
            "required": ["entry"],
        },
    },
    "requires_confirmation": False,
}

SEARCH_DIARY_DEFINITION: dict = {
    "function": {
        "name": "search_diary",
        "description": (
            "Поиск по дневнику (долгосрочной памяти). Используй, когда нужно вспомнить "
            "прошлые договорённости, факты о пользователе, записанные события или предпочтения."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Поисковый запрос — что нужно найти в дневнике.",
                },
            },
            "required": ["query"],
        },
    },
    "requires_confirmation": False,
}
