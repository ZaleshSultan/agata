"""
Веб-поиск через DuckDuckGo (без API-ключа).
"""

import logging

logger = logging.getLogger(__name__)

try:
    from ddgs import DDGS
except ImportError:
    try:
        from duckduckgo_search import DDGS  # type: ignore[no-redef]
    except ImportError:
        DDGS = None  # type: ignore[assignment]


def web_search(query: str) -> str:
    """
    Ищет в интернете актуальную информацию по запросу.
    Используй, когда вопрос касается текущих событий, свежих фактов,
    или когда не уверена в ответе из собственных знаний.
    """
    if DDGS is None:
        return "Модуль ddgs не установлен. Установите пакетом: pip install ddgs"

    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
        if not results:
            return "Поиск не дал результатов."
        formatted = []
        for r in results:
            formatted.append(f"- {r.get('title', '')}: {r.get('body', '')} ({r.get('href', '')})")
        return "\n".join(formatted)
    except Exception as e:
        logger.error(f"Ошибка веб-поиска: {e}")
        return f"Не удалось выполнить поиск: {e}"


# ── JSON Schema для tool calling ──────────────────────────────────────────

TOOL_DEFINITION: dict = {
    "function": {
        "name": "web_search",
        "description": (
            "Ищет в интернете актуальную информацию по запросу. "
            "Используй, когда вопрос касается текущих событий, свежих новостей, "
            "или когда не уверена в ответе из собственных знаний."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Поисковый запрос для DuckDuckGo.",
                },
            },
            "required": ["query"],
        },
    },
    "requires_confirmation": False,
}
