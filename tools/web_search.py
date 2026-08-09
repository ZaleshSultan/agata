"""
Веб-поиск через DuckDuckGo (без API-ключа).
"""

import json
import logging

logger = logging.getLogger(__name__)

# A compact result set is enough for one final synthesis and avoids sending a
# large, repetitive web context to the local model.
MAX_RESULTS = 3
MAX_SNIPPET_CHARS = 500
MAX_CONTEXT_CHARS = 2_400

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
        return json.dumps(
            {"success": False, "error": "Модуль ddgs не установлен. Установите пакетом: pip install ddgs"},
            ensure_ascii=False,
        )

    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=MAX_RESULTS))
        if not results:
            return json.dumps({"success": True, "results": []}, ensure_ascii=False)

        compact_results: list[dict[str, str]] = []
        context_size = 0
        for r in results:
            title = str(r.get("title", "")).strip()
            snippet = str(r.get("body", "")).strip()[:MAX_SNIPPET_CHARS]
            url = str(r.get("href", "")).strip()
            item_size = len(title) + len(snippet) + len(url)
            if compact_results and context_size + item_size > MAX_CONTEXT_CHARS:
                break
            compact_results.append({"title": title, "snippet": snippet, "url": url})
            context_size += item_size

        logger.info("Веб-поиск: %d компактных результатов, %d символов", len(compact_results), context_size)
        return json.dumps({"success": True, "results": compact_results}, ensure_ascii=False)
    except Exception as e:
        logger.error("Ошибка веб-поиска: %s", e)
        return json.dumps({"success": False, "error": f"Не удалось выполнить поиск: {e}"}, ensure_ascii=False)


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
