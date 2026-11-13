"""Tool for searching and playing media content (YouTube, SoundCloud) with parsing and ranking."""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
from typing import Any

from media.base import ActionResult, MediaResult
from media.parser import parse_content_query
from media.reranker import get_ai_reranker
from media.soundcloud import SoundCloudProvider
from media.youtube import YouTubeProvider

logger = logging.getLogger(__name__)

TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "play_content",
        "description": "Найти и запустить видео, трек или лекцию на YouTube или SoundCloud.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Поисковый запрос для поиска контента (название трека, лекции, видео).",
                },
                "platform": {
                    "type": "string",
                    "enum": ["youtube", "soundcloud"],
                    "description": "Платформа для воспроизведения (по умолчанию 'youtube').",
                },
            },
            "required": ["query"],
        },
    },
    "requires_confirmation": False,
}


async def _async_play_content(query: str, platform: str = "youtube") -> ActionResult:
    platform_key = platform.lower().strip() if platform else "youtube"

    # 1. Parse content query deterministically (extract title, artist, content type)
    parsed = parse_content_query(query)
    logger.info(
        "[CONTENT] Parsed query: title='%s', artist='%s', type='%s', search_query='%s'",
        parsed.title,
        parsed.artist,
        parsed.content_type,
        parsed.search_query,
    )

    if platform_key in {"soundcloud", "саундклауд"}:
        provider = SoundCloudProvider()
        canonical_platform = "soundcloud"
    else:
        provider = YouTubeProvider()
        canonical_platform = "youtube"

    logger.info("[CONTENT] Searching '%s' on %s", parsed.search_query, canonical_platform)
    try:
        results = await provider.search(
            query=parsed.search_query,
            content_type=parsed.content_type,
            query_title=parsed.title,
            query_artist=parsed.artist,
        )
        if not results:
            return ActionResult(
                success=False,
                platform=canonical_platform,
                query=query,
                stage="search",
                reason=f"Ничего не найдено по запросу '{query}' на {canonical_platform}.",
            )

        # 2. Optional AI reranking (top 5), falls back to deterministic order
        reranker = get_ai_reranker()
        reranked = await reranker.rerank(parsed.search_query, results)
        top_result = reranked[0]

        logger.info(
            "[CONTENT] Selected result: '%s' (%s, score=%s, channel='%s')",
            top_result.title,
            top_result.url,
            top_result.score,
            top_result.channel,
        )

        # 3. Real playback & verification via persistent MediaSession
        action_result = await provider.play(top_result)
        return action_result

    except Exception as exc:
        logger.error("[CONTENT] Playback execution failed: %s", exc)
        return ActionResult(
            success=False,
            platform=canonical_platform,
            query=query,
            stage="execution",
            reason=str(exc),
        )


def play_content(query: str, platform: str = "youtube") -> str:
    """Synchronous entry point for the play_content tool.

    Executes on the dedicated persistent browser event loop to preserve Playwright context.
    """
    logger.info("[TOOL] play_content(query='%s', platform='%s')", query, platform)
    from browser.manager import run_browser_coro
    try:
        action_result = run_browser_coro(_async_play_content(query, platform), timeout=45.0)
        return json.dumps(action_result.as_dict(), ensure_ascii=False)
    except Exception as exc:
        logger.error("[TOOL] play_content error: %s", exc)
        return json.dumps(
            {
                "success": False,
                "action": "play_content",
                "platform": platform,
                "query": query,
                "stage": "tool_runner",
                "reason": str(exc),
            },
            ensure_ascii=False,
        )
