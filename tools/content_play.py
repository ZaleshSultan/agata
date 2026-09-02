"""Tool for searching and playing media content (YouTube, SoundCloud)."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from media.base import ActionResult, MediaResult
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

    if platform_key in {"soundcloud", "саундклауд"}:
        provider = SoundCloudProvider()
        canonical_platform = "soundcloud"
    else:
        provider = YouTubeProvider()
        canonical_platform = "youtube"

    logger.info("[CONTENT] Searching '%s' on %s", query, canonical_platform)
    try:
        results = await provider.search(query)
        if not results:
            return ActionResult(
                success=False,
                platform=canonical_platform,
                query=query,
                stage="search",
                reason=f"Ничего не найдено по запросу '{query}' на {canonical_platform}.",
            )

        top_result = results[0]
        logger.info("[CONTENT] Top ranked: '%s' (%s, score=%s)", top_result.title, top_result.url, top_result.score)
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

    Returns JSON string with execution results.
    """
    logger.info("[TOOL] play_content(query='%s', platform='%s')", query, platform)
    try:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # If called inside an existing running event loop
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                action_result = pool.submit(asyncio.run, _async_play_content(query, platform)).result()
        else:
            action_result = asyncio.run(_async_play_content(query, platform))

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
