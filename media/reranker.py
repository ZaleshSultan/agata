"""Optional AI Reranker interface and provider fallback for media search candidates."""

from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

import config
from media.base import MediaResult

logger = logging.getLogger(__name__)


@runtime_checkable
class AIRerankerProvider(Protocol):
    """Interface for interchangeable AI reranking providers (Gemini, OpenRouter, local)."""

    async def rank_results(self, query: str, candidates: list[MediaResult]) -> list[MediaResult]:
        """Rerank top candidates semantically."""
        ...


class DefaultAIReranker:
    """Optional AI reranker that delegates to an interchangeable provider if configured.

    If AI provider is disabled or fails, falls back immediately and cleanly
    to the deterministic ranking without calling the main LLM.
    """

    def __init__(self, provider: AIRerankerProvider | None = None) -> None:
        self._provider = provider

    async def rerank(self, query: str, candidates: list[MediaResult]) -> list[MediaResult]:
        """Rerank up to top 5 candidates. Returns deterministic results if AI is unavailable."""
        if not candidates or len(candidates) <= 1:
            return candidates

        # Check if AI reranker is enabled via config / environment
        reranker_enabled = getattr(config, "MEDIA_AI_RERANKER_ENABLED", False)
        if not reranker_enabled and self._provider is None:
            return candidates

        top_candidates = candidates[:5]
        rest = candidates[5:]

        if self._provider is not None:
            try:
                reranked_top = await self._provider.rank_results(query, top_candidates)
                if reranked_top:
                    return reranked_top + rest
            except Exception as exc:
                logger.warning("[RERANKER] AI provider error: %s (falling back to deterministic ranking)", exc)

        return candidates


_global_reranker: DefaultAIReranker | None = None


def get_ai_reranker() -> DefaultAIReranker:
    """Return the global AI reranker instance."""
    global _global_reranker
    if _global_reranker is None:
        _global_reranker = DefaultAIReranker()
    return _global_reranker
