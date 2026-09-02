"""Deterministic ranking for media search results without LLM."""

from __future__ import annotations

import re
from media.base import MediaResult


def _tokenize(text: str) -> set[str]:
    """Extract lowercase alpha-numeric tokens."""
    return set(re.findall(r"\w+", text.lower()))


def rank_media_results(query: str, items: list[MediaResult]) -> list[MediaResult]:
    """Score and rank media search items deterministically.

    Scoring criteria:
    - Exact query substring match (+10.0)
    - Jaccard token overlap (+5.0 * overlap_ratio)
    - Word prefix matches (+1.0 per matching prefix)
    - Position penalty (-0.1 * position)
    """
    if not items:
        return []

    q_lower = query.lower().strip()
    q_tokens = _tokenize(query)

    scored: list[MediaResult] = []
    for item in items:
        score = 0.0
        t_lower = item.title.lower()
        t_tokens = _tokenize(item.title)

        # 1. Exact query match in title
        if q_lower in t_lower:
            score += 10.0

        # 2. Token overlap (Jaccard-like)
        if q_tokens:
            common = q_tokens.intersection(t_tokens)
            overlap_ratio = len(common) / len(q_tokens)
            score += 5.0 * overlap_ratio

        # 3. Position penalty (first results get a slight edge)
        score -= 0.1 * item.position

        item_with_score = MediaResult(
            url=item.url,
            title=item.title,
            duration=item.duration,
            position=item.position,
            score=round(score, 3),
        )
        scored.append(item_with_score)

    scored.sort(key=lambda x: x.score, reverse=True)
    return scored
