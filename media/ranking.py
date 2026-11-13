"""Deterministic type-specific ranking for media search results without LLM."""

from __future__ import annotations

import re
from media.base import MediaResult


def _tokenize(text: str) -> set[str]:
    """Extract lowercase alpha-numeric tokens."""
    return set(re.findall(r"\w+", text.lower()))


def _parse_duration_seconds(duration_str: str) -> int:
    """Parse duration strings like '1:15:30', '12:45', '0:55' into seconds."""
    if not duration_str:
        return 0
    parts = duration_str.strip().split(":")
    try:
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        if len(parts) == 2:
            return int(parts[0]) * 60 + int(parts[1])
        if len(parts) == 1 and parts[0].isdigit():
            return int(parts[0])
    except ValueError:
        pass
    return 0


# Negative indicators for original music tracks
_MUSIC_NEGATIVE_PATTERNS = re.compile(
    r"\b(?:дисс\s+на|пароди\w*|реакци\w*|караоке|минусовк\w*|минус\b|cover\b|кавер\b|remix\b|ремикс\b)\b",
    re.IGNORECASE,
)

# Educational/course indicators for lectures
_LECTURE_INDICATORS = re.compile(
    r"\b(?:лекци\w*|курс\w*|урок\w*|семинар\w*|вебинар\w*|доклад\w*|полный\s+курс|lecture\w*|course\w*|university|академи\w*)\b",
    re.IGNORECASE,
)

# Short-form video indicators (penalized for long-form content)
_SHORT_INDICATORS = re.compile(
    r"\b(?:shorts?|шортс|клип\w*|мем\w*|meme\w*|тикток|tiktok)\b",
    re.IGNORECASE,
)


def score_music_result(
    title: str,
    channel: str,
    description: str,
    duration_sec: int,
    verified: bool,
    position: int,
    query_title: str,
    query_artist: str,
    raw_query: str,
) -> float:
    """Score candidates for music track queries."""
    score = 0.0
    t_lower = title.lower()
    c_lower = channel.lower()
    d_lower = description.lower()

    # 1. Exact title match
    if query_title and query_title.lower() in t_lower:
        score += 15.0

    # 2. Artist match in title or channel
    if query_artist:
        artist_lower = query_artist.lower()
        if artist_lower in t_lower:
            score += 12.0
        if artist_lower in c_lower:
            score += 10.0

    # 3. Both title and artist in video title
    if query_title and query_artist:
        if query_title.lower() in t_lower and query_artist.lower() in t_lower:
            score += 20.0

    # 4. Verified official channel
    if verified:
        score += 6.0
    if "official" in t_lower or "official" in c_lower or "официальн" in t_lower:
        score += 4.0

    # 5. Negative indicators penalty (unless user explicitly searched for them)
    if not any(k in raw_query.lower() for k in ["дисс", "пародия", "реакция", "караоке", "кавер", "cover", "remix"]):
        if _MUSIC_NEGATIVE_PATTERNS.search(t_lower):
            score -= 20.0

    # 6. Song duration signals (typically 2 to 6 minutes)
    if 100 <= duration_sec <= 420:
        score += 2.0
    elif duration_sec > 1200:  # Full albums/compilations when single song requested
        score -= 5.0
    elif 0 < duration_sec < 60:  # Very short previews/shorts
        score -= 10.0

    # 7. Position tie-breaker
    score -= 0.1 * position
    return score


def score_lecture_result(
    title: str,
    channel: str,
    description: str,
    duration_sec: int,
    position: int,
    raw_query: str,
) -> float:
    """Score candidates for educational lecture/course queries."""
    score = 0.0
    t_lower = title.lower()
    d_lower = description.lower()
    q_lower = raw_query.lower()

    # Clean topic words (excluding "лекция", "про", "о", "курс")
    topic_words = [
        w for w in re.findall(r"\w+", q_lower)
        if w not in {"лекция", "лекции", "лекцию", "про", "о", "об", "курс", "курсы", "урок", "видео", "поставь", "включи"}
    ]

    # 1. Topic words coverage in title
    if topic_words:
        matched_topic = sum(1 for w in topic_words if w in t_lower)
        coverage = matched_topic / len(topic_words)
        score += 15.0 * coverage
        # Also check description
        matched_desc = sum(1 for w in topic_words if w in d_lower)
        score += 4.0 * (matched_desc / len(topic_words))

    # 2. Educational / lecture / course indicators
    if _LECTURE_INDICATORS.search(t_lower):
        score += 12.0
    elif _LECTURE_INDICATORS.search(d_lower):
        score += 5.0

    # 3. Duration signals for lecture: long-form is strongly preferred
    if duration_sec >= 2700:  # > 45 minutes (full university lecture)
        score += 10.0
    elif duration_sec >= 1200:  # > 20 minutes
        score += 7.0
    elif duration_sec >= 600:  # > 10 minutes
        score += 3.0
    elif 0 < duration_sec < 180:  # < 3 minutes (short/clip)
        score -= 15.0

    # 4. Heavy penalty for short-form memes or clips
    if _SHORT_INDICATORS.search(t_lower):
        score -= 20.0

    # 5. Position tie-breaker
    score -= 0.1 * position
    return score


def score_general_result(
    title: str,
    description: str,
    position: int,
    raw_query: str,
) -> float:
    """Score generic media items."""
    score = 0.0
    t_lower = title.lower()
    q_lower = raw_query.lower().strip()
    q_tokens = _tokenize(raw_query)

    # 1. Exact query match in title
    if q_lower in t_lower:
        score += 10.0

    # 2. Token overlap ratio
    if q_tokens:
        t_tokens = _tokenize(title)
        common = q_tokens.intersection(t_tokens)
        score += 6.0 * (len(common) / len(q_tokens))

    # 3. Position penalty
    score -= 0.1 * position
    return score


def rank_media_results(
    query: str,
    items: list[MediaResult],
    content_type: str = "general",
    query_title: str = "",
    query_artist: str = "",
) -> list[MediaResult]:
    """Score and rank media search items deterministically using type-specific profiles.

    Supports 'music', 'lecture', 'podcast', and 'general' content types.
    """
    if not items:
        return []

    scored: list[MediaResult] = []
    for item in items:
        dur_sec = _parse_duration_seconds(item.duration)

        if content_type == "music":
            s = score_music_result(
                title=item.title,
                channel=item.channel,
                description=item.description,
                duration_sec=dur_sec,
                verified=item.verified_channel,
                position=item.position,
                query_title=query_title or query,
                query_artist=query_artist,
                raw_query=query,
            )
        elif content_type == "lecture":
            s = score_lecture_result(
                title=item.title,
                channel=item.channel,
                description=item.description,
                duration_sec=dur_sec,
                position=item.position,
                raw_query=query,
            )
        else:
            s = score_general_result(
                title=item.title,
                description=item.description,
                position=item.position,
                raw_query=query,
            )

        scored.append(
            MediaResult(
                url=item.url,
                title=item.title,
                duration=item.duration,
                position=item.position,
                score=round(s, 3),
                channel=item.channel,
                description=item.description,
                verified_channel=item.verified_channel,
                content_type=content_type,
            )
        )

    scored.sort(key=lambda x: x.score, reverse=True)
    return scored
