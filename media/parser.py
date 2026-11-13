"""Deterministic parser for media content queries and content-type detection."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(slots=True)
class ParsedContentQuery:
    raw_query: str
    title: str
    artist: str = ""
    content_type: str = "general"
    search_query: str = ""


# Prefixes to strip: "поставь", "включи", "сыграй", "воспроизведи", "найди"
_CONTENT_PREFIXES = re.compile(
    r"^\s*(?:поставь|включи|вруби|сыграй|проиграй|запусти|воспроизведи|воспроизвести|послушай|послушать|найди)\s+",
    re.IGNORECASE,
)

# Content type detection keywords
_TYPE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "lecture",
        re.compile(
            r"\b(?:лекци\w*|курс\w*|урок\w*|семинар\w*|вебинар\w*|доклад\w*|lecture\w*|course\w*|tutorial\w*)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "podcast",
        re.compile(r"\b(?:подкаст\w*|интервью|podcast\w*)\b", re.IGNORECASE),
    ),
    (
        "concert",
        re.compile(r"\b(?:концерт\w*|лайв|live|concert\w*)\b", re.IGNORECASE),
    ),
    (
        "movie",
        re.compile(r"\b(?:фильм\w*|кино|сериал\w*|документалк\w*|movie|film)\b", re.IGNORECASE),
    ),
    (
        "music",
        re.compile(
            r"\b(?:песн\w*|трек\w*|музык\w*|клип\w*|сингл\w*|альбом\w*|song\w*|track\w*|music\w*)\b",
            re.IGNORECASE,
        ),
    ),
]


def detect_content_type(text: str) -> str:
    """Deterministically detect intended media type from text."""
    for c_type, pattern in _TYPE_PATTERNS:
        if pattern.search(text):
            return c_type
    return "general"


def _normalize_russian_noun(word: str) -> str:
    """Normalize simple Russian accusative/genitive noun endings for song titles and artists."""
    w = word.strip()
    if not w:
        return ""

    # Common song title accusatives: "Монополию" -> "Монополия"
    if w.endswith("ию"):
        return w[:-2] + "ия"
    if w.endswith("ую"):
        return w[:-2] + "ая"
    if w.endswith("ю"):
        return w[:-1] + "я"
    if w.endswith("у") and len(w) > 3:
        return w[:-1] + "а"

    # Common artist genitives: "Монеточки" -> "Монеточка", "Земфиры" -> "Земфира"
    if w.endswith("очки"):
        return w[:-4] + "очка"
    if w.endswith("ечки"):
        return w[:-4] + "ечка"
    if w.endswith("ки") and len(w) > 3:
        return w[:-2] + "ка"
    if w.endswith("цы") and len(w) > 3:
        return w[:-2] + "ца"
    if w.endswith("ы") and len(w) > 3:
        return w[:-1] + "а"
    if w.endswith("и") and len(w) > 3:
        return w[:-1] + "я"
    if w.endswith("а") and len(w) > 4 and w[-2] in "тдкпрст":  # e.g., Скриптонита -> Скриптонит
        return w[:-1]

    return w


def _normalize_title_phrase(title: str) -> str:
    """Clean and normalize title phrase."""
    words = title.strip().split()
    if not words:
        return ""
    # Normalize the last or first word if it looks inflected
    normalized_words = [_normalize_russian_noun(w) if i == 0 or i == len(words) - 1 else w for i, w in enumerate(words)]
    return " ".join(normalized_words)


def parse_content_query(text: str) -> ParsedContentQuery:
    """Parse user command into structured title, artist, content type, and search query.

    Handles:
    - [title] от [artist] (e.g. "Монополия от Монеточки", "Blinding Lights от The Weeknd")
    - [title] исполнителя/группы [artist]
    - [title] [artist] in genitive (e.g. "Монополию Монеточки")
    - [artist] [title] (e.g. "The Weeknd Blinding Lights", "Монеточка Монополия")
    - "song title" "artist"
    """
    clean = _CONTENT_PREFIXES.sub("", text.strip()).strip().rstrip(".,!?:;\"'")
    c_type = detect_content_type(clean)

    # 1. Quoted patterns: "title" "artist"
    quoted_match = re.findall(r'["\']([^"\']+)["\']', clean)
    if len(quoted_match) >= 2:
        title = quoted_match[0].strip()
        artist = quoted_match[1].strip()
        search_query = f'"{title}" "{artist}"'
        return ParsedContentQuery(raw_query=text, title=title, artist=artist, content_type=c_type or "music", search_query=search_query)

    # 2. [title] от [artist]
    ot_match = re.search(r"^(?P<title>.+?)\s+(?:от|by)\s+(?P<artist>[^,]+)$", clean, re.IGNORECASE)
    if ot_match:
        raw_title = ot_match.group("title").strip().strip(" .,!?:;\"'")
        raw_artist = ot_match.group("artist").strip().strip(" .,!?:;\"'")
        # Strip media noun prefix from title if present: "песню Монополия" -> "Монополия"
        raw_title = re.sub(r"^(?:песн\w*|трек\w*|музык\w*|клип\w*)\s+", "", raw_title, flags=re.IGNORECASE).strip()
        norm_title = _normalize_title_phrase(raw_title)
        norm_artist = _normalize_russian_noun(raw_artist)
        search_query = f"{norm_artist} {norm_title}"
        return ParsedContentQuery(
            raw_query=text,
            title=norm_title or raw_title,
            artist=norm_artist or raw_artist,
            content_type=c_type if c_type != "general" else "music",
            search_query=search_query,
        )

    # 3. [title] исполнителя / группы / автора [artist]
    isp_match = re.search(r"^(?P<title>.+?)\s+(?:исполнителя|группы|автора)\s+(?P<artist>[^,]+)$", clean, re.IGNORECASE)
    if isp_match:
        raw_title = isp_match.group("title").strip().strip(" .,!?:;\"'")
        raw_artist = isp_match.group("artist").strip().strip(" .,!?:;\"'")
        raw_title = re.sub(r"^(?:песн\w*|трек\w*|музык\w*)\s+", "", raw_title, flags=re.IGNORECASE).strip()
        norm_title = _normalize_title_phrase(raw_title)
        norm_artist = _normalize_russian_noun(raw_artist)
        search_query = f"{norm_artist} {norm_title}"
        return ParsedContentQuery(
            raw_query=text,
            title=norm_title or raw_title,
            artist=norm_artist or raw_artist,
            content_type=c_type if c_type != "general" else "music",
            search_query=search_query,
        )

    # 4. [title] [artist] in genitive: e.g. "Монополию Монеточки" or "Кукушку Кино"
    # Match two words or title + artist ending in genitive
    gen_match = re.search(r"^(?P<title>[А-Яа-яA-Za-z0-9\s]+?)\s+(?P<artist>[А-Яа-яA-Za-z]+(?:очки|ечки|ки|цы|и|ы|а|я))$", clean, re.IGNORECASE)
    if gen_match and c_type in {"music", "general"}:
        cand_title = gen_match.group("title").strip()
        cand_artist = gen_match.group("artist").strip()
        # Avoid breaking phrases like "про нейробиологию"
        if not cand_title.lower().startswith("про ") and not cand_title.lower().startswith("о "):
            cand_title = re.sub(r"^(?:песн\w*|трек\w*|музык\w*)\s+", "", cand_title, flags=re.IGNORECASE).strip()
            norm_title = _normalize_russian_noun(cand_title)
            norm_artist = _normalize_russian_noun(cand_artist)
            search_query = f"{norm_artist} {norm_title}"
            return ParsedContentQuery(
                raw_query=text,
                title=norm_title or cand_title,
                artist=norm_artist or cand_artist,
                content_type="music",
                search_query=search_query,
            )

    # 5. Default fallback: title = full clean query
    stripped = re.sub(r"^(?:песн\w*|трек\w*|музык\w*|видео|клип\w*)\s+", "", clean, flags=re.IGNORECASE).strip()
    norm = _normalize_title_phrase(stripped or clean)
    clean_title = norm or clean
    search_query = clean_title
    return ParsedContentQuery(
        raw_query=text,
        title=clean_title,
        artist="",
        content_type=c_type,
        search_query=search_query,
    )

