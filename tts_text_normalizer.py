"""
TTS Text Normalizer.

Cleans text prior to XTTS/pyttsx3 speech synthesis:
- Strips Markdown (headings, bold, italics, code blocks, bullet points, links)
- Converts LaTeX expressions and math symbols into natural spoken Russian
- Sanitizes or replaces raw URLs with concise spoken descriptions
- Removes technical brackets, emojis, and unpronounceable formatting artifacts
"""

import logging
import re

logger = logging.getLogger(__name__)

# Common LaTeX Greek letters
_LATEX_GREEK = {
    r"\alpha": "альфа",
    r"\beta": "бета",
    r"\gamma": "гамма",
    r"\delta": "дельта",
    r"\epsilon": "эпсилон",
    r"\lambda": "лямбда",
    r"\mu": "мю",
    r"\pi": "пи",
    r"\sigma": "сигма",
    r"\omega": "омега",
    r"\Delta": "дельта",
    r"\Theta": "тета",
    r"\Sigma": "сигма",
    r"\Omega": "омега",
}

# LaTeX math operators
_LATEX_SYMBOLS = {
    r"\times": " умножить на ",
    r"\cdot": " умножить на ",
    r"\div": " разделить на ",
    r"\pm": " плюс-минус ",
    r"\approx": " приблизительно равно ",
    r"\neq": " не равно ",
    r"\ne": " не равно ",
    r"\leq": " меньше либо равно ",
    r"\le": " меньше либо равно ",
    r"\geq": " больше либо равно ",
    r"\ge": " больше либо равно ",
    r"\infty": " бесконечность ",
    r"\rightarrow": " стремится к ",
}

# Domain name mappings for human-friendly spoken URLs
_URL_DOMAIN_NAMES = {
    "youtube.com": "Ютуб",
    "youtu.be": "Ютуб",
    "google.com": "Гугл",
    "github.com": "Гитхаб",
    "telegram.org": "Телеграм",
    "t.me": "Телеграм",
    "spotify.com": "Спотифай",
    "vk.com": "ВКонтакте",
    "yandex.ru": "Яндекс",
    "ya.ru": "Яндекс",
    "wikipedia.org": "Википедия",
    "reddit.com": "Реддит",
}


def _replace_url_match(match: re.Match) -> str:
    """Replace a raw URL with a friendly spoken description."""
    url = match.group(0).lower()
    for domain, spoken in _URL_DOMAIN_NAMES.items():
        if domain in url:
            return f"ссылка на {spoken}"
    return "ссылка"


def _normalize_latex_block(content: str) -> str:
    """Normalize a LaTeX or formula snippet into spoken Russian words."""
    res = content

    # Greek letters
    for tex, word in _LATEX_GREEK.items():
        res = res.replace(tex, f" {word} ")

    # Operators
    for tex, word in _LATEX_SYMBOLS.items():
        res = res.replace(tex, word)

    # \frac{a}{b} -> a разделить на b
    res = re.sub(r"\\frac\{([^}]+)\}\{([^}]+)\}", r"\1 разделить на \2", res)

    # \sqrt{x} -> корень из x
    res = re.sub(r"\\sqrt\{([^}]+)\}", r"корень из \1", res)
    res = re.sub(r"\\sqrt\s*([a-zA-Z0-9]+)", r"корень из \1", res)

    # Powers ^2, ^3, ^n, ^{n}
    res = re.sub(r"\^\{2\}|\^2\b", " в квадрате", res)
    res = re.sub(r"\^\{3\}|\^3\b", " в кубе", res)
    res = re.sub(r"\^\{([^}]+)\}", r" в степени \1", res)
    res = re.sub(r"\^([a-zA-Z0-9]+)", r" в степени \1", res)

    # Subscripts _{0}, _0
    res = re.sub(r"_\{([^}]+)\}", r" с индексом \1", res)
    res = re.sub(r"_([a-zA-Z0-9]+)", r" с индексом \1", res)

    # Remove remaining TeX backslashes
    res = re.sub(r"\\[a-zA-Z]+", "", res)

    # Replace basic math symbols
    res = res.replace("=", " равно ")
    res = res.replace("+", " плюс ")
    res = re.sub(r"(?<=\w|\d)\s*-\s*(?=\w|\d)", " минус ", res)
    res = res.replace("*", " умножить на ")
    res = res.replace("/", " разделить на ")

    return res


def normalize_text_for_tts(text: str) -> str:
    """
    Clean and transform text into a natural spoken form for TTS.

    Args:
        text: Raw text (possibly containing Markdown, LaTeX, formulas, URLs).

    Returns:
        Spoken-friendly plain text in Russian.
    """
    if not text or not text.strip():
        return ""

    cleaned = text

    # 1. Remove code blocks
    cleaned = re.sub(r"```[\s\S]*?```", " фрагмент кода ", cleaned)

    # 2. Markdown links: [Title](url) -> Title
    cleaned = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", cleaned)

    # 3. Inline LaTeX: $$...$$ and $...$
    cleaned = re.sub(r"\$\$([\s\S]+?)\$\$", lambda m: _normalize_latex_block(m.group(1)), cleaned)
    cleaned = re.sub(r"\$([^\$\n]+?)\$", lambda m: _normalize_latex_block(m.group(1)), cleaned)

    # 4. Raw URLs
    cleaned = re.sub(r"https?://[^\s)\]]+", _replace_url_match, cleaned)

    # 5. Markdown headings (# Heading -> Heading)
    cleaned = re.sub(r"^#{1,6}\s*", "", cleaned, flags=re.MULTILINE)

    # 6. Markdown bold / italic / strikethrough / inline code
    cleaned = re.sub(r"`([^`]+)`", r"\1", cleaned)
    cleaned = re.sub(r"\*\*([^*]+)\*\*", r"\1", cleaned)
    cleaned = re.sub(r"__([^_]+)__", r"\1", cleaned)
    cleaned = re.sub(r"\*([^*]+)\*", r"\1", cleaned)
    cleaned = re.sub(r"_([^_]+)_", r"\1", cleaned)
    cleaned = re.sub(r"~~([^~]+)~~", r"\1", cleaned)

    # 7. List markers at line start
    cleaned = re.sub(r"^\s*[-*+]\s+", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"^\s*\d+\.\s+", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"^\s*>\s*", "", cleaned, flags=re.MULTILINE)

    # 8. Standalone math expressions like "D = b^2 - 4ac" or "12 + 17 = 29"
    if "=" in cleaned:
        cleaned = re.sub(
            r"([^\n=]+)\s*=\s*([^.,\n]+)",
            lambda m: f"{_normalize_latex_block(m.group(1))} равно {_normalize_latex_block(m.group(2))}",
            cleaned,
        )

    # 9. Clean remaining technical symbols
    cleaned = re.sub(r"[{}\[\]\\|~<>]", " ", cleaned)

    # 10. Collapse multiple spaces and trim
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    return cleaned
