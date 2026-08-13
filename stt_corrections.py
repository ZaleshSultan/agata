"""
Domain-Aware Speech-to-Text Corrections for Agata.

Post-processing layer between raw STT (Whisper) and Action Router / Parsers.
Applies domain-specific corrections to phonetic and transcription inaccuracies
(e.g., "тус" -> "плюс", "умножит" -> "умножь", "отверждаю" -> "подтверждаю",
"5 с четвертой степени" -> "5 в четвертой степени", "телеграмм" -> "телеграм")
without performing dangerous global replacements.
"""

import logging
import re
from typing import Mapping

logger = logging.getLogger(__name__)

# ── 1. Math specific corrections ───────────────────────────────────────────

# Common word replacements in math queries
_MATH_WORD_REPLACEMENTS: dict[str, str] = {
    "тус": "плюс",
    "плюз": "плюс",
    "плюсик": "плюс",
    "умножит": "умножь",
    "умнож": "умножь",
    "умножать": "умножить",
    "умножа": "умножь",
    "умноженное": "умножить на",
    "помножит": "помножь",
    "перемножит": "перемножь",
    "перемнож": "перемножь",
    "делить": "разделить",
    "делен": "делённое",
    "деленное": "делённое",
    "подели": "раздели",
    "поделить": "разделить",
    "вычти": "вычесть",
    "отними": "отнять",
    "квадрат": "в квадрате",
    "куб": "в кубе",
}

# Regex patterns for spoken math phrases where Whisper confuses prepositions or cases
_MATH_PHRASE_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # "5 с четвертой степени" -> "5 в четвертой степени"
    (re.compile(r"(\b\w+\b|\d+)\s+с\s+((?:перв|втор|трет|четв|пят|шест|седьм|восьм|девят|десят)\w*\s+степен\w*)", re.IGNORECASE), r"\1 в \2"),
    # "5 с 4 степени" -> "5 в 4 степени"
    (re.compile(r"(\b\w+\b|\d+)\s+с\s+(\d+\s+степен\w*)", re.IGNORECASE), r"\1 в \2"),
    # "в степени N" / "в N степени"
    (re.compile(r"\bв\s+степень\s+(\d+|\b\w+\b)", re.IGNORECASE), r"в степени \1"),
    # "корень с N" -> "корень из N"
    (re.compile(r"\bкорень\s+с\s+", re.IGNORECASE), "корень из "),
]


def correct_math_text(text: str) -> str:
    """
    Apply domain-aware phonetic and acoustic corrections for mathematical queries.

    Examples:
        "25 тус 17 умножит на 2" -> "25 плюс 17 умножь на 2"
        "5 с четвертой степени" -> "5 в четвертой степени"
        "сто делить на два" -> "сто разделить на два"
    """
    if not text or not text.strip():
        return text

    corrected = text

    # Apply phrase-level corrections
    for pattern, replacement in _MATH_PHRASE_PATTERNS:
        corrected = pattern.sub(replacement, corrected)

    # Word-level replacement with boundary check
    words = corrected.split()
    new_words = []
    for w in words:
        clean_w = w.strip(".,!?:;\"'").lower()
        if clean_w in _MATH_WORD_REPLACEMENTS:
            replacement = _MATH_WORD_REPLACEMENTS[clean_w]
            # Preserve punctuation if any
            prefix = w[:len(w) - len(w.lstrip(".,!?:;\"'"))]
            suffix = w[len(w.rstrip(".,!?:;\"'")):]
            new_words.append(prefix + replacement + suffix)
        else:
            new_words.append(w)

    result = " ".join(new_words)
    if result != text:
        logger.info("[STT CORRECTION: MATH] '%s' -> '%s'", text, result)
    return result


# ── 2. Command specific corrections ────────────────────────────────────────

_COMMAND_APP_REPLACEMENTS: dict[str, str] = {
    "телеграмм": "телеграм",
    "телеграмму": "телеграм",
    "дискордик": "дискорд",
    "хромиум": "хром",
    "споти": "spotify",
    "спотик": "spotify",
    "спотифай": "spotify",
    "вскод": "vscode",
    "vs код": "vscode",
    "вс код": "vscode",
    "гуглхром": "хром",
    "гугл хром": "хром",
    "ютубчик": "youtube",
    "гитхабчик": "github",
}


def correct_command_text(text: str) -> str:
    """
    Apply domain-aware corrections for commands (apps, websites, targets).

    Examples:
        "открой телеграмм" -> "открой телеграм"
        "закрой телеграмму" -> "закрой телеграм"
        "открой спотик" -> "открой spotify"
    """
    if not text or not text.strip():
        return text

    corrected = text.strip()

    # If the text looks like math ("25 тус 17", etc.), also run math corrections
    if any(char.isdigit() for char in corrected) or any(op in corrected.lower() for op in ("плюс", "минус", "умнож", "делит", "раздели", "корень", "степен", "тус")):
        corrected = correct_math_text(corrected)

    # Word-level replacements
    words = corrected.split()
    new_words = []
    for w in words:
        clean_w = w.strip(".,!?:;\"'").lower()
        if clean_w in _COMMAND_APP_REPLACEMENTS:
            replacement = _COMMAND_APP_REPLACEMENTS[clean_w]
            prefix = w[:len(w) - len(w.lstrip(".,!?:;\"'"))]
            suffix = w[len(w.rstrip(".,!?:;\"'")):]
            new_words.append(prefix + replacement + suffix)
        else:
            new_words.append(w)

    result = " ".join(new_words)
    if result != text:
        logger.info("[STT CORRECTION: CMD] '%s' -> '%s'", text, result)
    return result


# ── 3. Confirmation specific corrections ───────────────────────────────────

_CONFIRMATION_REPLACEMENTS: dict[str, str] = {
    # TRUE acoustic confusions
    "отверждаю": "подтверждаю",
    "отверждаем": "подтверждаю",
    "подверждаю": "подтверждаю",
    "подверждаем": "подтверждаю",
    "подтверди": "подтверждаю",
    "отверди": "подтверждаю",
    "подтвержу": "подтверждаю",
    "подтвержи": "подтверждаю",
    "подтвердить": "подтверждаю",
    "агата да": "да",
    "ага да": "да",
    "да агата": "да",
    "ну да": "да",
    "валяи": "валяй",
    "даваи": "давай",
    "канешна": "конечно",
    "конешно": "конечно",
    "угум": "угу",

    # FALSE acoustic confusions
    "агата нет": "нет",
    "нет агата": "нет",
    "отменит": "отменить",
    "отменяи": "отменяй",
    "не надо агата": "не надо",
    "не стоит агата": "не стоит",
}


def correct_confirmation_text(text: str) -> str:
    """
    Apply acoustic corrections for short confirmation utterances.

    Examples:
        "отверждаю" -> "подтверждаю"
        "подверждаю" -> "подтверждаю"
        "конешно" -> "конечно"
    """
    if not text or not text.strip():
        return text

    cleaned = " ".join(text.strip().casefold().split()).strip(" .,!?:;\"'")
    if not cleaned:
        return text

    # Direct full phrase replacement
    if cleaned in _CONFIRMATION_REPLACEMENTS:
        replaced = _CONFIRMATION_REPLACEMENTS[cleaned]
        logger.info("[STT CORRECTION: CONFIRM] '%s' -> '%s'", text, replaced)
        return replaced

    # Word-level replacement
    words = cleaned.split()
    new_words = []
    for w in words:
        if w in _CONFIRMATION_REPLACEMENTS:
            new_words.append(_CONFIRMATION_REPLACEMENTS[w])
        else:
            new_words.append(w)

    result = " ".join(new_words)
    if result != cleaned:
        logger.info("[STT CORRECTION: CONFIRM] '%s' -> '%s'", text, result)
    return result
