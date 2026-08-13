"""
Ultra-fast Voice & Text Confirmation Parser.

Parses user confirmations (Yes / No) without invoking any LLM,
supporting exact words, phrase matching, and fuzzy tolerance for STT inaccuracies.
"""

import enum
import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)


class ConfirmationResult(str, enum.Enum):
    TRUE = "TRUE"
    FALSE = "FALSE"
    UNKNOWN = "UNKNOWN"


_CONFIRM_TRUE_EXACT = frozenset({
    "да", "ага", "угу", "конечно", "подтверждаю", "выполняй", "давай",
    "точно", "верно", "ок", "окей", "okay", "yes", "y", "д", "валяй",
    "согласен", "согласна", "делай", "разрешаю", "давай делай", "конечно давай",
    "плюс", "подтвердить", "подтверди", "подтвердите",
})

_CONFIRM_FALSE_EXACT = frozenset({
    "нет", "не надо", "отмена", "отмени", "стоп", "не выполняй",
    "не нужно", "не стоит", "отказ", "no", "n", "н", "нельзя", "отменяй",
    "не делай", "отбой", "хватит", "минус", "отменить", "отмените",
})

_CONFIRM_TRUE_PHRASES = (
    "да конечно", "ну давай", "да давай", "конечно да", "я подтверждаю",
    "давай делай", "делай конечно", "подтверждаю действие",
)

_CONFIRM_FALSE_PHRASES = (
    "нет не надо", "нет отмена", "нет не нужно", "не надо делать",
    "я отказываюсь", "отмени действие", "лучше не надо",
)


def parse_confirmation(user_text: str, strict: bool = False) -> ConfirmationResult:
    """
    Parse a user confirmation into TRUE, FALSE, or UNKNOWN.

    Args:
        user_text: Raw or normalized user input string.
        strict: If True, uses stricter fuzzy thresholds (e.g. for destructive actions).

    Returns:
        ConfirmationResult: TRUE, FALSE, or UNKNOWN.
    """
    if not user_text or not user_text.strip():
        return ConfirmationResult.UNKNOWN

    # 0. Apply domain-aware STT acoustic corrections first
    from stt_corrections import correct_confirmation_text
    corrected = correct_confirmation_text(user_text)

    cleaned = " ".join(corrected.strip().casefold().split()).strip(" .,!?:;\"'")
    if not cleaned:
        return ConfirmationResult.UNKNOWN

    # 1. Exact match against known words
    if cleaned in _CONFIRM_TRUE_EXACT:
        logger.info("[CONFIRM] exact match TRUE: '%s'", cleaned)
        return ConfirmationResult.TRUE

    if cleaned in _CONFIRM_FALSE_EXACT:
        logger.info("[CONFIRM] exact match FALSE: '%s'", cleaned)
        return ConfirmationResult.FALSE

    # 2. Phrase matching
    if any(cleaned == p or cleaned.startswith(p + " ") or cleaned.endswith(" " + p) for p in _CONFIRM_TRUE_PHRASES):
        logger.info("[CONFIRM] phrase match TRUE: '%s'", cleaned)
        return ConfirmationResult.TRUE

    if any(cleaned == p or cleaned.startswith(p + " ") or cleaned.endswith(" " + p) for p in _CONFIRM_FALSE_PHRASES):
        logger.info("[CONFIRM] phrase match FALSE: '%s'", cleaned)
        return ConfirmationResult.FALSE

    # 3. Word token matching
    words = cleaned.split()
    word_set = set(words)

    # Check negative first to prioritize safety
    false_intersect = word_set & _CONFIRM_FALSE_EXACT
    if false_intersect and not (word_set & _CONFIRM_TRUE_EXACT):
        logger.info("[CONFIRM] word token match FALSE: %s in '%s'", false_intersect, cleaned)
        return ConfirmationResult.FALSE

    true_intersect = word_set & _CONFIRM_TRUE_EXACT
    if true_intersect and not false_intersect:
        logger.info("[CONFIRM] word token match TRUE: %s in '%s'", true_intersect, cleaned)
        return ConfirmationResult.TRUE

    # 4. Fuzzy fallback on single-word or short utterances
    if len(words) <= 2:
        from fuzzy_resolver import fuzzy_resolve
        threshold = 0.90 if strict else 0.82

        # Check against positive words
        pos_match = fuzzy_resolve(cleaned, list(_CONFIRM_TRUE_EXACT), threshold=threshold)
        if pos_match:
            logger.info("[CONFIRM] fuzzy match TRUE: '%s' -> '%s' (score=%.2f)", cleaned, pos_match.matched_candidate, pos_match.score)
            return ConfirmationResult.TRUE

        neg_match = fuzzy_resolve(cleaned, list(_CONFIRM_FALSE_EXACT), threshold=threshold)
        if neg_match:
            logger.info("[CONFIRM] fuzzy match FALSE: '%s' -> '%s' (score=%.2f)", cleaned, neg_match.matched_candidate, neg_match.score)
            return ConfirmationResult.FALSE

    logger.info("[CONFIRM] unclassified confirmation: '%s'", cleaned)
    return ConfirmationResult.UNKNOWN
