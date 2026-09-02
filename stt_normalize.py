"""
Нормализация текста между STT и router.

Whisper возвращает raw-текст с ошибками распознавания, разговорными
формами глаголов и вводными фразами, которые затрудняют детерминированный
роутинг. Этот модуль приводит текст к каноническому виду, понятному
action_router'у, не трогая смысл.

Вызов:
    from stt_normalize import normalize_stt
    canonical = normalize_stt(raw_stt_text)

Всё изменение — в normalize_stt(). Остальные функции — вспомогательные,
используются только внутри модуля.
"""

import logging
import re

logger = logging.getLogger(__name__)

# ── Коррекция имени ассистента ─────────────────────────────────────────────
# Whisper иногда путает "Агата" в схожие слова.
# Только точные совпадения отдельного слова — не агрессивная замена.
_ASSISTANT_NAME_CORRECTIONS: dict[str, str] = {
    "огата": "агата",
    "огадок": "агата",
    "агат": "агата",
    "ахата": "агата",
}

# ── Обращение к ассистенту в начале фразы ─────────────────────────────────
# "Агата, открой хром" → "открой хром"
# "Агата открой хром" → "открой хром"
_ASSISTANT_ADDRESS = re.compile(
    r"^\s*агата\s*[,!]?\s*",
    re.IGNORECASE,
)

# ── Вводные фразы перед командой ──────────────────────────────────────────
# Инфинитив без вспомогательного глагола: "открыть X" → "открой X"
# "запустить X" → "запусти X"
# "можешь открыть X" / "можешь запустить X" → соответствующий императив
_FILLER_OTKRYT = re.compile(
    r"^\s*(?:можешь\s+|пожалуйста\s+|пожалуйста,?\s+)?открыть\s+",
    re.IGNORECASE,
)
_FILLER_ZAPUSTIT = re.compile(
    r"^\s*(?:можешь\s+|пожалуйста\s+|пожалуйста,?\s+)?запустить\s+",
    re.IGNORECASE,
)
_FILLER_VKLYUCHIT = re.compile(
    r"^\s*(?:можешь\s+|пожалуйста\s+|пожалуйста,?\s+)?включить\s+",
    re.IGNORECASE,
)
_FILLER_CLOSE = re.compile(
    r"^\s*(?:можешь\s+|пожалуйста\s+|пожалуйста,?\s+)?(?:закрыть|выключить)\s+",
    re.IGNORECASE,
)

# ── Нормализация глагольных форм ───────────────────────────────────────────
# Приводим 1-е лицо / другие формы → повелительное наклонение
# для совместимости с паттернами action_router.
# Структура: (паттерн, замена)
_VERB_NORMALIZATIONS: list[tuple[re.Pattern[str], str]] = [
    # "открою X" → "открой X"
    (re.compile(r"^открою\s+", re.IGNORECASE), "открой "),
    # "открываю X" → "открой X"
    (re.compile(r"^открываю\s+", re.IGNORECASE), "открой "),
    # "запущу X" → "запусти X"
    (re.compile(r"^запущу\s+", re.IGNORECASE), "запусти "),
    # "запускаю X" → "запусти X"
    (re.compile(r"^запускаю\s+", re.IGNORECASE), "запусти "),
    # "включаю X" → "включи X"
    (re.compile(r"^включаю\s+", re.IGNORECASE), "включи "),
    # "закрываю X" → "закрой X"
    (re.compile(r"^закрываю\s+", re.IGNORECASE), "закрой "),
    # "закрою X" → "закрой X"
    (re.compile(r"^закрою\s+", re.IGNORECASE), "закрой "),
    # "выключаю X" → "выключи X"
    (re.compile(r"^выключаю\s+", re.IGNORECASE), "выключи "),
]

# ── Очистка trailing punctuation у команд ────────────────────────────────
# Whisper нередко ставит точку в конце: "открой хром." → "открой хром"
_TRAILING_PUNCT = re.compile(r"[.!,;]+\s*$")

# ── Нормализация имён сервисов / приложений из STT ───────────────────────
# Whisper даёт вариации произношений; приводим к каноническому виду,
# который совпадает с alias'ами в target_resolver и action_router.
_SERVICE_NAME_NORMALIZATIONS: dict[str, str] = {
    "саундклауде": "саундклауд",
    "саунд клауде": "саундклауд",
    "саунд клауд": "саундклауд",
    "саунд-клауд": "саундклауд",
    "файерфокс": "файрфокс",
    "фаерфокс": "файрфокс",
    "фаер фокс": "файрфокс",
    "файер фокс": "файрфокс",
    "ю туб": "ютуб",
    "ю-туб": "ютуб",
}

# Build a compiled regex for service name replacement (longest match first)
_SERVICE_NAME_RE = re.compile(
    r"\b(?:" + "|".join(
        re.escape(k) for k in sorted(_SERVICE_NAME_NORMALIZATIONS, key=len, reverse=True)
    ) + r")\b",
    re.IGNORECASE,
)


def _normalize_service_names(text: str) -> str:
    """Привести STT-вариации имён сервисов к каноническому виду."""
    def _replace(match: re.Match[str]) -> str:
        return _SERVICE_NAME_NORMALIZATIONS.get(match.group(0).lower(), match.group(0))
    return _SERVICE_NAME_RE.sub(_replace, text)


def _correct_assistant_name(text: str) -> str:
    """Исправить ошибочные варианты имени ассистента как отдельного слова."""
    words = text.split()
    corrected = []
    for word in words:
        # Убираем пунктуацию вокруг слова для сравнения
        stripped = word.strip(".,!?:;\"'").lower()
        if stripped in _ASSISTANT_NAME_CORRECTIONS:
            # Сохраняем пунктуацию, только заменяем само слово
            suffix = word[len(word.rstrip(".,!?:;\"'")):]
            corrected.append(_ASSISTANT_NAME_CORRECTIONS[stripped] + suffix)
        else:
            corrected.append(word)
    return " ".join(corrected)


def _strip_assistant_address(text: str) -> str:
    """Убрать обращение к ассистенту в начале фразы."""
    return _ASSISTANT_ADDRESS.sub("", text).strip()


def _normalize_verb_form(text: str) -> str:
    """Привести разговорные формы глагола к повелительному наклонению."""
    for pattern, replacement in _VERB_NORMALIZATIONS:
        new_text = pattern.sub(replacement, text)
        if new_text != text:
            return new_text
    return text


def _strip_filler_phrases(text: str) -> str:
    """Убрать вводные фразы вида 'можешь открыть'/'пожалуйста закрыть'."""
    if _FILLER_OTKRYT.match(text):
        target = _FILLER_OTKRYT.sub("", text).strip()
        return f"открой {target}"
    if _FILLER_ZAPUSTIT.match(text):
        target = _FILLER_ZAPUSTIT.sub("", text).strip()
        return f"запусти {target}"
    if _FILLER_VKLYUCHIT.match(text):
        target = _FILLER_VKLYUCHIT.sub("", text).strip()
        return f"включи {target}"
    if _FILLER_CLOSE.match(text):
        target = _FILLER_CLOSE.sub("", text).strip()
        return f"закрой {target}"
    return text


def normalize_stt(raw: str) -> str:
    """
    Нормализовать сырой текст STT к каноническому виду для роутера.

    Pipeline:
        raw text
          → исправить имя ассистента (Огадок → Агата)
          → убрать обращение (Агата, → stripped)
          → нормализовать имена сервисов (саундклауде → саундклауд)
          → убрать вводные фразы (можешь открыть → открой)
          → нормализовать форму глагола (открою → открой)
          → убрать trailing punct (открой хром. → открой хром)

    Args:
        raw: Текст, полученный от Whisper.

    Returns:
        Нормализованный текст, готовый для action_router.
    """
    if not raw or not raw.strip():
        return raw

    text = raw.strip()

    text = _correct_assistant_name(text)
    text = _strip_assistant_address(text)
    text = _normalize_service_names(text)
    text = _strip_filler_phrases(text)
    text = _normalize_verb_form(text)
    text = _TRAILING_PUNCT.sub("", text).strip()

    return text
