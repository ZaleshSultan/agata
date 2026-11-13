"""Deterministic routing for unambiguous actions and external-information queries.

The router deliberately produces a registered tool call instead of performing
an action itself. This keeps one execution path between an intent and an
operating-system side effect.
"""

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RoutedAction:
    """A tool invocation selected without asking the LLM."""

    kind: str
    tool_name: str
    arguments: dict[str, str]


_OPEN_ACTION_VERB = re.compile(
    r"^\s*(?:открой|открою|открыть|открывай|запусти|запущу|запустить|запускай|включи|включу|включить|включай|покажи|показать|зайди|зайти|перейди|перейти)[\s,:]+",
    re.IGNORECASE,
)
_CLOSE_ACTION_VERB = re.compile(
    r"^\s*(?:закрой|закрою|закрыть|закрывай|выключи|выключу|выключить|выключай)[\s,:]+",
    re.IGNORECASE,
)
_URL = re.compile(
    r"(?P<url>(?:https?://)?(?:www\.)?[a-z0-9][a-z0-9.-]*\.[a-z]{2,}(?:/[^\s]*)?)",
    re.IGNORECASE,
)
_APPLICATION_ALIASES = {
    # Chrome
    "chrome": "chrome",
    "google chrome": "chrome",
    "гугл хром": "chrome",
    "гугл-хром": "chrome",
    "хром": "chrome",
    # Firefox
    "firefox": "firefox",
    "mozilla firefox": "firefox",
    "файрфокс": "firefox",
    "фаерфокс": "firefox",
    "файерфокс": "firefox",
    "мозилла": "firefox",
    # VS Code
    "vscode": "vscode",
    "vs code": "vscode",
    "visual studio code": "vscode",
    "вс код": "vscode",
    "vs код": "vscode",
    # Discord
    "discord": "discord",
    "дискорд": "discord",
    # Telegram
    "telegram": "telegram",
    "телеграм": "telegram",
    "телега": "telegram",
    "телегу": "telegram",
    # Spotify
    "spotify": "spotify",
    "спотифай": "spotify",
    # Steam
    "steam": "steam",
    "стим": "steam",
    # Notepad
    "notepad": "notepad",
    "блокнот": "notepad",
    # Explorer
    "explorer": "explorer",
    "проводник": "explorer",
}

_EXPLICIT_WEB_PATTERNS = (
    "найди в интернете", "найди информацию", "найди всю информацию",
    "поищи в интернете", "поищи информацию", "поищи", "поиск",
    "загугли", "в интернете", "найдите", "найди",
)
_EXTERNAL_FACT_PATTERNS = (
    "дата рождения", "когда родил", "когда родилась", "когда родился",
    "кто такой", "кто такая", "кто это", "что известно о",
    "последние новости", "свежие новости", "что сейчас происходит",
    "какая сейчас цена", "текущая цена", "курс валют", "актуальн",
    "что произошло сегодня", "что произошло", "новости про",
    "что сейчас известно",
    # Weather / погода
    "погод", "прогноз погоды", "прогноз на", "температур",
    "weather", "forecast",
)
_CONVERSATIONAL_EXCLUSIONS = (
    "как дела", "поговорим", "что думаешь", "расскажи мне",
    "расскажи сказку", "расскажи анекдот", "как жизнь",
    "чем занимаешься", "привет", "добрый день", "добрый вечер",
)

_DATETIME_PATTERNS = (
    "который час", "сколько времени", "сколько сейчас времени",
    "какое сегодня число", "какая сегодня дата", "какой сегодня день",
    "какой день недели", "какое число", "текущее время", "текущая дата",
    "подскажи время", "подскажи дату",
)

_MATH_EXPRESSION = re.compile(r"^[0-9\s+\-*/().,%]+$")

_NUMBER_WORDS = {
    "ноль": 0,
    "один": 1,
    "одна": 1,
    "одно": 1,
    "два": 2,
    "две": 2,
    "три": 3,
    "четыре": 4,
    "пять": 5,
    "шесть": 6,
    "семь": 7,
    "восемь": 8,
    "девять": 9,
    "десять": 10,
    "одиннадцать": 11,
    "двенадцать": 12,
    "тринадцать": 13,
    "четырнадцать": 14,
    "пятнадцать": 15,
    "шестнадцать": 16,
    "семнадцать": 17,
    "восемнадцать": 18,
    "девятнадцать": 19,
    "двадцать": 20,
    "тридцать": 30,
    "сорок": 40,
    "пятьдесят": 50,
    "шестьдесят": 60,
    "семьдесят": 70,
    "восемьдесят": 80,
    "девяносто": 90,
    "сто": 100,
}
_NUMBER_WORD_RE = re.compile(
    r"\b(?:" + "|".join(sorted(map(re.escape, _NUMBER_WORDS), key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)
_VOLUME_DOWN_RE = re.compile(
    r"\b(?:уменьши|уменьшу|уменьшить|уменьшай|убавь|убавлю|убавить|убавляй|снизь|снижу|снизить|понизь|понижу|понизить|тише|потише|меньше|он\s+меньше)\b",
    re.IGNORECASE,
)
_VOLUME_UP_RE = re.compile(
    r"\b(?:увеличь|увеличу|увеличить|увеличивай|прибавь|прибавлю|прибавить|подними|подниму|поднять|повысь|повышу|повысить|громче|погромче|больше)\b",
    re.IGNORECASE,
)
_VOLUME_WORD_RE = re.compile(r"\b(?:громкость|звук|volume)\b", re.IGNORECASE)
_VOLUME_SET_RE = re.compile(r"^\s*(?:поставь|установи|выставь)\s+(?:на\s+)?\d{1,3}\s*(?:%|процент(?:а|ов)?)?\s*$", re.IGNORECASE)
_PERCENT_RE = re.compile(r"\b(?P<number>\d{1,3})\s*(?:%|процент(?:а|ов)?)?\b", re.IGNORECASE)
_MEDIA_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    # Bare pause/resume — highest priority, unambiguous
    (re.compile(r"^\s*(?:пауза|на паузу|поставь на паузу)\s*[.!]?\s*$", re.IGNORECASE), "play_pause"),
    (re.compile(r"^\s*(?:продолжи|возобнови|продолжить|возобновить)\s*[.!]?\s*$", re.IGNORECASE), "play_pause"),
    # Inline pause/resume
    (re.compile(r"\b(?:пауз\w*|приостанови|продолжи|возобнови)\b", re.IGNORECASE), "play_pause"),
    # Next/previous — allow bare "следующая" / "предыдущая" (with optional трек/песня)
    (re.compile(r"^\s*(?:следующ(?:ий|ая|ее))\s*(?:трек|песн\w*|композиц\w*)?\s*[.!]?\s*$", re.IGNORECASE), "next"),
    (re.compile(r"^\s*(?:предыдущ(?:ий|ая|ее))\s*(?:трек|песн\w*|композиц\w*)?\s*[.!]?\s*$", re.IGNORECASE), "previous"),
    (re.compile(r"\b(?:следующ(?:ий|ую)|переключи)\s+(?:трек|песн|композиц)", re.IGNORECASE), "next"),
    (re.compile(r"\b(?:предыдущ(?:ий|ую)|назад)\s+(?:трек|песн|композиц)", re.IGNORECASE), "previous"),
    (re.compile(r"\b(?:переключи)\s+(?:трек|песн|композиц)", re.IGNORECASE), "next"),
    # Music transport
    (re.compile(r"\b(?:останови|выключи|включи|запусти)\s+(?:музык|песн|трек|воспроизведени)", re.IGNORECASE), "play_pause"),
    # English variants
    (re.compile(r"\b(?:pause|resume)\b|\b(?:play|stop|start)\s+(?:the\s+)?(?:music|song|track|playback)\b", re.IGNORECASE), "play_pause"),
    (re.compile(r"\b(?:next|skip)\s+(?:song|track)\b", re.IGNORECASE), "next"),
    (re.compile(r"\b(?:previous|prev)\s+(?:song|track)\b", re.IGNORECASE), "previous"),
)


def _clean_command_for_routing(text: str) -> str:
    """Нормализация текста команды для детерминированного роутинга."""
    if not text:
        return ""
    cleaned = text.strip()
    # Удаление обращения "Агата" в начале
    cleaned = re.sub(r"^\s*агата\s*[,!]?\s*", "", cleaned, flags=re.IGNORECASE)
    # Нормализация пунктуации после глаголов действия ("открой, VS Code" -> "открой VS Code")
    cleaned = re.sub(
        r"^\s*(открой|открою|открыть|открывай|запусти|запущу|запустить|запускай|включи|включу|включить|включай|покажи|показать|зайди|зайти|перейди|перейти|закрой|закрою|закрыть|закрывай|выключи|выключу|выключить|выключай)[\s,:]+",
        r"\1 ",
        cleaned,
        flags=re.IGNORECASE,
    )
    return cleaned.strip()


def _replace_words_with_numbers(text: str) -> str:
    """Convert simple Russian number words to digits before regex routing."""
    def replace(match: re.Match[str]) -> str:
        return str(_NUMBER_WORDS[match.group(0).casefold().replace("ё", "е")])

    return _NUMBER_WORD_RE.sub(replace, text.replace("ё", "е"))


def _first_percent(text: str) -> str | None:
    match = _PERCENT_RE.search(text)
    if not match:
        return None
    number = int(match.group("number"))
    if 0 <= number <= 100:
        return str(number)
    return None


def route_explicit_action(user_text: str) -> RoutedAction | None:
    """Return a tool call only for a clearly expressed, supported action (OPEN or CLOSE)."""
    clean_cmd = _clean_command_for_routing(user_text)
    if not clean_cmd:
        return None

    # ── Обработка CLOSE команд ─────────────────────────────────────────────
    close_match = _CLOSE_ACTION_VERB.match(clean_cmd)
    if close_match:
        target = clean_cmd[close_match.end():].strip().strip(" .,!?:;\"'")
        if not target:
            return None

        logger.info("[ROUTER] command='%s'", user_text)
        logger.info("[ROUTER] action=CLOSE_TARGET")
        logger.info("[TARGET] resolving='%s'", target)

        from target_resolver import resolve_target
        resolved = resolve_target(target)
        if resolved is not None:
            logger.info("[TARGET] resolved=%s", resolved)
            if resolved.kind == "browser":
                return RoutedAction("CLOSE_TARGET", "close_target", {
                    "target": resolved.display_name,
                    "kind": "browser",
                    "url": resolved.target,
                })
            elif resolved.kind == "app":
                return RoutedAction("CLOSE_APP", "close_app", {
                    "app_name": resolved.display_name,
                    "resolved_target": resolved.target,
                })

        alias = _APPLICATION_ALIASES.get(" ".join(target.casefold().split()), target)
        return RoutedAction("CLOSE_APP", "close_app", {"app_name": alias})

    # ── Обработка OPEN команд ──────────────────────────────────────────────
    match = _OPEN_ACTION_VERB.match(clean_cmd)
    if not match:
        return None

    target = clean_cmd[match.end():].strip().strip(" .,!?:;\"'")
    if not target:
        return None

    logger.info("[ROUTER] normalized_command='%s'", clean_cmd)

    # 1. Прямой домен/URL
    url_match = _URL.fullmatch(target)
    if url_match:
        logger.info("[ROUTER] detected action=OPEN_BROWSER target='%s'", target)
        return RoutedAction("OPEN_BROWSER", "open_browser", {"url": url_match.group("url")})

    # 2. Использование универсального слоя разрешения целей (target_resolver)
    from target_resolver import resolve_target
    resolved = resolve_target(target)
    if resolved is not None:
        if resolved.kind == "browser":
            logger.info("[ROUTER] detected action=OPEN_BROWSER target='%s' url='%s'", target, resolved.target)
            return RoutedAction("OPEN_BROWSER", "open_browser", {"url": resolved.target})
        elif resolved.kind == "app":
            logger.info(
                "[ROUTER] detected action=OPEN_APP target='%s' resolved_path='%s' source='%s'",
                resolved.display_name, resolved.target, resolved.source,
            )
            return RoutedAction("OPEN_APP", "open_app", {
                "app_name": resolved.display_name,
                "resolved_target": resolved.target,
            })

    # 3. Резерв по статическим алиасам
    alias_app_name = _APPLICATION_ALIASES.get(" ".join(target.casefold().split()))
    if alias_app_name:
        logger.info("[ROUTER] detected action=OPEN_APP target='%s' (alias fallback, no resolved path)", alias_app_name)
        return RoutedAction("OPEN_APP", "open_app", {"app_name": alias_app_name})

    return None


def route_volume(user_text: str) -> RoutedAction | None:
    """Recognize direct volume changes without sending them to the LLM."""
    normalized = " ".join(_replace_words_with_numbers(user_text).casefold().split())
    if not normalized or not (
        _VOLUME_WORD_RE.search(normalized)
        or _VOLUME_DOWN_RE.search(normalized)
        or _VOLUME_UP_RE.search(normalized)
        or _VOLUME_SET_RE.fullmatch(normalized)
    ):
        return None

    amount = _first_percent(normalized)
    is_down = bool(_VOLUME_DOWN_RE.search(normalized))
    is_up = bool(_VOLUME_UP_RE.search(normalized))

    if is_down and amount is not None:
        return RoutedAction("VOLUME", "control_volume", {"action": "volume_down", "step": amount})
    if is_up and amount is not None:
        return RoutedAction("VOLUME", "control_volume", {"action": "volume_up", "step": amount})
    if is_down:
        return RoutedAction("VOLUME", "control_volume", {"action": "volume_down"})
    if is_up:
        return RoutedAction("VOLUME", "control_volume", {"action": "volume_up"})

    # Absolute setting: "громкость 30", "поставь громкость 30",
    # "установи громкость на 30". Relative commands with "меньше/тише/громче"
    # were handled above, so "громкость на 30 меньше" cannot become volume_set.
    if amount is not None and (_VOLUME_WORD_RE.search(normalized) or _VOLUME_SET_RE.fullmatch(normalized)):
        return RoutedAction("VOLUME", "control_volume", {"action": "volume_set", "level": amount})

    return None


def route_media(user_text: str) -> RoutedAction | None:
    """Route unambiguous media transport commands before the LLM.

    Uses dedicated deterministic transport parser to guarantee stop/pause/resume/next/previous
    never trigger content search or invoke an LLM.
    """
    from media.transport_parser import parse_media_command
    cmd = parse_media_command(user_text)
    if cmd is not None:
        logger.info("[ROUTER] transport action=%s query='%s'", cmd.action, user_text)
        return RoutedAction("MEDIA", "control_media", {"action": cmd.action})

    normalized = " ".join(user_text.casefold().split())
    if not normalized:
        return None
    for pattern, action in _MEDIA_PATTERNS:
        if pattern.search(normalized):
            return RoutedAction("MEDIA", "control_media", {"action": action})
    return None


def route_datetime(user_text: str) -> RoutedAction | None:
    """Распознать прямой вопрос о текущей дате/времени без участия LLM."""
    normalized = " ".join(user_text.casefold().split())
    if any(pattern in normalized for pattern in _DATETIME_PATTERNS):
        return RoutedAction("DATETIME", "get_current_datetime", {})
    return None


def route_calculation(user_text: str) -> RoutedAction | None:
    """Recognize a standalone arithmetic expression without involving an LLM."""
    expression = user_text.strip().rstrip("?").strip().replace("×", "*").replace("÷", "/")
    if not expression or not _MATH_EXPRESSION.fullmatch(expression):
        return None
    return RoutedAction("CALCULATE", "calculate", {"expression": expression})


def route_web_intent(user_text: str) -> RoutedAction | None:
    """Recognize queries that need fresh or externally verifiable information."""
    normalized = " ".join(user_text.casefold().split())
    if any(ex in normalized for ex in _CONVERSATIONAL_EXCLUSIONS):
        return None
    if any(pattern in normalized for pattern in _EXPLICIT_WEB_PATTERNS) or any(pattern in normalized for pattern in _EXTERNAL_FACT_PATTERNS):
        logger.info("[ROUTER] command='%s'", user_text)
        logger.info("[ROUTER] action=WEB_SEARCH")
        logger.info("[WEB] query='%s'", user_text.strip())
        return RoutedAction("WEB_SEARCH", "web_search", {"query": user_text.strip()})
    return None


# ── Content play routing ─────────────────────────────────────────────────

# Platform aliases extracted from user text ("в саундклауде", "на ютубе")
_PLATFORM_ALIASES: dict[str, str] = {
    "ютуб": "youtube",
    "ютубе": "youtube",
    "youtube": "youtube",
    "саундклауд": "soundcloud",
    "саундклауде": "soundcloud",
    "soundcloud": "soundcloud",
    "спотифай": "spotify",
    "spotify": "spotify",
}

# Regex to detect platform suffix: "в саундклауде", "на ютубе", "в youtube"
_PLATFORM_SUFFIX_RE = re.compile(
    r"\s+(?:в|на|through|on|in)\s+(?P<platform>"
    + "|".join(re.escape(k) for k in sorted(_PLATFORM_ALIASES, key=len, reverse=True))
    + r")\s*[.!]?\s*$",
    re.IGNORECASE,
)

# Content play verbs — commands that request playing specific content
_CONTENT_PLAY_VERB = re.compile(
    r"^\s*(?:поставь|послушай|послушать|включи|запусти|воспроизведи|воспроизвести|сыграй|проиграй)\s+",
    re.IGNORECASE,
)

# Exclusions: phrases that look like content play but aren't
_CONTENT_PLAY_EXCLUSIONS = re.compile(
    r"\b(?:"
    r"на\s+паузу"
    r"|на\s+стоп"
    r"|пауз\w*"
    r"|стоп"
    r"|останови\w*"
    r"|приостанови\w*"
    r"|дневник"
    r"|запись\s+в"
    r"|запись$"
    r"|таймер"
    r"|будильник"
    r"|напоминани"
    r"|громкость"
    r"|звук"
    r")\b",
    re.IGNORECASE,
)


# Bare media objects — if the ENTIRE object after the verb is just one of these,
# it's media transport, not content play. "включи музыку" → media, but
# "включи рок музыку" → content play.
_BARE_MEDIA_OBJECTS = re.compile(
    r"^\s*(?:музык\w*|песн\w*|трек\w*|воспроизведени\w*|плейлист\w*)\s*[.!]?\s*$",
    re.IGNORECASE,
)


def route_content_play(user_text: str) -> RoutedAction | None:
    """Detect content search/play intent deterministically.

    Distinguishes:
    - "поставь на паузу" → NOT content (media transport)
    - "поставь запись в дневник" → NOT content (diary)
    - "поставь музыку" → NOT content (bare media → media transport)
    - "поставь лекцию про нейробиологию" → CONTENT (YouTube default)
    - "поставь Монополию в саундклауде" → CONTENT (SoundCloud explicit)
    """
    clean = _clean_command_for_routing(user_text)
    if not clean:
        return None

    # Must start with a content play verb
    verb_match = _CONTENT_PLAY_VERB.match(clean)
    if not verb_match:
        return None

    remainder = clean[verb_match.end():].strip().rstrip(" .,!?:;\"'")
    if not remainder:
        return None

    # Exclude diary, timer, volume, pause commands
    if _CONTENT_PLAY_EXCLUSIONS.search(remainder):
        return None

    # Extract platform if present
    platform: str = "youtube"  # default
    query = remainder

    platform_match = _PLATFORM_SUFFIX_RE.search(remainder)
    if platform_match:
        platform_key = platform_match.group("platform").lower()
        resolved_platform = _PLATFORM_ALIASES.get(platform_key)
        if resolved_platform:
            platform = resolved_platform
            # Remove platform suffix from query
            query = remainder[:platform_match.start()].strip().rstrip(" .,!?:;\"'")

    # If the remaining query is a bare media object → media transport, not content
    if _BARE_MEDIA_OBJECTS.fullmatch(query):
        return None

    if not query:
        return None

    logger.info("[ROUTER] command='%s'", user_text)
    logger.info("[ROUTER] action=PLAY_CONTENT query='%s' platform='%s'", query, platform)
    return RoutedAction("PLAY_CONTENT", "play_content", {"query": query, "platform": platform})
