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
