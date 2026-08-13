"""
Deterministic routing for unambiguous actions and external-information queries.

The router deliberately produces a registered tool call instead of performing
an action itself. This keeps one execution path between an intent and an
operating-system side effect.

Pipeline:
1. Normalize text.
2. System actions (removable drives, snapshot).
3. Explicit Open / Close commands (with fuzzy matching and case stemming via target_resolver).
4. Natural language and numeric arithmetic calculations (without LLM).
5. Datetime & Day of Week queries (without LLM).
6. Web search intent.
7. Return None for CHAT/UNKNOWN (sent to LLM).
"""

import logging
import re
from dataclasses import dataclass
from typing import Any

from calc_parser import parse_natural_math_to_expression
from target_resolver import resolve_target

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RoutedAction:
    """A tool invocation selected without asking the LLM."""

    kind: str
    tool_name: str
    arguments: dict[str, str]
    source: str = "deterministic"
    confidence: float = 1.0


_OPEN_ACTION_VERB = re.compile(
    r"^\s*(?:открой|открою|открыть|открывай|запусти|запущу|запустить|запускай|включи|включу|включить|включай|покажи|показать|зайди|зайти|перейди|перейти)[\s,:]+",
    re.IGNORECASE,
)
_CLOSE_ACTION_VERB = re.compile(
    r"^\s*(?:закрой|закрою|закрыть|закрывай|выключи|выключу|выключить|выключай|останови|остановить|прибей|убей)[\s,:]+",
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
    "какой день недели", "какой сейчас день недели", "какое число",
    "текущее время", "текущая дата", "подскажи время", "подскажи дату",
    "время сейчас", "дата сегодня",
)

_SYSTEM_DRIVES_PATTERNS = (
    "съемные диски", "съёмные диски", "список дисков", "какие диски",
    "какие флешки", "покажи диски", "покажи флешки", "список флешек",
)

_MATH_EXPRESSION = re.compile(r"^[0-9\s+\-*/().,%*sqrt]+$", re.IGNORECASE)


def _clean_command_for_routing(text: str) -> str:
    """Нормализация текста команды для детерминированного роутинга."""
    if not text:
        return ""
    from stt_corrections import correct_command_text
    cleaned = correct_command_text(text.strip())
    # Удаление обращения "Агата" в начале
    cleaned = re.sub(r"^\s*агата\s*[,!]?\s*", "", cleaned, flags=re.IGNORECASE)
    # Нормализация пунктуации после глаголов действия ("открой, VS Code" -> "открой VS Code")
    cleaned = re.sub(
        r"^\s*(открой|открою|открыть|открывай|запусти|запущу|запустить|запускай|включи|включу|включить|включай|покажи|показать|зайди|зайти|перейди|перейти|закрой|закрою|закрыть|закрывай|выключи|выключу|выключить|выключай|останови|остановить|прибей|убей)[\s,:]+",
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

        logger.info("[ROUTER] command='%s' action=CLOSE_TARGET target='%s'", user_text, target)
        resolved = resolve_target(target)
        if resolved is not None:
            logger.info("[TARGET] resolved=%s (source=%s)", resolved, resolved.source)
            if resolved.kind == "browser":
                return RoutedAction("CLOSE_TARGET", "close_target", {
                    "target": resolved.display_name,
                    "kind": "browser",
                    "url": resolved.target,
                }, source=resolved.source)
            elif resolved.kind == "app":
                return RoutedAction("CLOSE_APP", "close_app", {
                    "app_name": resolved.display_name,
                    "resolved_target": resolved.target,
                }, source=resolved.source)

        alias = _APPLICATION_ALIASES.get(" ".join(target.casefold().split()), target)
        return RoutedAction("CLOSE_APP", "close_app", {"app_name": alias}, source="fallback_alias")

    # ── Обработка OPEN команд ──────────────────────────────────────────────
    match = _OPEN_ACTION_VERB.match(clean_cmd)
    if not match:
        return None

    target = clean_cmd[match.end():].strip().strip(" .,!?:;\"'")
    if not target:
        return None

    # 1. Прямой домен/URL
    url_match = _URL.fullmatch(target)
    if url_match:
        logger.info("[ROUTER] detected action=OPEN_BROWSER target='%s'", target)
        return RoutedAction("OPEN_BROWSER", "open_browser", {"url": url_match.group("url")}, source="url")

    # 2. Использование универсального слоя разрешения целей (target_resolver)
    resolved = resolve_target(target)
    if resolved is not None:
        if resolved.kind == "browser":
            logger.info("[ROUTER] detected action=OPEN_BROWSER target='%s' url='%s' source='%s'", target, resolved.target, resolved.source)
            return RoutedAction("OPEN_BROWSER", "open_browser", {"url": resolved.target}, source=resolved.source)
        elif resolved.kind == "app":
            logger.info(
                "[ROUTER] detected action=OPEN_APP target='%s' resolved_path='%s' source='%s'",
                resolved.display_name, resolved.target, resolved.source,
            )
            return RoutedAction("OPEN_APP", "open_app", {
                "app_name": resolved.display_name,
                "resolved_target": resolved.target,
            }, source=resolved.source)

    # 3. Резерв по статическим алиасам
    alias_app_name = _APPLICATION_ALIASES.get(" ".join(target.casefold().split()))
    if alias_app_name:
        logger.info("[ROUTER] detected action=OPEN_APP target='%s' (alias fallback)", alias_app_name)
        return RoutedAction("OPEN_APP", "open_app", {"app_name": alias_app_name}, source="static_alias")

    return None


def route_datetime(user_text: str) -> RoutedAction | None:
    """Распознать прямой вопрос о текущей дате/времени без участия LLM."""
    clean = _clean_command_for_routing(user_text)
    normalized = " ".join(clean.casefold().split()).strip(" ?.,!:")
    if any(pattern in normalized for pattern in _DATETIME_PATTERNS) or normalized in ("время", "дата", "день недели"):
        logger.info("[ROUTER] detected action=DATETIME query='%s'", user_text)
        return RoutedAction("DATETIME", "get_current_datetime", {}, source="deterministic")
    return None


def route_system_action(user_text: str) -> RoutedAction | None:
    """Распознать простые системные команды (диски, флешки)."""
    clean = _clean_command_for_routing(user_text)
    normalized = " ".join(clean.casefold().split()).strip(" ?.,!:")
    if any(pattern in normalized for pattern in _SYSTEM_DRIVES_PATTERNS) or normalized in ("диски", "флешки"):
        logger.info("[ROUTER] detected action=SYSTEM query='%s'", user_text)
        return RoutedAction("SYSTEM", "list_removable_drives", {}, source="deterministic")
    return None


def route_calculation(user_text: str) -> RoutedAction | None:
    """Recognize natural language math phrases or direct arithmetic expressions without LLM."""
    clean = _clean_command_for_routing(user_text)

    # 1. Попытка распарсить словесное русское выражение (e.g. "двенадцать плюс семнадцать")
    parsed_expr = parse_natural_math_to_expression(clean)
    if parsed_expr:
        logger.info("[ROUTER] detected action=CALCULATE parsed_expr='%s' from='%s'", parsed_expr, user_text)
        return RoutedAction("CALCULATE", "calculate", {"expression": parsed_expr}, source="natural_language_calc")

    # 2. Прямое строковое арифметическое выражение (e.g. "12 + 17", "25 * 4")
    expression = clean.strip().rstrip("?").strip().replace("×", "*").replace("÷", "/")
    if expression and _MATH_EXPRESSION.fullmatch(expression) and any(c.isdigit() for c in expression) and any(op in expression for op in ("+", "-", "*", "/", "%", "**", "sqrt")):
        logger.info("[ROUTER] detected action=CALCULATE numeric_expr='%s'", expression)
        return RoutedAction("CALCULATE", "calculate", {"expression": expression}, source="numeric_calc")

    return None


def route_web_intent(user_text: str) -> RoutedAction | None:
    """Recognize queries that need fresh or externally verifiable information."""
    normalized = " ".join(user_text.casefold().split())
    if any(ex in normalized for ex in _CONVERSATIONAL_EXCLUSIONS):
        return None
    if any(pattern in normalized for pattern in _EXPLICIT_WEB_PATTERNS) or any(pattern in normalized for pattern in _EXTERNAL_FACT_PATTERNS):
        logger.info("[ROUTER] command='%s' action=WEB_SEARCH query='%s'", user_text, user_text.strip())
        return RoutedAction("WEB_SEARCH", "web_search", {"query": user_text.strip()}, source="web_pattern")
    return None


def classify_command(user_text: str) -> tuple[RoutedAction | None, str, bool]:
    """
    Unified Fast Route Classifier.

    Returns:
        (routed_action, classification_kind, llm_skipped)
    """
    clean_cmd = _clean_command_for_routing(user_text)
    if not clean_cmd:
        return None, "EMPTY", False

    # 1. Explicit App/Browser Open/Close
    action = route_explicit_action(clean_cmd)
    if action is not None:
        logger.info("[FAST ROUTER] classification=%s source=%s llm_skipped=true", action.kind, action.source)
        return action, action.kind, True

    # 2. Calculation
    calc_action = route_calculation(clean_cmd)
    if calc_action is not None:
        logger.info("[FAST ROUTER] classification=CALCULATE source=%s llm_skipped=true", calc_action.source)
        return calc_action, "CALCULATE", True

    # 3. Datetime
    dt_action = route_datetime(clean_cmd)
    if dt_action is not None:
        logger.info("[FAST ROUTER] classification=DATETIME source=deterministic llm_skipped=true",)
        return dt_action, "DATETIME", True

    # 4. System Action
    sys_action = route_system_action(clean_cmd)
    if sys_action is not None:
        logger.info("[FAST ROUTER] classification=SYSTEM source=deterministic llm_skipped=true")
        return sys_action, "SYSTEM", True

    # 5. Web Search Intent
    web_action = route_web_intent(clean_cmd)
    if web_action is not None:
        logger.info("[FAST ROUTER] classification=WEB_SEARCH source=web_intent llm_skipped=false")
        return web_action, "WEB_SEARCH", False

    # 6. Conversational / Unknown -> Needs LLM
    logger.info("[FAST ROUTER] classification=UNKNOWN llm_required=true")
    return None, "UNKNOWN", False
