"""Deterministic routing for unambiguous actions and external-information queries.

The router deliberately produces a registered tool call instead of performing
an action itself.  This keeps one execution path (and leaves room for a future
security gateway) between an intent and an operating-system side effect.
"""

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class RoutedAction:
    """A tool invocation selected without asking the LLM."""

    kind: str
    tool_name: str
    arguments: dict[str, str]


_OPEN_ACTION_VERB = re.compile(
    r"^\s*(?:открой|открыть|запусти|запустить|включи|включить)\s+",
    re.IGNORECASE,
)
_CLOSE_ACTION_VERB = re.compile(
    r"^\s*(?:закрой|закрыть|выключи|выключить)\s+",
    re.IGNORECASE,
)
_URL = re.compile(
    r"(?P<url>(?:https?://)?(?:www\.)?[a-z0-9][a-z0-9.-]*\.[a-z]{2,}(?:/[^\s]*)?)",
    re.IGNORECASE,
)
_APPLICATION_ALIASES = {
    "chrome": "chrome",
    "google chrome": "chrome",
    "гугл хром": "chrome",
    "хром": "chrome",
    "vscode": "vscode",
    "vs code": "vscode",
    "visual studio code": "vscode",
    "вс код": "vscode",
    "discord": "discord",
    "дискорд": "discord",
    "telegram": "telegram",
    "телеграм": "telegram",
}
_SERVICE_URLS = {
    "youtube": "https://www.youtube.com",
    "ютуб": "https://www.youtube.com",
}

# These patterns intentionally target externally verifiable information rather
# than every question.  General educational questions (for example, "что такое
# TCP?") continue to the conversational LLM.
_EXPLICIT_WEB_PATTERNS = (
    "найди в интернете", "найди информацию", "найди всю информацию",
    "поищи", "поиск", "загугли", "в интернете", "найдите",
)
_EXTERNAL_FACT_PATTERNS = (
    "дата рождения", "когда родил", "когда родилась", "когда родился",
    "кто такой", "кто такая", "кто это", "что известно о",
    "последние новости", "свежие новости", "что сейчас происходит",
    "какая сейчас цена", "текущая цена", "курс валют", "актуальн",
)
_MATH_EXPRESSION = re.compile(r"^[0-9\s+\-*/().,%]+$")


def route_explicit_action(user_text: str) -> RoutedAction | None:
    """Return a tool call only for a clearly expressed, supported action."""
    close_match = _CLOSE_ACTION_VERB.match(user_text)
    if close_match:
        target = user_text[close_match.end():].strip().strip(" .,!?:;\"'")
        app_name = _APPLICATION_ALIASES.get(" ".join(target.casefold().split()))
        if app_name:
            return RoutedAction("CLOSE_APP", "close_app", {"app_name": app_name})
        return None

    match = _OPEN_ACTION_VERB.match(user_text)
    if not match:
        return None

    target = user_text[match.end():].strip().strip(" .,!?:;\"'")
    if not target:
        return None

    # A domain is a URL, even if its name happens to contain an app name.
    url_match = _URL.fullmatch(target)
    if url_match:
        return RoutedAction("OPEN_BROWSER", "open_browser", {"url": url_match.group("url")})

    service_url = _SERVICE_URLS.get(" ".join(target.casefold().split()))
    if service_url:
        return RoutedAction("OPEN_BROWSER", "open_browser", {"url": service_url})

    app_name = _APPLICATION_ALIASES.get(" ".join(target.casefold().split()))
    if app_name:
        return RoutedAction("OPEN_APP", "open_app", {"app_name": app_name})
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
    if any(pattern in normalized for pattern in _EXPLICIT_WEB_PATTERNS):
        return RoutedAction("WEB_SEARCH", "web_search", {"query": user_text.strip()})
    if any(pattern in normalized for pattern in _EXTERNAL_FACT_PATTERNS):
        return RoutedAction("WEB_SEARCH", "web_search", {"query": user_text.strip()})
    return None
