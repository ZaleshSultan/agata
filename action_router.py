"""Deterministic routing for unambiguous local actions.

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


_ACTION_VERB = re.compile(
    r"^\s*(?:открой|открыть|запусти|запустить|включи|включить)\s+",
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
}


def route_explicit_action(user_text: str) -> RoutedAction | None:
    """Return a tool call only for a clearly expressed, supported action."""
    match = _ACTION_VERB.match(user_text)
    if not match:
        return None

    target = user_text[match.end():].strip().strip(" .,!?:;\"'")
    if not target:
        return None

    # A domain is a URL, even if its name happens to contain an app name.
    url_match = _URL.fullmatch(target)
    if url_match:
        return RoutedAction("OPEN_BROWSER", "open_browser", {"url": url_match.group("url")})

    app_name = _APPLICATION_ALIASES.get(" ".join(target.casefold().split()))
    if app_name:
        return RoutedAction("OPEN_APP", "open_app", {"app_name": app_name})
    return None
