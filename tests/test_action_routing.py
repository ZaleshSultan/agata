import json
import unittest
from unittest.mock import patch

from action_router import route_calculation, route_datetime, route_explicit_action, route_web_intent
from agent import Agent
from tools import get_tool_schemas, list_tool_names, system_control
from tools.calc import calculate
from tools.datetime_tool import TIMEZONE_NAME, get_current_datetime
from tools.web_search import MAX_RESULTS, web_search


class FakeLLM:
    def __init__(self, content: str = "Тестовый ответ") -> None:
        self.content = content
        self.calls: list[dict] = []

    def chat(self, messages, tools=None, think=None):
        self.calls.append({"messages": messages, "tools": tools, "think": think})
        return {"type": "text", "content": self.content}


class ActionRoutingTests(unittest.TestCase):

    # ── OPEN ACTIONS ───────────────────────────────────────────────────────

    def test_open_youtube(self) -> None:
        """'открой YouTube' → OPEN_BROWSER https://www.youtube.com"""
        action = route_explicit_action("открой YouTube")
        self.assertIsNotNone(action)
        assert action is not None
        self.assertEqual(action.tool_name, "open_browser")
        self.assertEqual(action.arguments["url"], "https://www.youtube.com")

    def test_open_spotify_web(self) -> None:
        """'открой Spotify' → OPEN_BROWSER https://open.spotify.com"""
        action = route_explicit_action("открой Spotify")
        self.assertIsNotNone(action)
        assert action is not None
        self.assertEqual(action.tool_name, "open_browser")
        self.assertEqual(action.arguments["url"], "https://open.spotify.com")

    def test_open_github(self) -> None:
        """'открой GitHub' → OPEN_BROWSER https://github.com"""
        action = route_explicit_action("открой GitHub")
        self.assertIsNotNone(action)
        assert action is not None
        self.assertEqual(action.tool_name, "open_browser")
        self.assertEqual(action.arguments["url"], "https://github.com")

    def test_chrome_russian_alias_routes_to_open_app(self) -> None:
        action = route_explicit_action("Открой хром")
        self.assertIsNotNone(action)
        assert action is not None
        self.assertEqual(action.tool_name, "open_app")
        self.assertEqual(action.kind, "OPEN_APP")

    def test_vscode_routes_to_open_app(self) -> None:
        action = route_explicit_action("Запусти VS Code")
        self.assertIsNotNone(action)
        assert action is not None
        self.assertEqual(action.tool_name, "open_app")
        self.assertEqual(action.kind, "OPEN_APP")

    # ── CLOSE ACTIONS ──────────────────────────────────────────────────────

    def test_close_youtube(self) -> None:
        """'закрой YouTube' → CLOSE_TARGET / close_target"""
        action = route_explicit_action("закрой YouTube")
        self.assertIsNotNone(action)
        assert action is not None
        self.assertEqual(action.tool_name, "close_target")
        self.assertEqual(action.kind, "CLOSE_TARGET")

    def test_close_chrome(self) -> None:
        """'закрой Chrome' → CLOSE_APP / close_app"""
        action = route_explicit_action("закрой Chrome")
        self.assertIsNotNone(action)
        assert action is not None
        self.assertEqual(action.tool_name, "close_app")
        self.assertEqual(action.kind, "CLOSE_APP")

    def test_close_vscode(self) -> None:
        """'закрой VS Code' → CLOSE_APP / close_app"""
        action = route_explicit_action("закрой VS Code")
        self.assertIsNotNone(action)
        assert action is not None
        self.assertEqual(action.tool_name, "close_app")
        self.assertEqual(action.kind, "CLOSE_APP")

    # ── WEB SEARCH ROUTING ─────────────────────────────────────────────────

    def test_web_search_explicit_query(self) -> None:
        """'кто такой Джеффри Эпштейн?' → WEB_SEARCH"""
        action = route_web_intent("кто такой Джеффри Эпштейн?")
        self.assertIsNotNone(action)
        assert action is not None
        self.assertEqual(action.tool_name, "web_search")

    def test_normal_chat_not_web_search(self) -> None:
        """'как дела?' → NOT web search"""
        action = route_web_intent("как дела?")
        self.assertIsNone(action)

        talk = route_web_intent("поговорим")
        self.assertIsNone(talk)

    # ── DATETIME ROUTING ───────────────────────────────────────────────────

    def test_current_datetime(self) -> None:
        """'который час?' → DATETIME action"""
        action = route_datetime("который час?")
        self.assertIsNotNone(action)
        assert action is not None
        self.assertEqual(action.tool_name, "get_current_datetime")

        date_action = route_datetime("какая сегодня дата?")
        self.assertIsNotNone(date_action)
        assert date_action is not None
        self.assertEqual(date_action.tool_name, "get_current_datetime")

    def test_current_timezone(self) -> None:
        """Verify timezone constant is Asia/Almaty"""
        self.assertEqual(TIMEZONE_NAME, "Asia/Almaty")

    # ── WHISPER PUNCTUATION TOLERANCE ──────────────────────────────────────

    def test_open_comma_vscode(self) -> None:
        action = route_explicit_action("открой, VS Code")
        self.assertIsNotNone(action)
        assert action is not None
        self.assertEqual(action.tool_name, "open_app")

    def test_agata_open_vscode(self) -> None:
        action = route_explicit_action("Агата, открой VS Code")
        self.assertIsNotNone(action)
        assert action is not None
        self.assertEqual(action.tool_name, "open_app")

    def test_conversation_is_not_action(self) -> None:
        self.assertIsNone(route_explicit_action("Привет, как дела?"))

    # ── AGENT INTEGRATION ──────────────────────────────────────────────────

    def test_routed_success_is_reported_only_after_tool_success(self) -> None:
        agent = Agent()
        with patch("agent.execute_tool", return_value=json.dumps({"success": True, "application": "Chrome"})) as execute:
            answer = agent.run("Открой Chrome")
        execute.assert_called_once()
        self.assertIn("открыт", answer)

    def test_routed_failure_is_not_reported_as_success(self) -> None:
        agent = Agent()
        with patch("agent.execute_tool", return_value=json.dumps({"success": False, "error": "Chrome не найден"})):
            answer = agent.run("Открой Chrome")
        self.assertIn("Не удалось", answer)
        self.assertNotIn("открыт.", answer)

    def test_calculator_still_works(self) -> None:
        self.assertEqual(calculate("25 * 4"), "100")

    def test_close_target_is_registered(self) -> None:
        self.assertIn("close_target", list_tool_names())

    def test_action_bypasses_llm(self) -> None:
        agent = Agent()
        fake_llm = FakeLLM()
        agent._llm = fake_llm
        with patch("agent.execute_tool", return_value=json.dumps({"success": True, "application": "Discord"})):
            answer = agent.run("Открой Discord")
        self.assertIn("Discord", answer)
        self.assertEqual(fake_llm.calls, [])

    def test_datetime_bypasses_classifier_and_llm(self) -> None:
        agent = Agent()
        fake_llm = FakeLLM()
        agent._llm = fake_llm
        with patch("agent.execute_tool", return_value="Сейчас Вторник, 11 августа 2026, 08:16.") as execute:
            answer = agent.run("который час?")
        execute.assert_called_once_with("get_current_datetime", {})
        self.assertEqual(answer, "Сейчас Вторник, 11 августа 2026, 08:16.")
        self.assertEqual(fake_llm.calls, [])

    def test_web_search_filters_blocked_domains(self) -> None:
        from tools.web_search import _is_blocked
        self.assertTrue(_is_blocked("https://poiski.pro/person/123"))
        self.assertTrue(_is_blocked("http://nashi-predki.ru/page"))
        self.assertFalse(_is_blocked("https://ru.wikipedia.org/wiki/Test"))


if __name__ == "__main__":
    unittest.main()
