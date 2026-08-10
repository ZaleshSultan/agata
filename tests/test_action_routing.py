import json
import unittest
from unittest.mock import patch

from action_router import route_calculation, route_explicit_action, route_web_intent
from agent import Agent
from tools import get_tool_schemas, list_tool_names, system_control
from tools.calc import calculate
from tools.web_search import MAX_RESULTS, web_search


class FakeLLM:
    def __init__(self, content: str = "Тестовый ответ") -> None:
        self.content = content
        self.calls: list[dict] = []

    def chat(self, messages, tools=None, think=None):
        self.calls.append({"messages": messages, "tools": tools, "think": think})
        return {"type": "text", "content": self.content}


class ActionRoutingTests(unittest.TestCase):
    def test_chrome_russian_alias_routes_to_registered_tool(self) -> None:
        action = route_explicit_action("Открой хром")
        self.assertIsNotNone(action)
        assert action is not None
        self.assertEqual((action.tool_name, action.arguments), ("open_app", {"app_name": "chrome"}))

    def test_vscode_routes_to_registered_tool(self) -> None:
        action = route_explicit_action("Запусти VS Code")
        self.assertIsNotNone(action)
        assert action is not None
        self.assertEqual((action.tool_name, action.arguments), ("open_app", {"app_name": "vscode"}))

    def test_discord_routes_to_registered_tool(self) -> None:
        action = route_explicit_action("Открой Discord")
        self.assertIsNotNone(action)
        assert action is not None
        self.assertEqual((action.tool_name, action.arguments), ("open_app", {"app_name": "discord"}))

    def test_close_chrome_routes_to_confirmed_tool(self) -> None:
        action = route_explicit_action("Закрой Chrome")
        self.assertIsNotNone(action)
        assert action is not None
        self.assertEqual((action.tool_name, action.arguments), ("close_app", {"app_name": "chrome"}))

    def test_domain_routes_to_browser_not_application(self) -> None:
        action = route_explicit_action("Открой youtube.com")
        self.assertIsNotNone(action)
        assert action is not None
        self.assertEqual((action.tool_name, action.arguments), ("open_browser", {"url": "youtube.com"}))

    def test_youtube_name_routes_to_browser_not_application(self) -> None:
        action = route_explicit_action("Открой YouTube")
        self.assertIsNotNone(action)
        assert action is not None
        self.assertEqual((action.tool_name, action.arguments), ("open_browser", {"url": "https://www.youtube.com"}))

    def test_conversation_is_not_action(self) -> None:
        self.assertIsNone(route_explicit_action("Привет, как дела?"))

    def test_routed_success_is_reported_only_after_tool_success(self) -> None:
        agent = Agent()
        with patch("agent.execute_tool", return_value=json.dumps({"success": True, "application": "chrome"})) as execute:
            answer = agent.run("Открой Chrome")
        execute.assert_called_once_with("open_app", {"app_name": "chrome"})
        self.assertEqual(answer, "Chrome открыт.")

    def test_routed_failure_is_not_reported_as_success(self) -> None:
        agent = Agent()
        with patch("agent.execute_tool", return_value=json.dumps({"success": False, "error": "Chrome не найден"})):
            answer = agent.run("Открой Chrome")
        self.assertIn("Не удалось", answer)
        self.assertNotIn("открыт.", answer)

    def test_calculator_still_works(self) -> None:
        self.assertEqual(calculate("25 * 4"), "100")

    def test_chrome_is_whitelisted(self) -> None:
        self.assertIn("chrome", system_control.ALLOWED_APPS)

    def test_close_app_is_registered(self) -> None:
        self.assertIn("close_app", list_tool_names())

    def test_tool_schemas_support_exclusion(self) -> None:
        schemas = get_tool_schemas(exclude={"calculate", "web_search", "open_app"})
        names = {schema["function"]["name"] for schema in schemas}
        self.assertNotIn("calculate", names)
        self.assertNotIn("web_search", names)
        self.assertNotIn("open_app", names)
        self.assertIn("get_current_datetime", names)

    def test_action_bypasses_llm(self) -> None:
        agent = Agent()
        fake_llm = FakeLLM()
        agent._llm = fake_llm
        with patch("agent.execute_tool", return_value=json.dumps({"success": True, "application": "discord"})):
            answer = agent.run("Открой Discord")
        self.assertEqual(answer, "Discord открыт.")
        self.assertEqual(fake_llm.calls, [])

    def test_calculation_bypasses_classifier_and_llm(self) -> None:
        agent = Agent()
        fake_llm = FakeLLM()
        agent._llm = fake_llm
        with patch("agent.execute_tool", return_value="300") as execute:
            answer = agent.run("150+150?")
        execute.assert_called_once_with("calculate", {"expression": "150+150"})
        self.assertEqual(answer, "300")
        self.assertEqual(fake_llm.calls, [])

    def test_web_intent_recognizes_explicit_and_factual_queries(self) -> None:
        explicit = route_web_intent("Найди в интернете день рождения Сэди Синк")
        factual = route_web_intent("Дата рождения Сэди Синк")
        self.assertIsNotNone(explicit)
        self.assertIsNotNone(factual)
        assert explicit is not None and factual is not None
        self.assertEqual(explicit.tool_name, "web_search")
        self.assertEqual(factual.tool_name, "web_search")

    def test_general_knowledge_is_not_forced_to_web(self) -> None:
        self.assertIsNone(route_web_intent("Что такое TCP?"))

    def test_web_pipeline_uses_one_search_and_one_final_llm(self) -> None:
        agent = Agent()
        fake_llm = FakeLLM("Сэди Синк родилась согласно найденным источникам.")
        agent._llm = fake_llm
        search_result = json.dumps(
            {"success": True, "results": [{"title": "Source", "snippet": "Fact", "url": "https://example.test"}]}
        )
        with patch("agent.execute_tool", return_value=search_result) as execute:
            answer = agent.run("Дата рождения Сэди Синк")
        execute.assert_called_once_with("web_search", {"query": "Дата рождения Сэди Синк"})
        self.assertEqual(answer, "Сэди Синк родилась согласно найденным источникам.")
        self.assertEqual(len(fake_llm.calls), 1)
        self.assertIsNone(fake_llm.calls[0]["tools"])
        self.assertIs(fake_llm.calls[0]["think"], False)
        self.assertIn("Fact", fake_llm.calls[0]["messages"][0]["content"])

    def test_empty_web_result_does_not_call_llm(self) -> None:
        agent = Agent()
        fake_llm = FakeLLM()
        agent._llm = fake_llm
        with patch("agent.execute_tool", return_value=json.dumps({"success": True, "results": []})):
            answer = agent.run("Кто такой Залеш Султан?")
        self.assertIn("Не нашла", answer)
        self.assertEqual(fake_llm.calls, [])

    def test_web_search_uses_one_compact_backend_call(self) -> None:
        class StubDDGS:
            calls = []

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def text(self, query, max_results):
                self.calls.append((query, max_results))
                return [
                    {"title": "One", "body": "a" * 700, "href": "https://one.test"},
                    {"title": "Two", "body": "b" * 100, "href": "https://two.test"},
                ]

        with patch("tools.web_search.DDGS", StubDDGS):
            payload = json.loads(web_search("test query"))
        self.assertEqual(StubDDGS.calls, [("test query", MAX_RESULTS)])
        self.assertTrue(payload["success"])
        self.assertEqual(len(payload["results"]), 2)
        self.assertLessEqual(len(payload["results"][0]["snippet"]), 500)


if __name__ == "__main__":
    unittest.main()
