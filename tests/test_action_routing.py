import json
import unittest
from unittest.mock import patch

from action_router import route_explicit_action
from agent import Agent
from tools import system_control
from tools.calc import calculate


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

    def test_domain_routes_to_browser_not_application(self) -> None:
        action = route_explicit_action("Открой youtube.com")
        self.assertIsNotNone(action)
        assert action is not None
        self.assertEqual((action.tool_name, action.arguments), ("open_browser", {"url": "youtube.com"}))

    def test_conversation_is_not_action(self) -> None:
        self.assertIsNone(route_explicit_action("Привет, как дела?"))

    def test_routed_success_is_reported_only_after_tool_success(self) -> None:
        agent = Agent()
        with patch("agent.execute_tool", return_value=json.dumps({"success": True, "application": "chrome"})) as execute:
            answer = agent.run("Открой Chrome")
        execute.assert_called_once_with("open_app", {"app_name": "chrome"})
        self.assertEqual(answer, "chrome открыт.")

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


if __name__ == "__main__":
    unittest.main()
