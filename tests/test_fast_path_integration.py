"""
Integration tests for Agent Fast Path (deterministic vs conversational routing).
"""

import json
import unittest
from unittest.mock import MagicMock, patch

from agent import Agent


class FakeLLM:
    def __init__(self, content: str = "Диалоговый ответ от LLM") -> None:
        self.content = content
        self.calls: list[dict] = []

    def chat(self, messages, tools=None, think=None):
        self.calls.append({"messages": messages, "tools": tools, "think": think})
        return {"type": "text", "content": self.content}

    def chat_stream(self, messages, think=None):
        self.calls.append({"messages": messages, "tools": None, "think": think})
        yield "Диалоговый "
        yield "ответ от "
        yield "LLM."


class TestFastPathIntegration(unittest.TestCase):

    def setUp(self) -> None:
        self.agent = Agent()
        self.fake_llm = FakeLLM()
        self.agent._llm = self.fake_llm

    @patch("agent.execute_tool")
    def test_open_chrome_fast_path(self, mock_tool) -> None:
        mock_tool.return_value = json.dumps({"success": True, "application": "Chrome"})
        with patch.object(self.agent.memory, "prepare_context") as mock_prep:
            answer = self.agent.run("Открой Chrome")

        self.assertEqual(len(self.fake_llm.calls), 0, "LLM must NOT be called for 'Открой Chrome'")
        mock_prep.assert_not_called()
        self.assertIn("открыт", answer)

    @patch("agent.execute_tool")
    def test_open_telegram_inflections_fast_path(self, mock_tool) -> None:
        mock_tool.return_value = json.dumps({"success": True, "application": "Telegram"})
        for cmd in ["Открой Telegram", "Открой телеграмму", "Открой телегу", "Запусти телеграм"]:
            with self.subTest(cmd=cmd):
                self.fake_llm.calls.clear()
                with patch.object(self.agent.memory, "prepare_context") as mock_prep:
                    answer = self.agent.run(cmd)
                self.assertEqual(len(self.fake_llm.calls), 0, f"LLM must NOT be called for '{cmd}'")
                mock_prep.assert_not_called()
                self.assertIn("Telegram", answer)

    @patch("agent.execute_tool")
    def test_open_spotify_slang_fast_path(self, mock_tool) -> None:
        mock_tool.return_value = json.dumps({"success": True, "application": "Spotify"})
        for cmd in ["Открой Spotify", "Открой спотик", "Открой споти", "Запусти спотифай"]:
            with self.subTest(cmd=cmd):
                self.fake_llm.calls.clear()
                with patch.object(self.agent.memory, "prepare_context") as mock_prep:
                    answer = self.agent.run(cmd)
                self.assertEqual(len(self.fake_llm.calls), 0, f"LLM must NOT be called for '{cmd}'")
                mock_prep.assert_not_called()
                self.assertIn("Spotify", answer)

    @patch("agent.execute_tool")
    def test_open_youtube_fast_path(self, mock_tool) -> None:
        mock_tool.return_value = json.dumps({"success": True, "url": "https://www.youtube.com"})
        for cmd in ["Открой YouTube", "Открой ютуб", "Открой ютубчик"]:
            with self.subTest(cmd=cmd):
                self.fake_llm.calls.clear()
                with patch.object(self.agent.memory, "prepare_context") as mock_prep:
                    answer = self.agent.run(cmd)
                self.assertEqual(len(self.fake_llm.calls), 0, f"LLM must NOT be called for '{cmd}'")
                mock_prep.assert_not_called()
                self.assertIn("YouTube", answer)

    @patch("agent.execute_tool")
    def test_calculator_natural_math_fast_path(self, mock_tool) -> None:
        mock_tool.side_effect = lambda tool, args: args.get("expression", "")
        queries = [
            "Двенадцать плюс семнадцать",
            "Двадцать пять умножить на восемь",
            "Сто разделить на четыре",
            "Двадцать пять минус семь",
        ]
        for query in queries:
            with self.subTest(query=query):
                self.fake_llm.calls.clear()
                with patch.object(self.agent.memory, "prepare_context") as mock_prep:
                    answer = self.agent.run(query)
                self.assertEqual(len(self.fake_llm.calls), 0, f"LLM must NOT be called for calculation '{query}'")
                mock_prep.assert_not_called()

    @patch("agent.execute_tool")
    def test_datetime_fast_path(self, mock_tool) -> None:
        mock_tool.return_value = "Сейчас Вторник, 11 августа 2026, 08:16."
        for query in ["Который час", "Сколько времени", "Какое сегодня число"]:
            with self.subTest(query=query):
                self.fake_llm.calls.clear()
                with patch.object(self.agent.memory, "prepare_context") as mock_prep:
                    answer = self.agent.run(query)
                self.assertEqual(len(self.fake_llm.calls), 0, f"LLM must NOT be called for '{query}'")
                mock_prep.assert_not_called()
                self.assertIn("Сейчас", answer)

    def test_conversational_reaches_llm(self) -> None:
        query = "Расскажи мне почему люди видят сны"
        with patch.object(self.agent.memory, "prepare_context", return_value={"working_memory": "", "relevant_memories": "", "current_datetime": ""}) as mock_prep:
            answer = self.agent.run(query)

        self.assertEqual(len(self.fake_llm.calls), 1, "Conversational query MUST reach LLM")
        mock_prep.assert_called_once()
        self.assertIn("Диалоговый ответ от LLM", answer)


if __name__ == "__main__":
    unittest.main()
