"""Regression test suite for deterministic command routing and media/browser automation.

Verifies that all standard system, media, browser, and content commands execute
cleanly through the deterministic router without requiring an LLM or Ollama.
"""

import json
import unittest
from unittest.mock import AsyncMock, patch

from action_router import (
    route_calculation,
    route_content_play,
    route_datetime,
    route_explicit_action,
    route_media,
    route_volume,
    route_web_intent,
)
from agent import Agent
from media.base import ActionResult, MediaResult
from media.ranking import rank_media_results
from stt_normalize import normalize_stt
from target_resolver import resolve_target


class FakeLLM:
    """Mock LLM that logs calls and raises error if invoked unexpectedly."""

    def __init__(self, content: str = "LLM ответ") -> None:
        self.content = content
        self.calls: list[dict] = []

    def chat(self, messages, tools=None, think=None):
        self.calls.append({"messages": messages, "tools": tools, "think": think})
        return {"type": "text", "content": self.content}


class RoutingRegressionTests(unittest.TestCase):

    def setUp(self) -> None:
        self.agent = Agent()
        self.fake_llm = FakeLLM()
        self.agent._llm = self.fake_llm

    # ── 1. APP OPEN / TARGET RESOLVER ──────────────────────────────────────

    def test_open_chrome_without_llm(self) -> None:
        """'Открой хром' → OPEN_APP Chrome without LLM."""
        norm = normalize_stt("Открой хром")
        action = route_explicit_action(norm)
        self.assertIsNotNone(action)
        assert action is not None
        self.assertEqual(action.tool_name, "open_app")
        self.assertEqual(action.kind, "OPEN_APP")

        with patch("agent.execute_tool", return_value=json.dumps({"success": True, "application": "Chrome"})) as mock_exec:
            answer = self.agent.run("Открой хром")
        self.assertIn("открыт", answer)
        self.assertEqual(len(self.fake_llm.calls), 0)

    def test_open_firefox_without_llm(self) -> None:
        """'открой файрфокс' / 'открой фаерфокс' → OPEN_APP Firefox without LLM."""
        for cmd in ["открой файрфокс", "открой фаерфокс", "открой firefox", "открой мозилла"]:
            norm = normalize_stt(cmd)
            action = route_explicit_action(norm)
            self.assertIsNotNone(action, f"Failed to route '{cmd}'")
            assert action is not None
            self.assertIn(action.tool_name, ["open_app", "open_browser"])

        with patch("agent.execute_tool", return_value=json.dumps({"success": True, "application": "Firefox"})):
            answer = self.agent.run("открой файрфокс")
        self.assertIn("открыт", answer)
        self.assertEqual(len(self.fake_llm.calls), 0)

    def test_open_telegram_without_llm(self) -> None:
        """'открой телеграм' / 'открой телегу' → OPEN_APP Telegram without LLM."""
        for cmd in ["открой телеграм", "открой телегу", "открой telegram"]:
            norm = normalize_stt(cmd)
            action = route_explicit_action(norm)
            self.assertIsNotNone(action, f"Failed to route '{cmd}'")
            assert action is not None
            self.assertIn(action.tool_name, ["open_app", "open_browser"])

        with patch("agent.execute_tool", return_value=json.dumps({"success": True, "application": "Telegram"})):
            answer = self.agent.run("открой телегу")
        self.assertIn("открыт", answer)
        self.assertEqual(len(self.fake_llm.calls), 0)

    def test_open_youtube_without_llm(self) -> None:
        """'открой ютуб' → OPEN_BROWSER https://www.youtube.com without LLM."""
        for cmd in ["открой ютуб", "открой youtube", "открой ютьюб"]:
            norm = normalize_stt(cmd)
            action = route_explicit_action(norm)
            self.assertIsNotNone(action, f"Failed to route '{cmd}'")
            assert action is not None
            self.assertEqual(action.tool_name, "open_browser")
            self.assertEqual(action.arguments["url"], "https://www.youtube.com")

        with patch("agent.execute_tool", return_value=json.dumps({"success": True, "url": "https://www.youtube.com"})):
            answer = self.agent.run("открой ютуб")
        self.assertIn("браузере", answer)
        self.assertEqual(len(self.fake_llm.calls), 0)

    # ── 2. MEDIA TRANSPORT ─────────────────────────────────────────────────

    def test_pause_without_llm(self) -> None:
        """'пауза' / 'поставь на паузу' → MEDIA pause without LLM."""
        for cmd in ["пауза", "поставь на паузу", "на паузу"]:
            norm = normalize_stt(cmd)
            action = route_media(norm)
            self.assertIsNotNone(action, f"Failed to route '{cmd}'")
            assert action is not None
            self.assertEqual(action.arguments["action"], "pause")

        with patch("agent.execute_tool", return_value=json.dumps({"success": True})):
            answer = self.agent.run("пауза")
        self.assertIn("паузу", answer.lower())
        self.assertEqual(len(self.fake_llm.calls), 0)

    def test_resume_without_llm(self) -> None:
        """'продолжи' / 'возобнови' → MEDIA resume without LLM."""
        for cmd in ["продолжи", "возобнови", "продолжить"]:
            norm = normalize_stt(cmd)
            action = route_media(norm)
            self.assertIsNotNone(action, f"Failed to route '{cmd}'")
            assert action is not None
            self.assertEqual(action.arguments["action"], "resume")

        with patch("agent.execute_tool", return_value=json.dumps({"success": True})):
            answer = self.agent.run("продолжи")
        self.assertIn("продолжила", answer.lower())
        self.assertEqual(len(self.fake_llm.calls), 0)

    def test_next_track_without_llm(self) -> None:
        """'следующая песня' / 'следующий трек' / 'следующая' → MEDIA next."""
        for cmd in ["следующая песня", "следующий трек", "следующая", "переключи трек"]:
            norm = normalize_stt(cmd)
            action = route_media(norm)
            self.assertIsNotNone(action, f"Failed to route '{cmd}'")
            assert action is not None
            self.assertEqual(action.arguments["action"], "next")

        with patch("agent.execute_tool", return_value=json.dumps({"success": True})):
            answer = self.agent.run("следующая песня")
        self.assertIn("следующий трек", answer.lower())
        self.assertEqual(len(self.fake_llm.calls), 0)

    def test_previous_track_without_llm(self) -> None:
        """'предыдущая песня' / 'предыдущий трек' / 'предыдущая' → MEDIA previous."""
        for cmd in ["предыдущая песня", "предыдущий трек", "предыдущая", "назад трек"]:
            norm = normalize_stt(cmd)
            action = route_media(norm)
            self.assertIsNotNone(action, f"Failed to route '{cmd}'")
            assert action is not None
            self.assertEqual(action.arguments["action"], "previous")

        with patch("agent.execute_tool", return_value=json.dumps({"success": True})):
            answer = self.agent.run("предыдущая песня")
        self.assertIn("предыдущий трек", answer.lower())
        self.assertEqual(len(self.fake_llm.calls), 0)

    # ── 3. CONTENT PLAY (YouTube / SoundCloud) ─────────────────────────────

    def test_youtube_content_without_llm(self) -> None:
        """'поставь лекцию про нейробиологию' → PLAY_CONTENT youtube."""
        cmd = "поставь лекцию про нейробиологию"
        norm = normalize_stt(cmd)
        action = route_content_play(norm)
        self.assertIsNotNone(action)
        assert action is not None
        self.assertEqual(action.tool_name, "play_content")
        self.assertEqual(action.arguments["platform"], "youtube")
        self.assertIn("лекцию про нейробиологию", action.arguments["query"])

        with patch("agent.execute_tool", return_value=json.dumps({"success": True, "title": "Лекция про нейробиологию"})):
            answer = self.agent.run(cmd)
        self.assertIn("Включила", answer)
        self.assertEqual(len(self.fake_llm.calls), 0)

    def test_soundcloud_content_without_llm(self) -> None:
        """'поставь монополию в саундклауде' → PLAY_CONTENT soundcloud (NOT write_diary)."""
        cmd = "поставь монополию в саундклауде"
        norm = normalize_stt(cmd)
        action = route_content_play(norm)
        self.assertIsNotNone(action)
        assert action is not None
        self.assertEqual(action.tool_name, "play_content")
        self.assertEqual(action.arguments["platform"], "soundcloud")
        self.assertEqual(action.arguments["query"], "монополию")

        with patch("agent.execute_tool", return_value=json.dumps({"success": True, "title": "Монополия"})):
            answer = self.agent.run(cmd)
        self.assertIn("Включила", answer)
        self.assertEqual(len(self.fake_llm.calls), 0)

    # ── 4. WEB / WEATHER ROUTING ───────────────────────────────────────────

    def test_weather_routes_without_llm(self) -> None:
        """'какая сейчас погода?' → WEB_SEARCH web intent."""
        for cmd in ["какая сейчас погода?", "погода сегодня", "прогноз погоды", "что с погодой"]:
            norm = normalize_stt(cmd)
            action = route_web_intent(norm)
            self.assertIsNotNone(action, f"Failed to route '{cmd}' to web_search")
            assert action is not None
            self.assertEqual(action.tool_name, "web_search")

    def test_web_search_routes_without_llm(self) -> None:
        """'найди информацию про нейробиологию' → WEB_SEARCH."""
        cmd = "найди информацию про нейробиологию"
        norm = normalize_stt(cmd)
        action = route_web_intent(norm)
        self.assertIsNotNone(action)
        assert action is not None
        self.assertEqual(action.tool_name, "web_search")

    # ── 5. FALSE POSITIVE GUARDS ───────────────────────────────────────────

    def test_false_positive_pause_not_content(self) -> None:
        """'поставь на паузу' must be MEDIA pause, NOT content search."""
        action = route_content_play("поставь на паузу")
        self.assertIsNone(action)
        media_action = route_media("поставь на паузу")
        self.assertIsNotNone(media_action)
        assert media_action is not None
        self.assertEqual(media_action.arguments["action"], "pause")

    def test_false_positive_diary_not_content(self) -> None:
        """'поставь запись в дневник' must NOT be content play."""
        action = route_content_play("поставь запись в дневник")
        self.assertIsNone(action)

    def test_bare_music_is_media_transport_not_search(self) -> None:
        """'включи музыку' must be media transport, NOT a search query for 'музыку'."""
        content_action = route_content_play("включи музыку")
        self.assertIsNone(content_action)
        media_action = route_media("включи музыку")
        self.assertIsNotNone(media_action)

    # ── 6. LLM INTEGRITY / NO SILENT SUCCESS ───────────────────────────────

    def test_llm_cannot_claim_success_without_tool(self) -> None:
        """Agent must report failure when a tool execution fails, never claim success."""
        with patch("agent.execute_tool", return_value=json.dumps({"success": False, "error": "Приложение не найдено"})):
            answer = self.agent.run("Открой НеизвестноеПриложение123")
        # Should not claim success
        self.assertNotIn("открыт.", answer)

    def test_unknown_app_falls_back_cleanly(self) -> None:
        """Unknown app target returns None from resolver, does not crash."""
        res = resolve_target("какой-то несуществующий апп 999")
        self.assertIsNone(res)

    # ── 7. STT NORMALIZATION VARIATIONS ────────────────────────────────────

    def test_stt_variations_normalize_properly(self) -> None:
        """Test variations in STT output produce clean canonical input for routing."""
        self.assertEqual(normalize_stt("Агата, открой хром."), "открой хром")
        self.assertEqual(normalize_stt("Огадок, включи музыку."), "включи музыку")
        self.assertEqual(normalize_stt("можешь открыть файрфокс"), "открой файрфокс")
        self.assertEqual(normalize_stt("поставь трек в саундклауде"), "поставь трек в саундклауд")

    # ── 8. RANKING LOGIC ───────────────────────────────────────────────────

    def test_media_ranking_exact_match(self) -> None:
        """Exact query in title must rank above partial match."""
        items = [
            MediaResult(url="https://youtube.com/watch?v=1", title="Общий курс биологии", position=0),
            MediaResult(url="https://youtube.com/watch?v=2", title="Лекция про нейробиологию для начинающих", position=1),
            MediaResult(url="https://youtube.com/watch?v=3", title="Нейробиология мозга", position=2),
        ]
        ranked = rank_media_results("лекция про нейробиологию", items)
        self.assertEqual(ranked[0].url, "https://youtube.com/watch?v=2")


if __name__ == "__main__":
    unittest.main()
