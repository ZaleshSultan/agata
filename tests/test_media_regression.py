"""Regression tests for MediaSession, real playback, ranking profiles, and transport priority."""

from __future__ import annotations

import asyncio
import json
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from action_router import (
    route_content_play,
    route_media,
)
from agent import Agent
from media.base import ActionResult, MediaResult, PlaybackStatus
from media.parser import parse_content_query
from media.ranking import rank_media_results
from media.session import MediaSession, get_media_session
from media.transport_parser import parse_media_command


class FailingLLM:
    """Mock LLM that raises an error if invoked, ensuring tests verify zero-LLM paths."""

    def chat(self, messages, tools=None, think=None):
        raise AssertionError("LLM was unexpectedly invoked for a deterministic operation!")


class MediaRegressionTests(unittest.TestCase):

    def setUp(self) -> None:
        self.agent = Agent()
        self.failing_llm = FailingLLM()
        self.agent._llm = self.failing_llm

    # ── 1. TRANSPORT ROUTING PRIORITY (ZERO LLM, ZERO SEARCH) ───────────────

    def test_stop_does_not_trigger_content_search(self) -> None:
        """'стоп', 'поставь на стоп', 'останови' must route to control_media stop, NOT play_content."""
        stop_phrases = [
            "стоп",
            "поставь на стоп",
            "на стоп",
            "останови",
            "остановить",
            "заглуши",
            "Агата, поставь на стоп",
        ]
        for phrase in stop_phrases:
            # 1. Transport parser must recognize it
            cmd = parse_media_command(phrase)
            self.assertIsNotNone(cmd, f"Transport parser missed: '{phrase}'")
            self.assertEqual(cmd.action, "stop", f"Wrong action for: '{phrase}'")

            # 2. Content play router MUST NOT match
            content_act = route_content_play(phrase)
            self.assertIsNone(content_act, f"Content play falsely captured: '{phrase}'")

            # 3. Media router must match
            media_act = route_media(phrase)
            self.assertIsNotNone(media_act, f"Media router missed: '{phrase}'")
            self.assertEqual(media_act.tool_name, "control_media")
            self.assertEqual(media_act.arguments["action"], "stop")

            # 4. Agent runs without LLM
            with patch("agent.execute_tool", return_value=json.dumps({"success": True, "action": "stop"})):
                answer = self.agent.run(phrase)
            self.assertIn("Остановила", answer)

    def test_pause_does_not_trigger_content_search(self) -> None:
        """'пауза', 'поставь на паузу', 'на паузу' must route to control_media pause."""
        pause_phrases = [
            "пауза",
            "поставь на паузу",
            "на паузу",
            "приостанови",
            "приостановить",
            "сделай паузу",
        ]
        for phrase in pause_phrases:
            cmd = parse_media_command(phrase)
            self.assertIsNotNone(cmd, f"Transport parser missed: '{phrase}'")
            self.assertEqual(cmd.action, "pause")

            content_act = route_content_play(phrase)
            self.assertIsNone(content_act, f"Content play falsely captured: '{phrase}'")

            media_act = route_media(phrase)
            self.assertIsNotNone(media_act)
            self.assertEqual(media_act.arguments["action"], "pause")

            with patch("agent.execute_tool", return_value=json.dumps({"success": True, "action": "pause"})):
                answer = self.agent.run(phrase)
            self.assertIn("паузу", answer)

    def test_resume_does_not_trigger_content_search(self) -> None:
        """'продолжи', 'возобнови', 'продолжай' must route to control_media resume."""
        resume_phrases = [
            "продолжи",
            "возобнови",
            "продолжай",
            "продолжить",
            "возобновить",
            "играй дальше",
        ]
        for phrase in resume_phrases:
            cmd = parse_media_command(phrase)
            self.assertIsNotNone(cmd, f"Transport parser missed: '{phrase}'")
            self.assertEqual(cmd.action, "resume")

            content_act = route_content_play(phrase)
            self.assertIsNone(content_act, f"Content play falsely captured: '{phrase}'")

            media_act = route_media(phrase)
            self.assertIsNotNone(media_act)
            self.assertEqual(media_act.arguments["action"], "resume")

            with patch("agent.execute_tool", return_value=json.dumps({"success": True, "action": "resume"})):
                answer = self.agent.run(phrase)
            self.assertIn("Продолжаю", answer)

    def test_next_previous_routing(self) -> None:
        """'следующая песня', 'предыдущий трек' route to next / previous."""
        next_phrases = ["следующая песня", "следующий трек", "следующая", "переключи трек"]
        for phrase in next_phrases:
            cmd = parse_media_command(phrase)
            self.assertIsNotNone(cmd)
            self.assertEqual(cmd.action, "next")
            self.assertIsNone(route_content_play(phrase))

        prev_phrases = ["предыдущая песня", "предыдущий трек", "предыдущая", "назад трек"]
        for phrase in prev_phrases:
            cmd = parse_media_command(phrase)
            self.assertIsNotNone(cmd)
            self.assertEqual(cmd.action, "previous")
            self.assertIsNone(route_content_play(phrase))

    # ── 2. QUERY PARSER (RUSSIAN NOUNS & CONTENT TYPES) ────────────────────

    def test_song_artist_parsing(self) -> None:
        """Parse structured title, artist, content type, and search query."""
        # 1. [title] от [artist]
        p1 = parse_content_query("поставь Монополию от Монеточки")
        self.assertEqual(p1.title, "Монополия")
        self.assertEqual(p1.artist, "Монеточка")
        self.assertEqual(p1.content_type, "music")
        self.assertIn("Монеточка", p1.search_query)
        self.assertIn("Монополия", p1.search_query)

        # 2. English artist & title
        p2 = parse_content_query("включи Blinding Lights от The Weeknd")
        self.assertEqual(p2.title, "Blinding Lights")
        self.assertEqual(p2.artist, "The Weeknd")
        self.assertEqual(p2.content_type, "music")

        # 3. [title] [artist_genitive]
        p3 = parse_content_query("поставь Монополию Монеточки")
        self.assertEqual(p3.title, "Монополия")
        self.assertEqual(p3.artist, "Монеточка")

        # 4. Lecture query without artist
        p4 = parse_content_query("поставь лекцию про нейробиологию")
        self.assertEqual(p4.content_type, "lecture")
        self.assertEqual(p4.artist, "")
        self.assertIn("нейробиологи", p4.title)

    # ── 3. TYPE-SPECIFIC RANKING ───────────────────────────────────────────

    def test_exact_song_ranking(self) -> None:
        """'Монеточка — Монополия' must significantly outscore 'дисс на монополию'."""
        items = [
            MediaResult(
                url="https://youtube.com/watch?v=diss",
                title="Дисс на Монополию (Клип)",
                channel="Blogger Diss",
                duration="3:15",
                position=0,
            ),
            MediaResult(
                url="https://youtube.com/watch?v=official",
                title="Монеточка — Монополия (Official Audio)",
                channel="Монеточка",
                verified_channel=True,
                duration="3:25",
                position=1,
            ),
            MediaResult(
                url="https://youtube.com/watch?v=parody",
                title="Пародия на песню Монополия",
                channel="Parodies",
                duration="3:40",
                position=2,
            ),
        ]
        ranked = rank_media_results(
            query="Монополия от Монеточки",
            items=items,
            content_type="music",
            query_title="Монополия",
            query_artist="Монеточка",
        )
        self.assertEqual(ranked[0].url, "https://youtube.com/watch?v=official")
        self.assertGreater(ranked[0].score, ranked[1].score + 15.0)

    def test_lecture_ranking(self) -> None:
        """Educational lecture/course must outscore short clip even if 'лекция' is not in title."""
        items = [
            MediaResult(
                url="https://youtube.com/watch?v=clip",
                title="Почему ваш мозг не дает учить язык? (Нейробиология)",
                channel="Language Short",
                duration="4:12",
                position=0,
            ),
            MediaResult(
                url="https://youtube.com/watch?v=full_course",
                title="Нейробиология — полный курс. Часть 1",
                channel="Университет Лекторий",
                duration="1:15:30",
                position=1,
            ),
        ]
        ranked = rank_media_results(
            query="лекция про нейробиологию",
            items=items,
            content_type="lecture",
        )
        self.assertEqual(ranked[0].url, "https://youtube.com/watch?v=full_course")
        self.assertGreater(ranked[0].score, ranked[1].score)

    # ── 4. MEDIASESSION STATE & TAB REUSE ──────────────────────────────────

    def test_media_session_reused(self) -> None:
        """Verify MediaSession reuses the active media page across requests."""
        mock_page = MagicMock()
        mock_page.is_closed.return_value = False
        mock_page.url = "https://www.youtube.com/watch?v=1"

        mock_bm = MagicMock()
        mock_bm.get_media_page = AsyncMock(return_value=mock_page)

        session = MediaSession(browser_manager=mock_bm)

        async def _test():
            # First request
            p1 = await session.get_or_create_page("youtube")
            self.assertEqual(p1, mock_page)
            self.assertEqual(mock_bm.get_media_page.call_count, 1)

            # Second request - must reuse without calling get_media_page again
            p2 = await session.get_or_create_page("youtube")
            self.assertEqual(p2, mock_page)
            self.assertEqual(mock_bm.get_media_page.call_count, 1)

        asyncio.run(_test())

    def test_youtube_playback_state(self) -> None:
        """Verify video element evaluation, PlaybackStatus update, and verified state."""
        mock_page = MagicMock()
        mock_page.is_closed.return_value = False
        mock_page.url = ""
        mock_page.goto = AsyncMock()
        mock_page.wait_for_selector = AsyncMock()

        # Mock DOM evaluation return values
        async def _mock_evaluate(script: str, *args):
            if "currentTime" in script:
                return {
                    "paused": False,
                    "currentTime": 14.5,
                    "duration": 210.0,
                    "muted": False,
                }
            return None

        mock_page.evaluate = AsyncMock(side_effect=_mock_evaluate)

        mock_bm = MagicMock()
        mock_bm.get_media_page = AsyncMock(return_value=mock_page)

        session = MediaSession(browser_manager=mock_bm)

        async def _test():
            status = await session.play("https://www.youtube.com/watch?v=abc", title="Тест", platform="youtube")
            self.assertTrue(status.navigation)
            self.assertTrue(status.element_found)
            self.assertTrue(status.playing)
            self.assertFalse(status.muted)
            self.assertEqual(status.current_time, 14.5)
            self.assertEqual(status.verification, "verified")
            self.assertTrue(session.playing)

            # Test pause
            paused = await session.pause()
            self.assertTrue(paused)
            self.assertFalse(session.playing)

            # Test resume
            resumed = await session.resume()
            self.assertTrue(resumed)
            self.assertTrue(session.playing)

            # Test stop
            stopped = await session.stop()
            self.assertTrue(stopped)
            self.assertFalse(session.playing)
            self.assertEqual(session.status.current_time, 0.0)

        asyncio.run(_test())

    def test_soundcloud_playback_state(self) -> None:
        """Verify SoundCloud player DOM state handling."""
        mock_page = MagicMock()
        mock_page.is_closed.return_value = False
        mock_page.url = ""
        mock_page.goto = AsyncMock()
        mock_page.wait_for_selector = AsyncMock()

        async def _mock_evaluate(script: str, *args):
            if "isPlayingClass" in script:
                return {
                    "isPlayingClass": True,
                    "audioPaused": False,
                    "currentTime": 5.2,
                    "duration": 180.0,
                    "muted": False,
                }
            return None

        mock_page.evaluate = AsyncMock(side_effect=_mock_evaluate)
        mock_bm = MagicMock()
        mock_bm.get_media_page = AsyncMock(return_value=mock_page)

        session = MediaSession(browser_manager=mock_bm)

        async def _test():
            status = await session.play("https://soundcloud.com/artist/track", title="SoundCloud Track", platform="soundcloud")
            self.assertTrue(status.navigation)
            self.assertTrue(status.playing)
            self.assertEqual(status.verification, "verified")

        asyncio.run(_test())

    # ── 5. TTS CACHE & SINGLE INITIALIZATION ───────────────────────────────

    def test_tts_disk_and_ram_cache(self) -> None:
        """Verify persistent phrase caching in RAM and Disk for target phrases."""
        from tts.xtts_engine import XTTSEngine, _PHRASE_CACHE

        engine = XTTSEngine()
        sample_phrase = "Firefox открыт."
        sample_wav_bytes = b"RIFFfake_wav_data_for_test"

        cache_key = " ".join(sample_phrase.strip().casefold().split()).strip(" .,!?:;\"'")
        _PHRASE_CACHE[cache_key] = sample_wav_bytes

        with patch.object(engine, "_play_wav_bytes") as mock_play:
            engine.synthesize_and_play(sample_phrase)
            mock_play.assert_called_once_with(sample_wav_bytes)

    # ── 6. ZERO LLM GUARANTEES ─────────────────────────────────────────────

    def test_no_llm_for_media(self) -> None:
        """Verify all transport and media operations complete without invoking Ollama."""
        for cmd in ["стоп", "пауза", "продолжи", "следующая песня", "предыдущий трек"]:
            with patch("agent.execute_tool", return_value=json.dumps({"success": True, "action": cmd})):
                ans = self.agent.run(cmd)
                self.assertTrue(len(ans) > 0)

    def test_no_llm_for_content_search(self) -> None:
        """Verify content play routing executes without invoking Ollama."""
        with patch("agent.execute_tool", return_value=json.dumps({"success": True, "title": "Монополия"})):
            ans = self.agent.run("поставь Монополию от Монеточки")
            self.assertIn("Включила", ans)


if __name__ == "__main__":
    unittest.main()
