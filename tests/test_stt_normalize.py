"""
Тесты для stt_normalize.normalize_stt() и расширенного action_router.

Покрытие:
- Голосовые варианты команд → OPEN_APP / CLOSE_APP
- Aliases приложений
- Коррекция имени ассистента
- Убирание обращения к ассистенту
- Нормализация формы глагола
- Вопросы НЕ роутятся как action
"""

import unittest

from stt_normalize import normalize_stt
from action_router import route_explicit_action


class NormalizationTests(unittest.TestCase):

    # ── Глагольные формы ───────────────────────────────────────────────────

    def test_open_first_person_future(self) -> None:
        """'открою хром' → 'открой хром'"""
        self.assertEqual(normalize_stt("открою хром"), "открой хром")

    def test_open_first_person_present(self) -> None:
        """'открываю хром' → 'открой хром'"""
        self.assertEqual(normalize_stt("открываю хром"), "открой хром")

    def test_launch_first_person_future(self) -> None:
        """'запущу хром' → 'запусти хром'"""
        self.assertEqual(normalize_stt("запущу хром"), "запусти хром")

    def test_launch_first_person_present(self) -> None:
        """'запускаю хром' → 'запусти хром'"""
        self.assertEqual(normalize_stt("запускаю хром"), "запусти хром")

    def test_close_first_person_future(self) -> None:
        """'закрою хром' → 'закрой хром'"""
        self.assertEqual(normalize_stt("закрою хром"), "закрой хром")

    # ── Имя ассистента ─────────────────────────────────────────────────────

    def test_ogadok_corrected_to_agata(self) -> None:
        """'Привет, Огадок' → имя исправлено"""
        result = normalize_stt("Привет, Огадок")
        self.assertIn("агата", result.lower())
        self.assertNotIn("огадок", result.lower())

    def test_ogata_corrected_to_agata(self) -> None:
        """'Огата открой хром' → 'Агата' / обращение удалено"""
        result = normalize_stt("Огата открой хром")
        self.assertNotIn("огата", result.lower())

    def test_agat_corrected_to_agata(self) -> None:
        """'Агат, открой хром' — имя исправлено"""
        result = normalize_stt("Агат, открой хром")
        self.assertNotIn("агат,", result.lower())

    # ── Обращение к ассистенту в начале ────────────────────────────────────

    def test_address_stripped_with_comma(self) -> None:
        """'Агата, открой хром' → 'открой хром'"""
        self.assertEqual(normalize_stt("Агата, открой хром"), "открой хром")

    def test_address_stripped_without_comma(self) -> None:
        """'Агата открой хром' → 'открой хром'"""
        self.assertEqual(normalize_stt("Агата открой хром"), "открой хром")

    def test_address_with_verb_normalization(self) -> None:
        """'Агата, открою хром' → 'открой хром'"""
        self.assertEqual(normalize_stt("Агата, открою хром"), "открой хром")

    # ── Вводные фразы ──────────────────────────────────────────────────────

    def test_mozhesh_otkryt_stripped(self) -> None:
        """'можешь открыть хром' → 'открой хром'"""
        self.assertEqual(normalize_stt("можешь открыть хром"), "открой хром")

    def test_otkryt_stripped(self) -> None:
        """'открыть хром' → 'открой хром'"""
        self.assertEqual(normalize_stt("открыть хром"), "открой хром")

    def test_zapustit_stripped(self) -> None:
        """'запустить хром' → 'запусти хром'"""
        self.assertEqual(normalize_stt("запустить хром"), "запусти хром")

    # ── Trailing punctuation ───────────────────────────────────────────────

    def test_trailing_dot_removed(self) -> None:
        """'открой хром.' → 'открой хром'"""
        self.assertEqual(normalize_stt("открой хром."), "открой хром")

    def test_trailing_exclamation_removed(self) -> None:
        """'открой хром!' → 'открой хром'"""
        self.assertEqual(normalize_stt("открой хром!"), "открой хром")

    # ── Без изменений ──────────────────────────────────────────────────────

    def test_regular_chat_unchanged(self) -> None:
        """Обычная фраза не должна изменяться существенно."""
        result = normalize_stt("Привет, как дела?")
        self.assertIn("привет", result.lower())

    def test_empty_string(self) -> None:
        self.assertEqual(normalize_stt(""), "")

    def test_whitespace_only(self) -> None:
        self.assertEqual(normalize_stt("   "), "   ")


class VoiceCommandRoutingTests(unittest.TestCase):
    """Проверяем, что normalized text правильно роутится."""

    def _route_voice(self, raw: str):
        """Normalize raw STT, then route."""
        return route_explicit_action(normalize_stt(raw))

    # ── Все варианты "открой хром" ─────────────────────────────────────────

    def test_otkroi_khrom(self) -> None:
        action = self._route_voice("открой хром")
        self.assertIsNotNone(action)
        assert action is not None
        self.assertEqual(action.tool_name, "open_app")
        self.assertEqual(action.kind, "OPEN_APP")

    def test_otkroyu_khrom(self) -> None:
        """'открою хром' после нормализации → OPEN_APP"""
        action = self._route_voice("открою хром")
        self.assertIsNotNone(action)
        assert action is not None
        self.assertEqual(action.tool_name, "open_app")

    def test_otkryvai_khrom(self) -> None:
        """'открывай хром' → OPEN_APP"""
        action = route_explicit_action("открывай хром")
        self.assertIsNotNone(action)
        assert action is not None
        self.assertEqual(action.tool_name, "open_app")

    def test_zapusti_khrom(self) -> None:
        action = self._route_voice("запусти хром")
        self.assertIsNotNone(action)
        assert action is not None
        self.assertEqual(action.tool_name, "open_app")

    def test_zapuskai_khrom(self) -> None:
        """'запускай хром' → OPEN_APP"""
        action = route_explicit_action("запускай хром")
        self.assertIsNotNone(action)
        assert action is not None
        self.assertEqual(action.tool_name, "open_app")

    def test_agata_otkroi_chrome(self) -> None:
        """'Агата открой Chrome' → OPEN_APP"""
        action = self._route_voice("Агата открой Chrome")
        self.assertIsNotNone(action)
        assert action is not None
        self.assertEqual(action.tool_name, "open_app")

    def test_agata_zapusti_chrome_with_comma(self) -> None:
        """'Агата, запусти Chrome' → OPEN_APP"""
        action = self._route_voice("Агата, запусти Chrome")
        self.assertIsNotNone(action)
        assert action is not None
        self.assertEqual(action.tool_name, "open_app")

    def test_otkroi_google_chrome(self) -> None:
        """'открой гугл хром' → OPEN_APP"""
        action = self._route_voice("открой гугл хром")
        self.assertIsNotNone(action)
        assert action is not None
        self.assertEqual(action.tool_name, "open_app")

    def test_mozhesh_otkryt_chrome(self) -> None:
        """'можешь открыть Chrome' → OPEN_APP"""
        action = self._route_voice("можешь открыть Chrome")
        self.assertIsNotNone(action)
        assert action is not None
        self.assertEqual(action.tool_name, "open_app")

    def test_otkroyu_khrom_with_dot(self) -> None:
        """'открою хром.' (Whisper trailing punct) → OPEN_APP"""
        action = self._route_voice("открою хром.")
        self.assertIsNotNone(action)
        assert action is not None
        self.assertEqual(action.tool_name, "open_app")

    # ── Вопрос не роутится как action ──────────────────────────────────────

    def test_question_how_to_open_is_not_action(self) -> None:
        """'Как открыть Chrome?' — НЕ должно стать OPEN_APP"""
        action = self._route_voice("Как открыть Chrome?")
        self.assertIsNone(action)

    def test_question_what_is_chrome(self) -> None:
        """'Что такое Chrome?' — не action"""
        action = self._route_voice("Что такое Chrome?")
        self.assertIsNone(action)

    def test_conversation_not_action(self) -> None:
        """Обычный разговор — не action"""
        action = self._route_voice("Привет, как дела?")
        self.assertIsNone(action)

    # ── Aliases: проверяем что роутинг работает ────────────────────────────

    def test_alias_khrom_to_open_app(self) -> None:
        action = route_explicit_action("открой хром")
        self.assertIsNotNone(action)
        assert action is not None
        self.assertEqual(action.tool_name, "open_app")

    def test_alias_chrome_to_open_app(self) -> None:
        action = route_explicit_action("открой Chrome")
        self.assertIsNotNone(action)
        assert action is not None
        self.assertEqual(action.tool_name, "open_app")

    def test_alias_gugol_khrom_to_open_app(self) -> None:
        action = route_explicit_action("открой гугл хром")
        self.assertIsNotNone(action)
        assert action is not None
        self.assertEqual(action.tool_name, "open_app")

    def test_alias_telega_to_open_app(self) -> None:
        action = route_explicit_action("открой телега")
        self.assertIsNotNone(action)
        assert action is not None
        self.assertEqual(action.tool_name, "open_app")

    def test_alias_telegramm_to_open_app(self) -> None:
        action = route_explicit_action("открой телеграм")
        self.assertIsNotNone(action)
        assert action is not None
        self.assertEqual(action.tool_name, "open_app")

    # ── Новые глаголы в action_router ──────────────────────────────────────

    def test_pokazhi_routes_to_open(self) -> None:
        """'покажи хром' → OPEN_APP"""
        action = route_explicit_action("покажи хром")
        self.assertIsNotNone(action)
        assert action is not None
        self.assertEqual(action.tool_name, "open_app")

    def test_zapuskai_routes_directly(self) -> None:
        """'запускай хром' → OPEN_APP (без нормализации)"""
        action = route_explicit_action("запускай хром")
        self.assertIsNotNone(action)
        assert action is not None
        self.assertEqual(action.tool_name, "open_app")


if __name__ == "__main__":
    unittest.main()

