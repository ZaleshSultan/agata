"""
Тесты для XTTS backend, fallback-механизма и фабрики движков.

Все XTTS операции замокированы — тесты не требуют наличия модели.
"""

import time
import unittest
from unittest.mock import patch, MagicMock

from tts.base import BaseTTSEngine
from tts.manager import TTSManager, FallbackEngine, _create_default_engine


# ─────────────────────────────────────────────────────────────────────────────
# Фейковые движки для тестов
# ─────────────────────────────────────────────────────────────────────────────

class FakeTTSEngine(BaseTTSEngine):
    """Фейковый TTS-движок для тестирования."""

    def __init__(self, delay: float = 0.02) -> None:
        self.spoken: list[str] = []
        self.delay = delay
        self.is_shut_down = False

    def synthesize_and_play(self, text: str) -> None:
        self.spoken.append(text)
        if self.delay > 0:
            time.sleep(self.delay)

    def shutdown(self) -> None:
        self.is_shut_down = True


class FailingEngine(BaseTTSEngine):
    """Движок, который всегда падает."""

    def __init__(self, error_msg: str = "XTTS synthesis failed") -> None:
        self.error_msg = error_msg
        self.call_count = 0

    def synthesize_and_play(self, text: str) -> None:
        self.call_count += 1
        raise RuntimeError(self.error_msg)

    def shutdown(self) -> None:
        pass


class FailOnceEngine(BaseTTSEngine):
    """Движок, который падает один раз, потом работает."""

    def __init__(self) -> None:
        self.spoken: list[str] = []
        self.call_count = 0

    def synthesize_and_play(self, text: str) -> None:
        self.call_count += 1
        if self.call_count == 1:
            raise RuntimeError("Transient error")
        self.spoken.append(text)

    def shutdown(self) -> None:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Тесты FallbackEngine
# ─────────────────────────────────────────────────────────────────────────────

class TestFallbackEngine(unittest.TestCase):

    def test_primary_success(self) -> None:
        """При успехе primary — fallback не вызывается."""
        primary = FakeTTSEngine()
        fallback = FakeTTSEngine()
        engine = FallbackEngine(primary, fallback)

        engine.synthesize_and_play("Привет")

        self.assertEqual(primary.spoken, ["Привет"])
        self.assertEqual(fallback.spoken, [])

    def test_primary_fail_fallback_used(self) -> None:
        """При ошибке primary — используется fallback."""
        primary = FailingEngine()
        fallback = FakeTTSEngine()
        engine = FallbackEngine(primary, fallback)

        engine.synthesize_and_play("Привет")

        self.assertEqual(primary.call_count, 1)
        self.assertEqual(fallback.spoken, ["Привет"])

    def test_transient_error_retries_primary(self) -> None:
        """После transient ошибки primary используется снова."""
        primary = FailOnceEngine()
        fallback = FakeTTSEngine()
        engine = FallbackEngine(primary, fallback)

        # Первый вызов — primary падает, fallback работает
        engine.synthesize_and_play("Первый")
        self.assertEqual(fallback.spoken, ["Первый"])

        # Второй вызов — primary работает
        engine.synthesize_and_play("Второй")
        self.assertEqual(primary.spoken, ["Второй"])

    def test_permanent_failure_disables_primary(self) -> None:
        """При перманентной ошибке primary отключается навсегда."""
        primary = FailingEngine("coqui-tts не установлен: ImportError")
        fallback = FakeTTSEngine()
        engine = FallbackEngine(primary, fallback)

        engine.synthesize_and_play("Первый")
        engine.synthesize_and_play("Второй")

        # Primary вызван только 1 раз (перманентная ошибка)
        self.assertEqual(primary.call_count, 1)
        self.assertEqual(fallback.spoken, ["Первый", "Второй"])

    def test_both_fail_no_crash(self) -> None:
        """Если оба движка падают — не crash, а silent failure."""
        primary = FailingEngine()
        fallback = FailingEngine("pyttsx3 тоже упал")
        engine = FallbackEngine(primary, fallback)

        # Не должно бросить исключение
        engine.synthesize_and_play("Тест")

    def test_shutdown_calls_both(self) -> None:
        """shutdown() вызывается для обоих движков."""
        primary = FakeTTSEngine()
        fallback = FakeTTSEngine()
        engine = FallbackEngine(primary, fallback)

        engine.shutdown()

        self.assertTrue(primary.is_shut_down)
        self.assertTrue(fallback.is_shut_down)


# ─────────────────────────────────────────────────────────────────────────────
# Тесты TTSManager с фейковым движком
# ─────────────────────────────────────────────────────────────────────────────

class TestTTSManagerWithFake(unittest.TestCase):

    def setUp(self) -> None:
        self.engine = FakeTTSEngine(delay=0.02)
        self.manager = TTSManager(engine=self.engine)
        self.manager.start()

    def tearDown(self) -> None:
        self.manager.stop()

    def test_enqueue_plays_in_order(self) -> None:
        """Предложения из очереди озвучиваются строго по порядку."""
        sentences = ["Первое.", "Второе.", "Третье."]
        for s in sentences:
            self.manager.enqueue(s)
        self.manager.wait_until_done()
        self.assertEqual(self.engine.spoken, sentences)

    def test_speak_blocks_until_done(self) -> None:
        """speak() ставит текст в очередь и блокирует."""
        self.manager.speak("Синхронная фраза.")
        self.assertIn("Синхронная фраза.", self.engine.spoken)
        self.assertFalse(self.manager.is_speaking())

    def test_is_speaking_state(self) -> None:
        """is_speaking() корректно отражает состояние."""
        self.assertFalse(self.manager.is_speaking())
        self.manager.enqueue("Длинное тестовое предложение для проверки.")
        time.sleep(0.01)
        self.assertTrue(self.manager.is_speaking())
        self.manager.wait_until_done()
        self.assertFalse(self.manager.is_speaking())
        self.assertGreater(self.manager.get_last_tts_finish_time(), 0.0)

    def test_empty_text_ignored(self) -> None:
        """Пустая строка не попадает в очередь."""
        self.manager.enqueue("")
        self.manager.enqueue("  ")
        self.manager.enqueue("Нормальный текст.")
        self.manager.wait_until_done()
        self.assertEqual(self.engine.spoken, ["Нормальный текст."])

    def test_multiple_messages_queue(self) -> None:
        """Несколько последовательных сообщений обрабатываются корректно."""
        for i in range(10):
            self.manager.enqueue(f"Сообщение номер {i}.")
        self.manager.wait_until_done()
        self.assertEqual(len(self.engine.spoken), 10)

    def test_engine_error_resilience(self) -> None:
        """Ошибка синтеза не убивает воркер."""
        failing = FailingEngine()
        fallback = FakeTTSEngine()
        wrapped = FallbackEngine(failing, fallback)
        manager = TTSManager(engine=wrapped)
        manager.start()

        manager.enqueue("Тест ошибки")
        manager.enqueue("После ошибки")
        manager.wait_until_done()

        self.assertIn("После ошибки", fallback.spoken)
        manager.stop()


# ─────────────────────────────────────────────────────────────────────────────
# Тесты фабрики движков
# ─────────────────────────────────────────────────────────────────────────────

class TestEngineFactory(unittest.TestCase):

    @patch("config.TTS_ENGINE", "pyttsx3")
    def test_pyttsx3_selected(self) -> None:
        """TTS_ENGINE=pyttsx3 → создаётся PyTTSx3Engine."""
        from tts.pyttsx3_engine import PyTTSx3Engine
        engine = _create_default_engine()
        self.assertIsInstance(engine, PyTTSx3Engine)

    @patch("config.TTS_ENGINE", "xtts")
    def test_xtts_selected_creates_fallback(self) -> None:
        """TTS_ENGINE=xtts → создаётся FallbackEngine."""
        engine = _create_default_engine()
        self.assertIsInstance(engine, FallbackEngine)

    @patch("config.TTS_ENGINE", "nonexistent")
    def test_unknown_engine_falls_back(self) -> None:
        """Неизвестный движок → pyttsx3."""
        from tts.pyttsx3_engine import PyTTSx3Engine
        engine = _create_default_engine()
        self.assertIsInstance(engine, PyTTSx3Engine)


# ─────────────────────────────────────────────────────────────────────────────
# Тесты валидации reference audio
# ─────────────────────────────────────────────────────────────────────────────

class TestReferenceAudioValidation(unittest.TestCase):

    def test_missing_file(self) -> None:
        """Отсутствующий файл — невалидно."""
        from tts.xtts_engine import _validate_reference_audio
        is_valid, detail = _validate_reference_audio("/nonexistent/path.wav")
        self.assertFalse(is_valid)
        self.assertIn("not found", detail.lower())

    def test_empty_path(self) -> None:
        """Пустой путь — невалидно."""
        from tts.xtts_engine import _validate_reference_audio
        is_valid, detail = _validate_reference_audio("")
        self.assertFalse(is_valid)


# ─────────────────────────────────────────────────────────────────────────────
# Тесты определения устройства
# ─────────────────────────────────────────────────────────────────────────────

class TestDeviceDetection(unittest.TestCase):

    def test_cpu_setting(self) -> None:
        """device_setting='cpu' → всегда 'cpu'."""
        from tts.xtts_engine import _resolve_device
        result = _resolve_device("cpu")
        self.assertEqual(result, "cpu")

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.get_device_name", return_value="RTX 4050")
    def test_auto_with_cuda(self, mock_name: MagicMock, mock_avail: MagicMock) -> None:
        """auto + CUDA available → 'cuda'."""
        from tts.xtts_engine import _resolve_device
        result = _resolve_device("auto")
        self.assertEqual(result, "cuda")

    @patch("torch.cuda.is_available", return_value=False)
    def test_auto_without_cuda(self, mock_avail: MagicMock) -> None:
        """auto + no CUDA → 'cpu'."""
        from tts.xtts_engine import _resolve_device
        result = _resolve_device("auto")
        self.assertEqual(result, "cpu")


# ─────────────────────────────────────────────────────────────────────────────
# Тесты API совместимости (модульный уровень)
# ─────────────────────────────────────────────────────────────────────────────

class TestModuleLevelAPI(unittest.TestCase):

    def test_all_public_functions_exist(self) -> None:
        """Все публичные функции из старого tts.py доступны."""
        import tts

        self.assertTrue(callable(tts.speak))
        self.assertTrue(callable(tts.enqueue))
        self.assertTrue(callable(tts.stop))
        self.assertTrue(callable(tts.start))
        self.assertTrue(callable(tts.is_speaking))
        self.assertTrue(callable(tts.get_last_tts_finish_time))
        self.assertTrue(callable(tts.wait_until_done))

    def test_classes_importable(self) -> None:
        """Классы импортируются из нового пакета."""
        from tts import BaseTTSEngine, TTSManager, FallbackEngine
        self.assertTrue(BaseTTSEngine is not None)
        self.assertTrue(TTSManager is not None)
        self.assertTrue(FallbackEngine is not None)


if __name__ == "__main__":
    unittest.main()
