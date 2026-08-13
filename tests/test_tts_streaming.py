"""
Тесты для очереди TTS и абстрактного слоя движков (TTSManager).
Используют FakeTTSEngine без вызова реального системного звука.
"""

import time
import unittest

from tts import BaseTTSEngine, TTSManager


class FakeTTSEngine(BaseTTSEngine):
    """Фейковый TTS-движок для изолированного тестирования очереди."""

    def __init__(self, delay_per_call: float = 0.05) -> None:
        self.spoken_texts: list[str] = []
        self.delay_per_call: float = delay_per_call
        self.is_shutdown: bool = False

    def synthesize_and_play(self, text: str) -> None:
        if text.startswith("FAIL"):
            raise RuntimeError("Симулированная ошибка TTS")
        self.spoken_texts.append(text)
        if self.delay_per_call > 0:
            time.sleep(self.delay_per_call)

    def shutdown(self) -> None:
        self.is_shutdown = True


class TestTTSStreaming(unittest.TestCase):

    def setUp(self) -> None:
        self.engine = FakeTTSEngine(delay_per_call=0.02)
        self.manager = TTSManager(engine=self.engine)
        self.manager.start()

    def tearDown(self) -> None:
        self.manager.stop()

    def test_enqueue_and_sequential_playback(self) -> None:
        """Предложения из очереди озвучиваются строго по порядку."""
        sentences = ["Первое предложение.", "Второе предложение.", "Третье предложение."]
        for s in sentences:
            self.manager.enqueue(s)

        self.manager.wait_until_done()

        self.assertEqual(self.engine.spoken_texts, sentences)

    def test_is_speaking_state(self) -> None:
        """is_speaking() возвращает True во время озвучивания и False после."""
        self.assertFalse(self.manager.is_speaking())

        self.manager.enqueue("Длинное тестовое предложение для проверки флага.")
        # Небольшая пауза чтобы воркер подхватил задачу
        time.sleep(0.01)
        self.assertTrue(self.manager.is_speaking())

        self.manager.wait_until_done()
        self.assertFalse(self.manager.is_speaking())
        self.assertGreater(self.manager.get_last_tts_finish_time(), 0.0)

    def test_engine_error_resilience(self) -> None:
        """Ошибка синтеза не роняет воркер, следующие фразу продолжают озвучиваться."""
        self.manager.enqueue("Первое предложение.")
        self.manager.enqueue("FAIL ошибка здесь")
        self.manager.enqueue("Второе предложение.")

        self.manager.wait_until_done()

        self.assertIn("Первое предложение.", self.engine.spoken_texts)
        self.assertIn("Второе предложение.", self.engine.spoken_texts)
        self.assertNotIn("FAIL ошибка здесь", self.engine.spoken_texts)

    def test_speak_backward_compatibility(self) -> None:
        """Метод speak() ставит фразу в очередь и блокирует до её завершения."""
        self.manager.speak("Синхронная фраза.")
        self.assertIn("Синхронная фраза.", self.engine.spoken_texts)
        self.assertFalse(self.manager.is_speaking())

    def test_clean_shutdown(self) -> None:
        """stop() корректно останавливает воркер и вызывает shutdown() у движка."""
        self.manager.stop()
        self.assertTrue(self.engine.is_shutdown)


if __name__ == "__main__":
    unittest.main()
