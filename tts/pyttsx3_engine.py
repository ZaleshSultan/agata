"""
Fallback TTS-движок на базе pyttsx3 (Windows SAPI5).

ВАЖНО: pyttsx3 использует COM на Windows, поэтому движок
должен быть инициализирован и использоваться в одном потоке.
TTSManager гарантирует это — все вызовы идут из воркер-потока.
"""

import logging

import config
from tts.base import BaseTTSEngine

logger = logging.getLogger(__name__)


class PyTTSx3Engine(BaseTTSEngine):
    """Fallback TTS-движок на базе pyttsx3 (Windows SAPI5)."""

    def __init__(self) -> None:
        self._engine = None

    def _ensure_engine(self) -> None:
        """Ленивая инициализация в воркер-потоке."""
        if self._engine is not None:
            return

        import pyttsx3  # type: ignore[import-untyped]

        self._engine = pyttsx3.init()
        self._engine.setProperty("rate", config.TTS_RATE)

        # Попробовать найти русский голос
        voices = self._engine.getProperty("voices")
        for voice in voices:
            name_lower = voice.name.lower()
            if "russian" in name_lower or "ru" in name_lower or "irina" in name_lower:
                self._engine.setProperty("voice", voice.id)
                logger.info("TTS голос: %s", voice.name)
                break
        else:
            if voices:
                logger.warning(
                    "Русский голос не найден, используется: %s. "
                    "Установи русский голос Windows для лучшего качества.",
                    voices[0].name,
                )

        logger.info("TTS (pyttsx3) инициализирован, скорость: %d", config.TTS_RATE)

    def synthesize_and_play(self, text: str) -> None:
        """Синтез и воспроизведение через pyttsx3."""
        self._ensure_engine()
        assert self._engine is not None
        self._engine.say(text)
        self._engine.runAndWait()

    def shutdown(self) -> None:
        """Остановить pyttsx3 движок."""
        if self._engine is not None:
            try:
                self._engine.stop()
            except Exception:
                pass
            self._engine = None
