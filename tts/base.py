"""
Абстрактный интерфейс TTS-движка.

Все движки (pyttsx3, XTTS, будущие) реализуют BaseTTSEngine.
TTSManager вызывает synthesize_and_play() из воркер-потока.
"""

from abc import ABC, abstractmethod


class BaseTTSEngine(ABC):
    """Абстрактный интерфейс TTS-движка.

    Для добавления нового движка достаточно реализовать этот интерфейс.
    """

    @abstractmethod
    def synthesize_and_play(self, text: str) -> None:
        """Синтезировать и воспроизвести текст. Блокирует до окончания."""

    def shutdown(self) -> None:
        """Освободить ресурсы движка (опционально)."""
