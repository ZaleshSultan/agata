"""
Синтез речи (Text-to-Speech) с очередью и абстракцией движка.

Архитектура:
  - BaseTTSEngine — абстрактный интерфейс для TTS-движка
  - PyTTSx3Engine — fallback (Windows SAPI5)
  - XTTSEngine — primary (XTTS-v2 voice cloning)
  - FallbackEngine — обёртка с автоматическим fallback
  - TTSManager — потокобезопасная очередь с фоновым воркером

Модуль экспортирует те же функции, что и раньше:
  speak(), is_speaking(), get_last_tts_finish_time()
плюс:
  enqueue(), start(), stop(), wait_until_done()

stt.py продолжает вызывать tts.is_speaking() и
tts.get_last_tts_finish_time() без изменений.
"""

from tts.base import BaseTTSEngine
from tts.manager import TTSManager, FallbackEngine, _clean_for_speech

import logging

logger = logging.getLogger(__name__)

__all__ = [
    "BaseTTSEngine",
    "TTSManager",
    "FallbackEngine",
    "start",
    "stop",
    "enqueue",
    "speak",
    "speak_confirmation",
    "is_speaking",
    "get_last_tts_finish_time",
    "wait_until_done",
]

# ─────────────────────────────────────────────────────────────────────────────
# Модульный API (обратная совместимость с stt.py и main.py)
# ─────────────────────────────────────────────────────────────────────────────

_manager: TTSManager | None = None


def _get_manager() -> TTSManager:
    """Получить или создать глобальный TTSManager."""
    global _manager
    if _manager is None:
        _manager = TTSManager()
    return _manager


def start() -> None:
    """Запустить фоновый TTS-воркер."""
    _get_manager().start()


def stop() -> None:
    """Остановить TTS-воркер."""
    if _manager is not None:
        _manager.stop()


def enqueue(text: str) -> None:
    """Добавить предложение в очередь TTS."""
    _get_manager().enqueue(text)


def speak(text: str) -> None:
    """Озвучить текст через TTS (совместимость: очередь + ожидание)."""
    _get_manager().speak(text)


def speak_confirmation(text: str) -> None:
    """Воспроизвести короткую фразу подтверждения с высоким приоритетом."""
    _get_manager().speak_confirmation(text)


def is_speaking() -> bool:
    """Проверить, воспроизводит ли TTS текст в данный момент."""
    mgr = _manager
    if mgr is None:
        return False
    return mgr.is_speaking()


def get_last_tts_finish_time() -> float:
    """Получить unix timestamp окончания последнего TTS-воспроизведения."""
    mgr = _manager
    if mgr is None:
        return 0.0
    return mgr.get_last_tts_finish_time()


def wait_until_done() -> None:
    """Дождаться завершения всех предложений в очереди."""
    mgr = _manager
    if mgr is not None:
        mgr.wait_until_done()
