"""
Синтез речи (Text-to-Speech).

Основной движок: pyttsx3 (работает на Windows из коробки).
Архитектура позволяет подключить piper-tts в будущем.

Побочные эффекты: воспроизведение аудио через системный аудиовыход.
"""

import logging
from typing import Optional

import numpy as np  # обязательный импорт по ТЗ

import config

logger = logging.getLogger(__name__)

# Ленивая инициализация TTS-движка
_engine = None


def _get_engine():
    """Инициализировать pyttsx3-движок при первом вызове."""
    global _engine
    if _engine is None:
        import pyttsx3  # type: ignore[import-untyped]
        _engine = pyttsx3.init()
        _engine.setProperty("rate", config.TTS_RATE)

        # Попробовать найти русский голос
        voices = _engine.getProperty("voices")
        for voice in voices:
            # На Windows русские голоса обычно содержат "Russian" или "RU" в имени
            name_lower = voice.name.lower()
            if "russian" in name_lower or "ru" in name_lower or "irina" in name_lower:
                _engine.setProperty("voice", voice.id)
                logger.info("TTS голос: %s", voice.name)
                break
        else:
            # Если русского голоса нет — используем дефолтный
            if voices:
                logger.warning(
                    "Русский голос не найден, используется: %s. "
                    "Установи русский голос Windows для лучшего качества.",
                    voices[0].name,
                )

        logger.info("TTS (pyttsx3) инициализирован, скорость: %d", config.TTS_RATE)
    return _engine


def speak(text: str) -> None:
    """
    Озвучить текст через TTS.

    Args:
        text: Текст для озвучивания.
    """
    if not text or not text.strip():
        logger.warning("Пустой текст для TTS — пропускаю")
        return

    # Убираем markdown-разметку, которая плохо звучит
    clean_text = _clean_for_speech(text)

    try:
        engine = _get_engine()
        engine.say(clean_text)
        engine.runAndWait()
        logger.info("Озвучено: '%s'", clean_text[:80])
    except Exception as exc:
        logger.error("Ошибка TTS: %s", exc)
        # Не роняем программу — просто не озвучиваем
        print(f"[TTS ошибка: {exc}]")


def _clean_for_speech(text: str) -> str:
    """Убрать из текста элементы, неудобные для озвучивания."""
    import re
    # Убираем markdown: **, *, `, #, [](), ~~
    cleaned = text
    cleaned = re.sub(r"\*\*(.+?)\*\*", r"\1", cleaned)  # **bold**
    cleaned = re.sub(r"\*(.+?)\*", r"\1", cleaned)      # *italic*
    cleaned = re.sub(r"`(.+?)`", r"\1", cleaned)         # `code`
    cleaned = re.sub(r"#{1,6}\s*", "", cleaned)           # # headers
    cleaned = re.sub(r"\[(.+?)\]\(.+?\)", r"\1", cleaned) # [link](url)
    cleaned = re.sub(r"~~(.+?)~~", r"\1", cleaned)        # ~~strikethrough~~
    # Убираем множественные пробелы
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned
