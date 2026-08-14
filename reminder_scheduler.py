"""
Фоновый планировщик напоминаний.

Раз в REMINDER_CHECK_INTERVAL_SECONDS проверяет просроченные напоминания
и озвучивает их через TTS. Работает в daemon-потоке, не блокирует
основной голосовой цикл.
"""

import logging
import threading
import time
from datetime import datetime

import config
from tools import reminder as reminder_tool

logger = logging.getLogger(__name__)


class ReminderScheduler:
    def __init__(self, tts_speak_callback):
        """
        Args:
            tts_speak_callback: функция speak(text: str) для озвучки —
                передавай существующий метод TTS-менеджера, не создавай
                новый TTS-движок здесь.
        """
        self._speak = tts_speak_callback
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        logger.info("[REMINDER SCHEDULER] запущен, интервал=%ds", config.REMINDER_CHECK_INTERVAL_SECONDS)

    def stop(self) -> None:
        self._stop_event.set()

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._check_due_reminders()
            except Exception as exc:
                logger.warning("[REMINDER SCHEDULER] ошибка проверки: %s", exc)
            self._stop_event.wait(config.REMINDER_CHECK_INTERVAL_SECONDS)

    def _check_due_reminders(self) -> None:
        due = reminder_tool.get_due_reminders()
        for r in due:
            text = f"Напоминаю: {r['title']}"
            logger.info("[REMINDER SCHEDULER] сработало: %s", r["title"])
            self._speak(text)
            reminder_tool.mark_reminder_done(r["id"])
