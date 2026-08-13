"""
Синтез речи (Text-to-Speech) с очередью и абстракцией движка.

Архитектура:
  - BaseTTSEngine — абстрактный интерфейс для TTS-движка
  - PyTTSx3Engine — текущий fallback (Windows SAPI5)
  - TTSManager — потокобезопасная очередь с фоновым воркером

Модуль экспортирует те же функции, что и раньше:
  speak(), is_speaking(), get_last_tts_finish_time()
плюс новые:
  enqueue(), start(), stop(), wait_until_done()

stt.py продолжает вызывать tts.is_speaking() и
tts.get_last_tts_finish_time() без изменений.
"""

import logging
import re
import threading
import time
from abc import ABC, abstractmethod
from queue import Queue, Empty, Full
from typing import Optional

import config

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Markdown cleanup (перенесено из старого tts.py, используется движками)
# ─────────────────────────────────────────────────────────────────────────────

def _clean_for_speech(text: str) -> str:
    """Убрать markdown и элементы, неудобные для озвучивания."""
    cleaned = text

    # Кодовые блоки
    cleaned = re.sub(r"```.*?```", "", cleaned, flags=re.DOTALL)

    # Inline code
    cleaned = re.sub(r"`([^`]+)`", r"\1", cleaned)

    # Bold / italic
    cleaned = re.sub(r"\*\*(.+?)\*\*", r"\1", cleaned)
    cleaned = re.sub(r"__(.+?)__", r"\1", cleaned)
    cleaned = re.sub(r"\*(.+?)\*", r"\1", cleaned)
    cleaned = re.sub(r"_(.+?)_", r"\1", cleaned)

    # Markdown links
    cleaned = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", cleaned)

    # Заголовки
    cleaned = re.sub(r"^#{1,6}\s*", "", cleaned, flags=re.MULTILINE)

    # Strikethrough
    cleaned = re.sub(r"~~(.+?)~~", r"\1", cleaned)

    # Убираем лишние пробелы
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    return cleaned


# ─────────────────────────────────────────────────────────────────────────────
# Абстракция TTS-движка
# ─────────────────────────────────────────────────────────────────────────────

class BaseTTSEngine(ABC):
    """Абстрактный интерфейс TTS-движка.

    Для добавления нового движка (Piper, GPT-SoVITS и т.д.)
    достаточно реализовать этот интерфейс.
    """

    @abstractmethod
    def synthesize_and_play(self, text: str) -> None:
        """Синтезировать и воспроизвести текст. Блокирует до окончания."""

    def shutdown(self) -> None:
        """Освободить ресурсы движка (опционально)."""


class PyTTSx3Engine(BaseTTSEngine):
    """Fallback TTS-движок на базе pyttsx3 (Windows SAPI5).

    ВАЖНО: pyttsx3 использует COM на Windows, поэтому движок
    должен быть инициализирован и использоваться в одном потоке.
    TTSManager гарантирует это — все вызовы идут из воркер-потока.
    """

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


# ─────────────────────────────────────────────────────────────────────────────
# TTSManager — очередь + фоновый воркер
# ─────────────────────────────────────────────────────────────────────────────

# Sentinel для остановки воркера
_POISON_PILL = None


class TTSManager:
    """Потокобезопасный менеджер TTS с фоновым воркером.

    Предложения помещаются в очередь через enqueue().
    Фоновый поток берёт их по одному и воспроизводит через движок.

    LLM может продолжать генерацию, пока TTS озвучивает предыдущее предложение.
    """

    def __init__(self, engine: BaseTTSEngine | None = None) -> None:
        self._engine: BaseTTSEngine = engine or PyTTSx3Engine()
        self._queue: Queue[str | None] = Queue(maxsize=config.TTS_QUEUE_MAX_SIZE)
        self._worker: threading.Thread | None = None
        self._lock = threading.Lock()
        self._is_speaking: bool = False
        self._last_finish_time: float = 0.0
        self._started: bool = False

    def start(self) -> None:
        """Запустить фоновый воркер-поток."""
        if self._started:
            return
        self._worker = threading.Thread(
            target=self._worker_loop,
            name="tts-worker",
            daemon=True,
        )
        self._worker.start()
        self._started = True
        logger.info("[TTS] Worker запущен")

    def stop(self) -> None:
        """Остановить воркер (дождаться завершения текущей фразы)."""
        if not self._started:
            return

        # Отправляем «яд» — воркер выйдет из цикла
        try:
            self._queue.put(_POISON_PILL, timeout=5.0)
        except Full:
            logger.warning("[TTS] Очередь переполнена при остановке, принудительно завершаю")

        if self._worker is not None:
            self._worker.join(timeout=10.0)

        self._engine.shutdown()
        self._started = False
        logger.info("[TTS] Worker остановлен")

    def enqueue(self, text: str) -> None:
        """Добавить предложение в очередь на озвучивание."""
        if not config.TTS_ENABLED:
            return
        if not text or not text.strip():
            return

        # Автозапуск при первом использовании
        if not self._started:
            self.start()

        try:
            self._queue.put(text, timeout=5.0)
            logger.info(
                "[TTS QUEUE] enqueued (%d in queue): '%s'",
                self._queue.qsize(),
                text[:60],
            )
        except Full:
            logger.warning("[TTS QUEUE] Очередь переполнена, пропускаю: '%s'", text[:60])

    def speak(self, text: str) -> None:
        """Совместимость: поставить текст в очередь и дождаться окончания.

        Для вызова из main.py, когда нужен синхронный режим
        (например, для коротких фраз от action router).
        """
        if not config.TTS_ENABLED:
            return
        if not text or not text.strip():
            logger.warning("Пустой текст для TTS — пропускаю")
            return

        # Очистка markdown для прямого вызова speak()
        clean_text = _clean_for_speech(text)
        if not clean_text:
            return

        self.enqueue(clean_text)
        self.wait_until_done()

    def is_speaking(self) -> bool:
        """Проверить, воспроизводит ли TTS текст в данный момент."""
        with self._lock:
            return self._is_speaking or not self._queue.empty()

    def get_last_tts_finish_time(self) -> float:
        """Получить unix timestamp окончания последнего TTS-воспроизведения."""
        with self._lock:
            return self._last_finish_time

    def wait_until_done(self) -> None:
        """Блокировать до тех пор, пока очередь не опустеет и речь не закончится."""
        # Ждём, пока очередь опустеет и текущая фраза доиграет
        while True:
            with self._lock:
                if not self._is_speaking and self._queue.empty():
                    return
            time.sleep(0.05)

    def _worker_loop(self) -> None:
        """Фоновый цикл воркера — берёт предложения из очереди и озвучивает."""
        logger.info("[TTS] Worker loop started")
        while True:
            try:
                text = self._queue.get(timeout=1.0)
            except Empty:
                continue

            # Poison pill — завершаем
            if text is None:
                self._queue.task_done()
                break

            with self._lock:
                self._is_speaking = True

            try:
                logger.info("[TTS] speaking: '%s'", text[:80])
                self._engine.synthesize_and_play(text)
                logger.info("[TTS] finished: '%s'", text[:80])
            except Exception as exc:
                logger.error("[TTS] Ошибка синтеза: %s", exc)
                print(f"[TTS ошибка: {exc}]")
            finally:
                with self._lock:
                    self._is_speaking = False
                    self._last_finish_time = time.time()
                self._queue.task_done()

        logger.info("[TTS] Worker loop ended")


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