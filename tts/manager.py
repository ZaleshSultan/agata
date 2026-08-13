"""
TTSManager — потокобезопасная очередь TTS с фоновым воркером.

Также содержит FallbackEngine — обёртка, которая при ошибке
основного движка автоматически переключается на fallback.
"""

import logging
import re
import threading
import time
from queue import Queue, Empty, Full
from typing import Optional

import config
from tts.base import BaseTTSEngine

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Markdown cleanup (используется движками и speak())
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
# FallbackEngine — обёртка с автоматическим fallback
# ─────────────────────────────────────────────────────────────────────────────

class FallbackEngine(BaseTTSEngine):
    """Обёртка: при ошибке primary движка переключается на fallback.

    Fallback срабатывает при любой ошибке synthesize_and_play():
    - отсутствие модели
    - CUDA error
    - synthesis error
    - playback error
    - и т.д.
    """

    def __init__(self, primary: BaseTTSEngine, fallback: BaseTTSEngine) -> None:
        self._primary = primary
        self._fallback = fallback
        self._primary_failed_permanently: bool = False

    def synthesize_and_play(self, text: str) -> None:
        """Попробовать primary, при ошибке — fallback."""
        if not self._primary_failed_permanently:
            try:
                self._primary.synthesize_and_play(text)
                return
            except Exception as exc:
                logger.error("[TTS] XTTS failed: %s", exc)
                logger.info("[TTS] fallback -> pyttsx3")
                # Если ошибка инициализации — помечаем как permanent
                err_msg = str(exc)
                if any(kw in err_msg for kw in (
                    "не установлен", "not found", "невалиден",
                    "Ошибка загрузки", "ImportError",
                )):
                    self._primary_failed_permanently = True
                    logger.warning(
                        "[TTS] Primary engine отключён навсегда: %s", exc
                    )

        # Fallback
        try:
            self._fallback.synthesize_and_play(text)
        except Exception as fallback_exc:
            logger.error(
                "[TTS] Fallback тоже упал: %s", fallback_exc
            )

    def shutdown(self) -> None:
        """Остановить оба движка."""
        try:
            self._primary.shutdown()
        except Exception:
            pass
        try:
            self._fallback.shutdown()
        except Exception:
            pass


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
        self._engine: BaseTTSEngine = engine or _create_default_engine()
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
                request_start = time.perf_counter()
                logger.info("[TTS] speaking: '%s'", text[:80])
                self._engine.synthesize_and_play(text)
                total_time = time.perf_counter() - request_start
                logger.info("[TTS] finished: '%s' (%.3fs)", text[:80], total_time)
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
# Фабрика движков
# ─────────────────────────────────────────────────────────────────────────────

def _create_default_engine() -> BaseTTSEngine:
    """Создать движок на основе config.TTS_ENGINE.

    Если выбран XTTS — оборачиваем в FallbackEngine с pyttsx3 fallback.
    Если XTTS недоступен при создании — сразу fallback.
    """
    from tts.pyttsx3_engine import PyTTSx3Engine

    engine_name = getattr(config, "TTS_ENGINE", "pyttsx3")

    if engine_name == "pyttsx3":
        logger.info("[TTS] backend=pyttsx3")
        return PyTTSx3Engine()

    if engine_name == "xtts":
        try:
            from tts.xtts_engine import XTTSEngine
            primary = XTTSEngine()
            fallback = PyTTSx3Engine()
            logger.info("[TTS] backend=xtts (с fallback на pyttsx3)")
            return FallbackEngine(primary, fallback)
        except Exception as exc:
            logger.error("[TTS] Не удалось создать XTTS движок: %s", exc)
            logger.info("[TTS] fallback -> pyttsx3")
            return PyTTSx3Engine()

    logger.warning("[TTS] Неизвестный движок '%s', используется pyttsx3", engine_name)
    return PyTTSx3Engine()
