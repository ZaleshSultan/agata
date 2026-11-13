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


from tts_text_normalizer import normalize_text_for_tts

def _clean_for_speech(text: str) -> str:
    """Убрать markdown, формулы, LaTeX и элементы, неудобные для озвучивания."""
    return normalize_text_for_tts(text)


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
        self._model_ready: bool = False

    @property
    def model_ready(self) -> bool:
        return self._model_ready

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
        """Добавить предложение в очередь на озвучивание (асинхронно, не блокирует вызов)."""
        if not config.TTS_ENABLED:
            return
        if not text or not text.strip():
            return

        clean_text = _clean_for_speech(text)
        if not clean_text:
            return

        # Автозапуск при первом использовании
        if not self._started:
            self.start()

        try:
            # Store tuple of (text, enqueue_timestamp)
            item = (clean_text, time.perf_counter())
            self._queue.put(item, timeout=5.0)
            logger.info(
                "[TTS QUEUE] enqueued (%d in queue): '%s'",
                self._queue.qsize(),
                clean_text[:60],
            )
        except Full:
            logger.warning("[TTS QUEUE] Очередь переполнена, пропускаю: '%s'", clean_text[:60])

    def speak(self, text: str) -> None:
        """Совместимость: поставить текст в очередь и дождаться окончания."""
        if not config.TTS_ENABLED:
            return
        if not text or not text.strip():
            logger.warning("Пустой текст для TTS — пропускаю")
            return

        clean_text = _clean_for_speech(text)
        if not clean_text:
            return

        self.enqueue(clean_text)
        self.wait_until_done()

    def speak_confirmation(self, text: str) -> None:
        """Воспроизвести короткую фразу подтверждения с высоким приоритетом и минимальной задержкой."""
        if not config.TTS_ENABLED:
            return
        if not text or not text.strip():
            return
        clean_text = _clean_for_speech(text)
        if not clean_text:
            return

        logger.info("[TTS CONFIRM] Fast confirmation speech: '%s'", clean_text)
        self.enqueue(clean_text)

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
        while True:
            with self._lock:
                if not self._is_speaking and self._queue.empty():
                    return
            time.sleep(0.05)

    def _worker_loop(self) -> None:
        """Фоновый цикл воркера — берёт предложения из очереди и озвучивает."""
        logger.info("[TTS] Worker loop started")

        # Pre-warm model once on startup
        try:
            target_engine = getattr(self._engine, "_primary", self._engine)
            if hasattr(target_engine, "warmup"):
                target_engine.warmup()
            self._model_ready = True
            logger.info("[TTS] Model pre-warmed, model_ready=True")
        except Exception as w_exc:
            logger.debug("[TTS] Pre-warm note: %s", w_exc)
            self._model_ready = True

        while True:
            try:
                item = self._queue.get(timeout=1.0)
            except Empty:
                continue

            # Poison pill — завершаем
            if item is None or item is _POISON_PILL:
                self._queue.task_done()
                break

            if isinstance(item, tuple):
                text, enqueued_at = item
            else:
                text, enqueued_at = item, time.perf_counter()

            with self._lock:
                self._is_speaking = True

            queue_wait_ms = (time.perf_counter() - enqueued_at) * 1000.0
            try:
                request_start = time.perf_counter()
                logger.info("[TTS] speaking: '%s'", text[:80])
                self._engine.synthesize_and_play(text)
                synth_play_ms = (time.perf_counter() - request_start) * 1000.0
                logger.info(
                    "[PERF] tts_queue_latency_ms=%.1fms tts_playback_latency_ms=%.1fms phrase='%s'",
                    queue_wait_ms,
                    synth_play_ms,
                    text[:50],
                )
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
