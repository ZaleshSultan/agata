"""
XTTS-v2 движок с zero-shot voice cloning.

Использует coqui-tts (idiap/coqui-ai-TTS fork) для синтеза речи
с клонированием голоса по reference audio.

Ключевые принципы:
  - Модель загружается ОДИН РАЗ при первом synthesize_and_play()
  - Speaker conditioning кэшируется после первого вычисления
  - CUDA используется при доступности, иначе fallback на CPU
  - Все ошибки перехватываются — Агата не падает из-за TTS
  - Reference audio валидируется перед использованием
"""

import logging
import os
import tempfile
import time
import wave
from pathlib import Path
from typing import Optional

import config
from tts.base import BaseTTSEngine

logger = logging.getLogger(__name__)


def _resolve_device(device_setting: str) -> str:
    """Определить устройство: cuda или cpu через runtime_capabilities."""
    from runtime_capabilities import resolve_tts_device
    return resolve_tts_device(device_setting)


def _validate_reference_audio(path: str) -> tuple[bool, str]:
    """Валидация reference audio файла.

    Проверяет: существование, читаемость, наличие аудиодорожки,
    sample rate, длительность, mono/stereo, непустой файл.

    Returns:
        (is_valid, detail_message)
    """
    audio_path = Path(path)

    if not audio_path.exists():
        return False, f"File not found: {path}"

    if not audio_path.is_file():
        return False, f"Not a file: {path}"

    if audio_path.stat().st_size == 0:
        return False, f"Empty file: {path}"

    if audio_path.stat().st_size < 1000:
        return False, f"File too small ({audio_path.stat().st_size} bytes): {path}"

    # Попробовать открыть как WAV
    try:
        with wave.open(str(audio_path), "rb") as wf:
            channels = wf.getnchannels()
            sample_rate = wf.getframerate()
            n_frames = wf.getnframes()
            sample_width = wf.getsampwidth()

            if n_frames == 0:
                return False, "Аудиофайл не содержит фреймов"

            duration = n_frames / sample_rate

            if duration < 3.0:
                return False, f"Слишком короткий ({duration:.1f}с), нужно минимум 3 секунды"

            if duration > 30.0:
                logger.warning(
                    "[TTS] Reference audio длинный (%.1fs). "
                    "Рекомендуется 6-15 секунд для лучшего качества.",
                    duration,
                )

            details = (
                f"channels={channels}, sample_rate={sample_rate}Hz, "
                f"duration={duration:.1f}s, sample_width={sample_width}B"
            )
            logger.info("[TTS] Reference audio валиден: %s", details)
            return True, details

    except wave.Error as exc:
        # Файл может быть в другом формате (mp3, ogg) — попробуем через soundfile
        try:
            import soundfile as sf
            info = sf.info(str(audio_path))
            duration = info.duration
            if duration < 3.0:
                return False, f"Слишком короткий ({duration:.1f}с)"
            details = (
                f"channels={info.channels}, sample_rate={info.samplerate}Hz, "
                f"duration={duration:.1f}s, format={info.format}"
            )
            logger.info("[TTS] Reference audio валиден (soundfile): %s", details)
            return True, details
        except Exception as sf_exc:
            return False, f"Не удалось прочитать аудио: {exc}; soundfile: {sf_exc}"

    except Exception as exc:
        return False, f"Ошибка валидации: {exc}"


import hashlib

# In-memory WAV cache for short static and confirmation phrases
_PHRASE_CACHE: dict[str, bytes] = {}
_DISK_CACHE_DIR = config.DATA_DIR / "tts_cache"


def _get_cache_path(cache_key: str) -> Path:
    _DISK_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    h = hashlib.md5(cache_key.encode("utf-8")).hexdigest()
    return _DISK_CACHE_DIR / f"{h}.wav"


def get_phrase_cache_size() -> int:
    """Return number of cached phrases."""
    return len(_PHRASE_CACHE)


def clear_phrase_cache() -> None:
    """Clear in-memory phrase cache."""
    _PHRASE_CACHE.clear()


class XTTSEngine(BaseTTSEngine):
    """XTTS-v2 движок с zero-shot voice cloning.

    Модель загружается ОДИН РАЗ при старте runtime и живёт всё время процесса.
    Speaker conditioning кэшируется для всех последующих запросов.
    Короткие повторяющиеся фразы (подтверждения) кэшируются на диске и в RAM.
    """

    def __init__(self) -> None:
        self._tts = None  # TTS model instance
        self._device: str = "cpu"
        self._model_loaded: bool = False
        self._speaker_wav: Optional[str] = None
        self._init_error: Optional[str] = None
        self._conditioning_latents = None

        # Принять лицензию Coqui CPML автоматически
        os.environ["COQUI_TOS_AGREED"] = "1"

    @property
    def model_ready(self) -> bool:
        return self._model_loaded and self._tts is not None

    def warmup(self) -> bool:
        """Pre-load model once on runtime startup to prevent first-command lag."""
        if self._model_loaded:
            return True
        try:
            self._load_model()
            return True
        except Exception as exc:
            logger.warning("[TTS] Warmup failed: %s", exc)
            return False

    def _load_model(self) -> None:
        """Загрузить XTTS-v2 модель (один раз за жизненный цикл процесса)."""
        if self._model_loaded:
            return

        if self._init_error is not None:
            raise RuntimeError(self._init_error)

        load_start = time.perf_counter()

        try:
            from TTS.api import TTS  # type: ignore[import-untyped]
        except ImportError as exc:
            self._init_error = f"coqui-tts не установлен: {exc}"
            raise RuntimeError(self._init_error) from exc

        # Определяем устройство
        self._device = _resolve_device(config.TTS_DEVICE)
        logger.info("[TTS] backend=xtts")
        logger.info("[TTS] device=%s", self._device)

        # Валидируем reference audio
        voice_path = config.TTS_VOICE_PATH
        is_valid, detail = _validate_reference_audio(voice_path)
        if not is_valid:
            self._init_error = f"Reference voice невалиден: {detail}"
            logger.error("[TTS] reference voice not found: %s", detail)
            raise RuntimeError(self._init_error)

        self._speaker_wav = voice_path

        # Загружаем модель
        try:
            model_name = config.TTS_XTTS_MODEL
            logger.info("[TTS] Загрузка модели %s (single-initialization) ...", model_name)
            self._tts = TTS(model_name=model_name).to(self._device)
            self._model_loaded = True
            load_elapsed = time.perf_counter() - load_start
            logger.info("[TTS] model_ready=%.3fs", load_elapsed)
        except Exception as exc:
            self._init_error = f"Ошибка загрузки модели: {exc}"
            logger.error("[TTS] Ошибка загрузки XTTS модели: %s", exc)
            raise RuntimeError(self._init_error) from exc

    def synthesize_and_play(self, text: str) -> None:
        """Синтезировать текст с клонированным голосом и воспроизвести.

        Проверяет двухуровневый кэш (RAM + Disk). При попадании воспроизводит без инференса.
        """
        request_start = time.perf_counter()
        cache_key = " ".join(text.strip().casefold().split()).strip(" .,!?:;\"'")

        # ── 1. Check RAM cache ──
        if getattr(config, "TTS_PHRASE_CACHE_ENABLED", True):
            if cache_key in _PHRASE_CACHE:
                cached_wav_bytes = _PHRASE_CACHE[cache_key]
                logger.info("[TTS CACHE] RAM HIT for '%s' (%d bytes) — skipping inference", text, len(cached_wav_bytes))
                self._play_wav_bytes(cached_wav_bytes)
                total_elapsed = time.perf_counter() - request_start
                logger.info("[TTS] cache_play_time=%.3fs for '%s'", total_elapsed, text)
                return

            # ── 2. Check Disk cache ──
            disk_path = _get_cache_path(cache_key)
            if disk_path.exists():
                try:
                    wav_data = disk_path.read_bytes()
                    _PHRASE_CACHE[cache_key] = wav_data
                    logger.info("[TTS CACHE] DISK HIT for '%s' (%d bytes) — skipping inference", text, len(wav_data))
                    self._play_wav_bytes(wav_data)
                    total_elapsed = time.perf_counter() - request_start
                    logger.info("[TTS] disk_cache_play_time=%.3fs for '%s'", total_elapsed, text)
                    return
                except Exception as d_exc:
                    logger.debug("[TTS CACHE] disk read error: %s", d_exc)

        # ── 3. Model Inference ──
        self._load_model()

        assert self._tts is not None
        assert self._speaker_wav is not None

        if len(text) > 250:
            logger.warning("[TTS] Текст обрезан до 250 символов (XTTS limit)")
            text = text[:247] + "..."

        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".wav", prefix="agata_tts_")
        os.close(tmp_fd)

        try:
            synthesis_start = time.perf_counter()
            self._tts.tts_to_file(
                text=text,
                file_path=tmp_path,
                speaker_wav=self._speaker_wav,
                language=config.TTS_LANGUAGE,
            )
            synthesis_elapsed = time.perf_counter() - synthesis_start
            logger.info("[TTS] synthesis_time=%.3fs", synthesis_elapsed)

            # Store in disk + RAM phrase cache if short confirmation phrase
            if getattr(config, "TTS_PHRASE_CACHE_ENABLED", True) and len(text) <= 60:
                try:
                    with open(tmp_path, "rb") as f:
                        wav_data = f.read()
                    _PHRASE_CACHE[cache_key] = wav_data
                    disk_path = _get_cache_path(cache_key)
                    disk_path.write_bytes(wav_data)
                    logger.info("[TTS CACHE] Saved '%s' to RAM & Disk cache (%s)", text, disk_path)
                except Exception as c_exc:
                    logger.debug("[TTS CACHE] Could not cache phrase: %s", c_exc)

            playback_start = time.perf_counter()
            self._play_audio(tmp_path)
            playback_elapsed = time.perf_counter() - playback_start
            logger.info("[TTS] playback_time=%.3fs", playback_elapsed)

            total_elapsed = time.perf_counter() - request_start
            logger.info("[TTS] total_time=%.3fs", total_elapsed)

        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


    @staticmethod
    def _play_wav_bytes(wav_bytes: bytes) -> None:
        """Воспроизвести WAV bytes из памяти напрямую."""
        import io
        import soundfile as sf
        import sounddevice as sd

        with io.BytesIO(wav_bytes) as bio:
            data, samplerate = sf.read(bio, dtype="float32")
        sd.play(data, samplerate)
        sd.wait()

    @staticmethod
    def _play_audio(wav_path: str) -> None:
        """Воспроизвести WAV файл через sounddevice."""
        import numpy as np
        import soundfile as sf
        import sounddevice as sd

        data, samplerate = sf.read(wav_path, dtype="float32")
        sd.play(data, samplerate)
        sd.wait()  # Блокирует до окончания воспроизведения

    def shutdown(self) -> None:
        """Освободить ресурсы XTTS модели."""
        if self._tts is not None:
            try:
                # Освобождаем GPU память
                del self._tts
                self._tts = None
                self._model_loaded = False

                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    logger.info("[TTS] XTTS модель выгружена, GPU кэш очищен")
            except Exception as exc:
                logger.warning("[TTS] Ошибка при выгрузке XTTS: %s", exc)
