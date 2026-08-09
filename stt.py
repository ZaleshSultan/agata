"""
Распознавание речи (Speech-to-Text).

Запись с микрофона через sounddevice, VAD по амплитуде,
распознавание через faster-whisper (модель small, язык ru).

Побочные эффекты: доступ к микрофону, загрузка модели whisper при первом вызове.
"""

import logging
import io
import wave
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import sounddevice as sd  # type: ignore[import-untyped]

import config

logger = logging.getLogger(__name__)

# Ленивая загрузка модели whisper
_model = None


def _get_model():
    """Загрузить модель faster-whisper при первом вызове."""
    global _model
    if _model is None:
        from faster_whisper import WhisperModel  # type: ignore[import-untyped]
        logger.info(
            "Загрузка модели whisper: size=%s, device=%s, compute=%s",
            config.WHISPER_MODEL_SIZE,
            config.WHISPER_DEVICE,
            config.WHISPER_COMPUTE_TYPE,
        )
        try:
            _model = WhisperModel(
                config.WHISPER_MODEL_SIZE,
                device=config.WHISPER_DEVICE,
                compute_type=config.WHISPER_COMPUTE_TYPE,
            )
        except Exception:
            # Fallback на CPU, если CUDA недоступна
            logger.warning("CUDA недоступна, загружаю whisper на CPU")
            _model = WhisperModel(
                config.WHISPER_MODEL_SIZE,
                device="cpu",
                compute_type="int8",
            )
    return _model


def _record_audio() -> np.ndarray:
    """
    Записать аудио с микрофона до тишины.

    Использует VAD по амплитуде: запись прекращается, когда амплитуда
    остаётся ниже порога SILENCE_THRESHOLD в течение SILENCE_DURATION секунд.

    Returns:
        numpy-массив с аудиоданными (float32, mono).
    """
    sample_rate: int = config.SAMPLE_RATE
    block_size: int = int(sample_rate * 0.1)  # 100мс блоки
    silence_blocks: int = int(config.SILENCE_DURATION / 0.1)
    max_blocks: int = int(config.MAX_RECORD_SECONDS / 0.1)

    frames: list[np.ndarray] = []
    silent_count: int = 0
    has_speech: bool = False

    logger.info("Запись... (говори)")

    def callback(indata: np.ndarray, frame_count: int, time_info, status) -> None:
        nonlocal silent_count, has_speech
        if status:
            logger.warning("Статус аудио: %s", status)
        frames.append(indata.copy())
        amplitude: float = float(np.abs(indata).mean())

        if amplitude > config.SILENCE_THRESHOLD:
            has_speech = True
            silent_count = 0
        else:
            if has_speech:
                silent_count += 1

    with sd.InputStream(
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
        blocksize=block_size,
        callback=callback,
    ):
        import time
        for _ in range(max_blocks):
            time.sleep(0.1)
            if has_speech and silent_count >= silence_blocks:
                logger.info("Тишина обнаружена — останавливаю запись")
                break
        else:
            logger.warning("Достигнут лимит записи (%dс)", config.MAX_RECORD_SECONDS)

    if not frames:
        return np.array([], dtype=np.float32)

    audio = np.concatenate(frames, axis=0).flatten()
    duration = len(audio) / sample_rate
    logger.info("Записано %.1fс аудио", duration)
    return audio


def _audio_to_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    """Конвертировать numpy-массив в WAV bytes для whisper."""
    # Конвертируем float32 [-1, 1] в int16
    audio_int16 = (audio * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16 bit
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())
    return buf.getvalue()


def record_and_transcribe() -> str:
    """
    Записать речь с микрофона и распознать в текст.

    Returns:
        Распознанный текст (строка) или пустая строка, если ничего не сказано.
    """
    audio = _record_audio()
    if len(audio) < config.SAMPLE_RATE * 0.3:  # Меньше 0.3с — считаем пустым
        logger.info("Слишком короткая запись — пропускаю")
        return ""

    # Сохраняем во временный WAV-файл для whisper
    wav_bytes = _audio_to_wav_bytes(audio, config.SAMPLE_RATE)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(wav_bytes)
        tmp_path = tmp.name

    try:
        text = transcribe_file(tmp_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return text


def transcribe_file(path: str) -> str:
    """
    Распознать речь из аудиофайла.

    Args:
        path: Путь к WAV/MP3 файлу.

    Returns:
        Распознанный текст.
    """
    model = _get_model()
    segments, info = model.transcribe(path, language="ru", beam_size=5)
    text_parts: list[str] = []
    for segment in segments:
        text_parts.append(segment.text.strip())
    text = " ".join(text_parts).strip()
    logger.info("Распознано: '%s' (язык: %s, вероятность: %.2f)",
                text[:100], info.language, info.language_probability)
    return text
