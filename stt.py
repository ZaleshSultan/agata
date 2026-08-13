"""
Распознавание речи (Speech-to-Text).

Запись с микрофона через sounddevice, energy-based VAD по RMS,
распознавание через faster-whisper.

Побочные эффекты:
- доступ к микрофону;
- загрузка модели Whisper при первом вызове.
"""

import io
import logging
import tempfile
import time
import wave
from collections import deque
from pathlib import Path

import numpy as np
import sounddevice as sd  # type: ignore[import-untyped]

import config
import tts
from stt_normalize import normalize_stt

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Whisper
# ─────────────────────────────────────────────────────────────────────────────

_model = None
_model_device: str = "cpu"
_model_compute: str = "int8"


def _get_model():
    """Загрузить модель faster-whisper при первом вызове (singleton)."""
    global _model, _model_device, _model_compute

    if _model is None:
        import time
        from faster_whisper import WhisperModel  # type: ignore[import-untyped]
        from runtime_capabilities import resolve_stt_device

        device, compute_type = resolve_stt_device(
            getattr(config, "WHISPER_DEVICE", "auto"),
            getattr(config, "WHISPER_COMPUTE_TYPE", "auto"),
        )
        logger.info(
            "[STT] Загрузка модели whisper: size=%s, resolved_device=%s, compute=%s",
            config.WHISPER_MODEL_SIZE,
            device,
            compute_type,
        )

        load_start = time.perf_counter()
        try:
            _model = WhisperModel(
                config.WHISPER_MODEL_SIZE,
                device=device,
                compute_type=compute_type,
            )
            _model_device = device
            _model_compute = compute_type
            logger.info("[STT] Model loaded successfully in %.3fs on %s (%s)", time.perf_counter() - load_start, device, compute_type)
        except Exception as exc:
            logger.warning(
                "[STT] Не удалось загрузить Whisper на %s (%s): %s",
                device,
                compute_type,
                exc,
            )
            logger.warning("[STT] Переход на CPU fallback: device=cpu, compute_type=int8")
            _model = WhisperModel(
                config.WHISPER_MODEL_SIZE,
                device="cpu",
                compute_type="int8",
            )
            _model_device = "cpu"
            _model_compute = "int8"
            logger.info("[STT] CPU fallback model loaded in %.3fs", time.perf_counter() - load_start)

    return _model


# ─────────────────────────────────────────────────────────────────────────────
# Audio recording / VAD
# ─────────────────────────────────────────────────────────────────────────────


def _record_audio() -> np.ndarray:
    """
    Записать речь с микрофона до обнаружения тишины.

    Используется energy-based VAD по RMS с многоуровневой защитой:
    - Ожидание завершения TTS + кулдаун;
    - Требование минимальной длительности звучащих кадров;
    - Сбор подробных метрик аудио для логов.
    """
    # ── Проверка TTS блокировки и кулдауна ─────────────────────────────────────
    if tts.is_speaking():
        logger.info("[STT] TTS воспроизводится — ожидаю завершения...")
        while tts.is_speaking():
            time.sleep(0.05)

    time_since_tts = time.time() - tts.get_last_tts_finish_time()
    if time_since_tts < config.TTS_COOLDOWN_SECONDS:
        cooldown_wait = config.TTS_COOLDOWN_SECONDS - time_since_tts
        logger.info("[STT] Ожидание кулдауна после TTS (%.2fс)...", cooldown_wait)
        time.sleep(cooldown_wait)

    sample_rate: int = config.SAMPLE_RATE
    block_duration: float = 0.05
    block_size: int = int(sample_rate * block_duration)

    silence_blocks: int = max(1, int(config.SILENCE_DURATION / block_duration))
    max_blocks: int = max(1, int(config.MAX_RECORD_SECONDS / block_duration))
    pre_roll_blocks: int = max(1, int(config.PRE_ROLL_SECONDS / block_duration))
    min_speech_blocks: int = max(1, int(config.MIN_SPEECH_SECONDS / block_duration))

    pre_roll: deque[np.ndarray] = deque(maxlen=pre_roll_blocks)
    frames: list[np.ndarray] = []

    has_speech = False
    speech_blocks = 0
    silent_count = 0
    voiced_blocks = 0
    max_rms = 0.0

    logger.info("[STT] recording started... (говорите)")

    def callback(indata: np.ndarray, frame_count: int, time_info, status) -> None:
        nonlocal has_speech, speech_blocks, silent_count, voiced_blocks, max_rms

        if status:
            logger.warning("[STT] Статус аудио: %s", status)

        block = indata.copy()
        rms = float(np.sqrt(np.mean(np.square(block))))
        max_rms = max(max_rms, rms)

        if rms >= config.SILENCE_THRESHOLD:
            voiced_blocks += 1

        if not has_speech:
            pre_roll.append(block)
            if rms >= config.SILENCE_THRESHOLD:
                speech_blocks += 1
                if speech_blocks >= min_speech_blocks:
                    has_speech = True
                    frames.extend(pre_roll)
                    pre_roll.clear()
                    frames.append(block)
                    silent_count = 0
            else:
                speech_blocks = max(0, speech_blocks - 1)
        else:
            frames.append(block)
            if rms >= config.SILENCE_THRESHOLD:
                silent_count = 0
            else:
                silent_count += 1

    try:
        with sd.InputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
            blocksize=block_size,
            callback=callback,
        ):
            for _ in range(max_blocks):
                time.sleep(block_duration)
                if has_speech and silent_count >= silence_blocks:
                    logger.info("[STT] Тишина обнаружена — останавливаю запись")
                    break
    except Exception as exc:
        logger.error("[STT] Ошибка записи с микрофона: %s", exc)
        return np.array([], dtype=np.float32)

    total_duration = (len(frames) * block_size) / sample_rate if frames else 0.0
    voiced_duration = voiced_blocks * block_duration

    logger.info(
        "[STT] recording=%.2fs voiced=%.2fs max_rms=%.5f speech_detected=%s",
        total_duration,
        voiced_duration,
        max_rms,
        has_speech,
    )

    if not has_speech or voiced_duration < config.MIN_VOICED_AUDIO_SECONDS:
        logger.info("[STT] rejected: no_speech")
        return np.array([], dtype=np.float32)

    audio = np.concatenate(frames, axis=0).flatten()
    return audio


# ─────────────────────────────────────────────────────────────────────────────
# WAV conversion
# ─────────────────────────────────────────────────────────────────────────────


def _audio_to_wav_bytes(
    audio: np.ndarray,
    sample_rate: int,
) -> bytes:
    """
    Конвертировать numpy float32 [-1, 1] в WAV bytes.
    """

    # Ограничиваем диапазон перед преобразованием.
    audio = np.clip(
        audio,
        -1.0,
        1.0,
    )

    audio_int16 = (
        audio * 32767
    ).astype(np.int16)

    buffer = io.BytesIO()

    with wave.open(
        buffer,
        "wb",
    ) as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(
            audio_int16.tobytes()
        )

    return buffer.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
# Record + transcribe
# ─────────────────────────────────────────────────────────────────────────────


def record_and_transcribe(is_confirmation: bool = False) -> str:
    """
    Записать речь с микрофона и распознать её.

    Args:
        is_confirmation: Если True, используется облегчённый режим отбраковки
                         для коротких ответов вроде "Да", "Нет", "Ага".

    Returns:
        Распознанный текст или пустую строку.
    """
    audio = _record_audio()

    # Слишком короткая запись (для confirmation допускаем более короткие клипы)
    min_audio_len = config.SAMPLE_RATE * (0.15 if is_confirmation else 0.25)
    if len(audio) < min_audio_len:
        logger.info("[STT] Слишком короткая запись — пропускаю")
        return ""

    # Проверка на явно битую/нулевую запись
    audio_rms = float(np.sqrt(np.mean(np.square(audio))))
    silence_check = config.SILENCE_THRESHOLD * (0.15 if is_confirmation else 0.25)
    if audio_rms < silence_check:
        logger.info("[STT] Аудио явно пустое (RMS=%.5f) — пропускаю", audio_rms)
        return ""

    # ─────────────────────────────────────────────────
    # Создаём временный WAV
    # ─────────────────────────────────────────────────
    wav_bytes = _audio_to_wav_bytes(
        audio,
        config.SAMPLE_RATE,
    )

    tmp_path: str | None = None

    try:
        with tempfile.NamedTemporaryFile(
            suffix=".wav",
            delete=False,
        ) as tmp:
            tmp.write(wav_bytes)
            tmp_path = tmp.name

        raw_text = transcribe_file(tmp_path, is_confirmation=is_confirmation)
        if not raw_text:
            return ""

        # ─────────────────────────────────────────────
        # Нормализация: raw STT → canonical text
        # ─────────────────────────────────────────────
        normalized = normalize_stt(raw_text)
        if normalized != raw_text:
            logger.info("[STT] raw: '%s' -> normalized: '%s'", raw_text, normalized)
        else:
            logger.info("[STT] recognized: '%s'", raw_text)
        return normalized

    except Exception as exc:
        logger.error("[STT] Ошибка STT: %s", exc)
        return ""

    finally:
        if tmp_path is not None:
            Path(tmp_path).unlink(missing_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# File transcription
# ─────────────────────────────────────────────────────────────────────────────


def transcribe_file(path: str, is_confirmation: bool = False) -> str:
    """
    Распознать речь из аудиофайла с проверкой метрик качества Whisper.
    """
    import time
    model = _get_model()

    transcribe_start = time.perf_counter()

    prompt = "Да, нет, отмена, подтверждаю." if is_confirmation else (
        "Агата, открой хром, закрой хром, открой ютуб, "
        "напомни мне, сколько времени, посчитай."
    )

    segments_gen, info = model.transcribe(
        path,
        language="ru",
        beam_size=5,
        temperature=0.0,
        initial_prompt=prompt,
        vad_filter=True,
        vad_parameters={
            "min_silence_duration_ms": 300 if is_confirmation else 500,
            "speech_pad_ms": 200,
        },
        condition_on_previous_text=False,
        no_speech_threshold=0.7 if is_confirmation else 0.6,
        compression_ratio_threshold=2.4,
        log_prob_threshold=-1.2 if is_confirmation else -1.0,
    )

    segment_list = list(segments_gen)
    transcribe_elapsed = time.perf_counter() - transcribe_start

    if not segment_list:
        logger.info("[STT] rejected: empty_whisper (%.3fs)", transcribe_elapsed)
        return ""

    text_parts = [s.text.strip() for s in segment_list if s.text.strip()]
    raw_text = " ".join(text_parts).strip()

    if not raw_text:
        logger.info("[STT] rejected: empty_text (%.3fs)", transcribe_elapsed)
        return ""

    avg_no_speech = sum(s.no_speech_prob for s in segment_list) / len(segment_list)
    avg_logprob = sum(s.avg_logprob for s in segment_list) / len(segment_list)
    max_compression = max(s.compression_ratio for s in segment_list)

    logger.info(
        "[STT DIAG] text=\"%s\" device=%s compute=%s duration=%.3fs no_speech_prob=%.2f avg_logprob=%.2f compression=%.2f is_confirm=%s",
        raw_text,
        _model_device,
        _model_compute,
        transcribe_elapsed,
        avg_no_speech,
        avg_logprob,
        max_compression,
        is_confirmation,
    )

    max_no_speech_allowed = 0.65 if is_confirmation else config.WHISPER_MAX_NO_SPEECH_PROB
    if avg_no_speech > max_no_speech_allowed:
        logger.info("[STT] rejected: high_no_speech_prob (%.2f > %.2f)", avg_no_speech, max_no_speech_allowed)
        return ""

    min_logprob_allowed = -1.5 if is_confirmation else config.WHISPER_MIN_AVG_LOGPROB
    if avg_logprob < min_logprob_allowed:
        logger.info("[STT] rejected: low_avg_logprob (%.2f < %.2f)", avg_logprob, min_logprob_allowed)
        return ""

    # Relaxed compression ratio check for short phrases / confirmation
    min_comp = 0.0 if is_confirmation or len(raw_text) <= 12 else 0.4
    if max_compression > 2.4 or max_compression < min_comp:
        logger.info("[STT] rejected: abnormal_compression_ratio (%.2f)", max_compression)
        return ""

    hallucination_blacklist = (
        "субтитры", "продолжение следует", "редактор субтитров",
        "спасибо за просмотр", "благодарим за просмотр", "подписывайтесь",
        "переводчик", "академия", "создатель субтитров", "работа над видео",
    )
    raw_text_lower = raw_text.lower()
    if any(phrase in raw_text_lower for phrase in hallucination_blacklist):
        logger.info("[STT] rejected: hallucination_blacklist ('%s')", raw_text)
        return ""

    logger.info("[STT] accepted: '%s'", raw_text)
    return raw_text