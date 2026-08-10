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
from stt_normalize import normalize_stt

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Whisper
# ─────────────────────────────────────────────────────────────────────────────

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

        except Exception as exc:
            logger.warning(
                "Не удалось загрузить Whisper на %s: %s",
                config.WHISPER_DEVICE,
                exc,
            )

            logger.warning(
                "Переход на CPU: device=cpu, compute_type=int8"
            )

            _model = WhisperModel(
                config.WHISPER_MODEL_SIZE,
                device="cpu",
                compute_type="int8",
            )

    return _model


# ─────────────────────────────────────────────────────────────────────────────
# Audio recording / VAD
# ─────────────────────────────────────────────────────────────────────────────


def _record_audio() -> np.ndarray:
    """
    Записать речь с микрофона до обнаружения тишины.

    Используется energy-based VAD по RMS:

    1. Ждём появления речи.
    2. Храним небольшой pre-roll до начала речи.
    3. После начала речи продолжаем запись.
    4. При достаточной тишине завершаем запись.
    5. Есть максимальный лимит длительности.

    Returns:
        numpy-массив float32, mono.
        Пустой массив, если речь не обнаружена.
    """

    sample_rate: int = config.SAMPLE_RATE

    # Размер одного блока — 50 мс.
    block_duration: float = 0.05
    block_size: int = int(sample_rate * block_duration)

    silence_blocks: int = max(
        1,
        int(config.SILENCE_DURATION / block_duration),
    )

    max_blocks: int = max(
        1,
        int(config.MAX_RECORD_SECONDS / block_duration),
    )

    pre_roll_blocks: int = max(
        1,
        int(config.PRE_ROLL_SECONDS / block_duration),
    )

    min_speech_blocks: int = max(
        1,
        int(config.MIN_SPEECH_SECONDS / block_duration),
    )

    # Храним последние N блоков до начала речи.
    # Это позволяет не отрезать первые слоги.
    pre_roll: deque[np.ndarray] = deque(
        maxlen=pre_roll_blocks
    )

    frames: list[np.ndarray] = []

    has_speech = False
    speech_blocks = 0
    silent_count = 0

    logger.info("Запись... (говори)")

    def callback(
        indata: np.ndarray,
        frame_count: int,
        time_info,
        status,
    ) -> None:
        nonlocal has_speech
        nonlocal speech_blocks
        nonlocal silent_count

        if status:
            logger.warning(
                "Статус аудио: %s",
                status,
            )

        block = indata.copy()

        # RMS лучше подходит для energy-based VAD,
        # чем abs(indata).mean().
        rms = float(
            np.sqrt(
                np.mean(
                    np.square(block)
                )
            )
        )

        # ─────────────────────────────────────────────
        # Ещё не начали запись речи
        # ─────────────────────────────────────────────

        if not has_speech:
            pre_roll.append(block)

            if rms >= config.SILENCE_THRESHOLD:
                speech_blocks += 1

                # Требуем несколько последовательных
                # активных блоков, чтобы не реагировать
                # на случайный щелчок/шум.
                if speech_blocks >= min_speech_blocks:
                    has_speech = True

                    # Возвращаем pre-roll, чтобы сохранить
                    # начало фразы.
                    frames.extend(pre_roll)
                    pre_roll.clear()

                    frames.append(block)

                    silent_count = 0

            else:
                # "Leaky bucket": одиночный короткий провал между
                # слогами не должен обнулять весь набранный прогресс —
                # только слегка снижать счётчик. Полный сброс делал
                # min_speech_blocks практически недостижимым при
                # естественной речи (подтверждено диагностикой mic_test.py).
                speech_blocks = max(0, speech_blocks - 1)

        # ─────────────────────────────────────────────
        # Уже записываем речь
        # ─────────────────────────────────────────────

        else:
            frames.append(block)

            if rms >= config.SILENCE_THRESHOLD:
                silent_count = 0
            else:
                silent_count += 1

    # ─────────────────────────────────────────────────
    # Запуск микрофона
    # ─────────────────────────────────────────────────

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

                if (
                    has_speech
                    and silent_count >= silence_blocks
                ):
                    logger.info(
                        "Тишина обнаружена — останавливаю запись"
                    )
                    break

            else:
                logger.warning(
                    "Достигнут лимит записи (%dс)",
                    config.MAX_RECORD_SECONDS,
                )

    except Exception as exc:
        logger.error(
            "Ошибка записи с микрофона: %s",
            exc,
        )
        return np.array(
            [],
            dtype=np.float32,
        )

    # ─────────────────────────────────────────────────
    # Проверка результата
    # ─────────────────────────────────────────────────

    if not frames:
        logger.info("Речь не обнаружена")

        return np.array(
            [],
            dtype=np.float32,
        )

    audio = np.concatenate(
        frames,
        axis=0,
    ).flatten()

    duration = len(audio) / sample_rate

    logger.info(
        "Записано %.1fс аудио",
        duration,
    )

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


def record_and_transcribe() -> str:
    """
    Записать речь с микрофона и распознать её.

    Returns:
        Распознанный текст или пустую строку.
    """

    audio = _record_audio()

    # Слишком короткая запись.
    if len(audio) < config.SAMPLE_RATE * 0.3:
        logger.info(
            "Слишком короткая запись — пропускаю"
        )
        return ""

    # Проверка на явно битую/нулевую запись —
    # порог сильно ниже блочного, т.к. это RMS по всему клипу с тишиной.
    audio_rms = float(np.sqrt(np.mean(np.square(audio))))
    if audio_rms < config.SILENCE_THRESHOLD * 0.25:
        logger.info("Аудио явно пустое (RMS=%.5f) — пропускаю", audio_rms)
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

        raw_text = transcribe_file(tmp_path)
        if not raw_text:
            return ""

        # ─────────────────────────────────────────────
        # Нормализация: raw STT → canonical text
        # ─────────────────────────────────────────────
        normalized = normalize_stt(raw_text)
        if normalized != raw_text:
            logger.info("STT raw: '%s'", raw_text)
            logger.info("STT normalized: '%s'", normalized)
        else:
            logger.info("STT: '%s'", raw_text)
        return normalized

    except Exception as exc:
        logger.error(
            "Ошибка STT: %s",
            exc,
        )
        return ""

    finally:
        if tmp_path is not None:
            Path(tmp_path).unlink(
                missing_ok=True
            )


# ─────────────────────────────────────────────────────────────────────────────
# File transcription
# ─────────────────────────────────────────────────────────────────────────────


def transcribe_file(path: str) -> str:
    """
    Распознать речь из аудиофайла.

    Args:
        path: Путь к WAV/MP3 файлу.

    Returns:
        Распознанный текст.
    """

    model = _get_model()

    logger.info(
        "Whisper: начинаю распознавание файла %s",
        path,
    )

    segments, info = model.transcribe(
        path,

        # Агата ориентирована на русский язык.
        language="ru",

        # beam_size=5 — стабильный выбор для RTX 4050 (не медленнее beam_size=1
        # на больших моделях, но точнее).
        beam_size=5,
        temperature=0.0,

        # initial_prompt — подсказка Whisper о домене и имени ассистента.
        # Это улучшает распознавание собственных имён и команд.
        initial_prompt="Агата, открой, закрой, запусти, найди. Привет, Агата!",

        # Дополнительный VAD внутри Whisper.
        vad_filter=True,
        vad_parameters={
            "min_silence_duration_ms": 500,
            "speech_pad_ms": 200,
        },

        # Не переносим предыдущий контекст между сегментами.
        # Это уменьшает вероятность галлюцинаций.
        condition_on_previous_text=False,

        # Настройки фильтрации плохого распознавания.
        no_speech_threshold=0.6,
        compression_ratio_threshold=2.4,
        log_prob_threshold=-1.0,
    )

    text_parts: list[str] = []

    for segment in segments:
        segment_text = segment.text.strip()

        if segment_text:
            text_parts.append(segment_text)

    text = " ".join(
        text_parts
    ).strip()

    logger.info(
        "Распознано: '%s' (язык: %s, вероятность: %.2f)",
        text[:200],
        info.language,
        info.language_probability,
    )

    return text