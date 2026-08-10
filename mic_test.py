"""
Диагностика микрофона: показывает живой RMS каждые 50мс в течение 8 секунд,
рядом с текущим порогом SILENCE_THRESHOLD из config.py.

Запуск: python mic_test.py
Говори что-нибудь в микрофон сразу после старта.
"""

import time
import numpy as np
import sounddevice as sd

import config

sample_rate = config.SAMPLE_RATE
block_duration = 0.05
block_size = int(sample_rate * block_duration)

print("=" * 60)
print(f"Текущий config.SILENCE_THRESHOLD = {config.SILENCE_THRESHOLD}")
print(f"Текущий config.MIN_SPEECH_SECONDS = {getattr(config, 'MIN_SPEECH_SECONDS', '?')}")
print(f"Устройство записи по умолчанию: {sd.query_devices(kind='input')['name']}")
print("=" * 60)
print("Говори что-нибудь в микрофон — 8 секунд...")
print()

max_rms_seen = 0.0
above_threshold_count = 0
total_blocks = 0


def callback(indata, frame_count, time_info, status):
    global max_rms_seen, above_threshold_count, total_blocks
    if status:
        print(f"[STATUS] {status}")
    rms = float(np.sqrt(np.mean(np.square(indata))))
    max_rms_seen = max(max_rms_seen, rms)
    total_blocks += 1
    marker = "🔊" if rms >= config.SILENCE_THRESHOLD else "  "
    above_threshold_count += rms >= config.SILENCE_THRESHOLD
    bar = "#" * min(int(rms * 500), 50)
    print(f"{marker} RMS={rms:.5f} {bar}")


with sd.InputStream(
    samplerate=sample_rate,
    channels=1,
    dtype="float32",
    blocksize=block_size,
    callback=callback,
):
    time.sleep(8)

print()
print("=" * 60)
print(f"Максимальный RMS за сессию: {max_rms_seen:.5f}")
print(f"Порог SILENCE_THRESHOLD:    {config.SILENCE_THRESHOLD}")
print(f"Блоков выше порога: {above_threshold_count} из {total_blocks}")
if max_rms_seen < config.SILENCE_THRESHOLD:
    print("⚠️  Максимальный RMS НИКОГДА не достиг порога — либо микрофон")
    print("   слишком тихий/не тот выбран, либо порог завышен.")
elif above_threshold_count < 3:
    print("⚠️  Порог превышался, но очень редко/коротко — возможно,")
    print("   min_speech_blocks не успевает набраться подряд.")
else:
    print("✅ Похоже, микрофон и порог в порядке — проблема может быть")
    print("   в чём-то другом (например, в устройстве по умолчанию")
    print("   именно внутри sd.InputStream отличается от системного).")
print("=" * 60)