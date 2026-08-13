"""
End-to-end XTTS-v2 test.

Loads the real model, synthesizes speech, and saves WAV.
Uses the same code path as XTTSEngine.synthesize_and_play(),
but saves WAV to a fixed path instead of a temp file.

Run:
    python tests/e2e_xtts_test.py
"""

import os
import sys
import time
import wave

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# Force UTF-8 output on Windows cp1251 consoles
sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]

import config
from tts.xtts_engine import XTTSEngine


def main() -> None:
    text = "Привет, Залеш. Я Агата. Это мой новый голос."
    output_dir = os.path.join(PROJECT_ROOT, "tests", "output")
    output_path = os.path.join(output_dir, "agata_xtts_test.wav")

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("XTTS-v2 End-to-End Test")
    print("=" * 60)
    print(f"  Text:      {text}")
    print(f"  Backend:   XTTS ({config.TTS_XTTS_MODEL})")
    print(f"  Device:    {config.TTS_DEVICE}")
    print(f"  Reference: {config.TTS_VOICE_PATH}")
    print(f"  Language:  {config.TTS_LANGUAGE}")
    print(f"  Output:    {output_path}")
    print("-" * 60)

    # -- 1. Create engine and load model -----------------------------------
    engine = XTTSEngine()

    print("[1/3] Loading XTTS-v2 model...")
    load_start = time.perf_counter()
    engine._load_model()
    load_elapsed = time.perf_counter() - load_start
    print(f"      Model loaded in {load_elapsed:.3f}s")
    print(f"      Device: {engine._device}")

    # Verify: this is XTTS, not pyttsx3
    assert engine._tts is not None, "Model not loaded!"
    assert engine._model_loaded is True, "Model flag not set!"
    assert engine._speaker_wav is not None, "Speaker WAV not set!"
    print("      [OK] XTTS model confirmed (NOT pyttsx3 fallback)")

    # -- 2. Synthesis ------------------------------------------------------
    print(f"[2/3] Synthesizing: '{text}'")
    synth_start = time.perf_counter()
    engine._tts.tts_to_file(
        text=text,
        file_path=output_path,
        speaker_wav=engine._speaker_wav,
        language=config.TTS_LANGUAGE,
    )
    synth_elapsed = time.perf_counter() - synth_start
    print(f"      Synthesis time: {synth_elapsed:.3f}s")

    # -- 3. Validate output ------------------------------------------------
    print("[3/3] Validating output...")

    assert os.path.exists(output_path), f"Output file not created: {output_path}"
    file_size = os.path.getsize(output_path)
    assert file_size > 0, "Output file is empty!"

    # Get WAV duration
    try:
        with wave.open(output_path, "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            channels = wf.getnchannels()
            duration = frames / rate
    except wave.Error:
        # XTTS may produce non-standard WAV — try soundfile
        import soundfile as sf
        info = sf.info(output_path)
        duration = info.duration
        rate = info.samplerate
        channels = info.channels

    print(f"      [OK] File exists: {output_path}")
    print(f"      [OK] File size:   {file_size:,} bytes")
    print(f"      [OK] Duration:    {duration:.3f}s")
    print(f"      [OK] Sample rate: {rate} Hz")
    print(f"      [OK] Channels:    {channels}")

    # -- Results -----------------------------------------------------------
    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  Model load time:  {load_elapsed:.3f}s")
    print(f"  Synthesis time:   {synth_elapsed:.3f}s")
    print(f"  Output duration:  {duration:.3f}s")
    print(f"  Output path:      {output_path}")
    print(f"  RTF (real-time):  {synth_elapsed / duration:.2f}x")
    print("=" * 60)
    print("[PASS] XTTS-v2 E2E TEST PASSED")

    # Cleanup model
    engine.shutdown()


if __name__ == "__main__":
    main()
