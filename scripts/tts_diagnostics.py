"""
Диагностика TTS-подсистемы Агаты.

Запуск:
    python scripts/tts_diagnostics.py

Показывает состояние Python, PyTorch, CUDA, TTS backend,
reference voice и готовность к синтезу.
"""

import sys
from pathlib import Path

# Добавляем корень проекта в path
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

import config


def _separator(title: str) -> str:
    return f"\n{'-' * 40}\n  {title}\n{'-' * 40}"


def main() -> None:
    print("=" * 50)
    print("  Agata TTS Diagnostics")
    print("=" * 50)

    # ── Python ──
    print(_separator("Python"))
    print(f"  Version:     {sys.version.split()[0]}")
    print(f"  Executable:  {sys.executable}")

    # ── PyTorch ──
    print(_separator("PyTorch"))
    try:
        import torch
        print(f"  Version:     {torch.__version__}")
        print(f"  CUDA build:  {torch.version.cuda or 'N/A'}")
        cuda_ok = torch.cuda.is_available()
        print(f"  CUDA avail:  {'yes' if cuda_ok else 'no'}")
        if cuda_ok:
            print(f"  GPU:         {torch.cuda.get_device_name(0)}")
            vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"  VRAM:        {vram:.1f} GB")
        else:
            print("  GPU:         N/A")
    except ImportError:
        print("  PyTorch не установлен!")

    # ── TTS Config ──
    print(_separator("TTS Configuration"))
    print(f"  Engine:      {config.TTS_ENGINE}")
    print(f"  Device:      {config.TTS_DEVICE}")
    print(f"  Language:    {config.TTS_LANGUAGE}")
    print(f"  Model:       {config.TTS_XTTS_MODEL}")
    print(f"  Fallback:    {config.TTS_FALLBACK_ENGINE}")
    print(f"  Enabled:     {config.TTS_ENABLED}")

    # ── Reference Voice ──
    print(_separator("Reference Voice"))
    voice_path = config.TTS_VOICE_PATH
    print(f"  Path:        {voice_path}")

    from tts.xtts_engine import _validate_reference_audio
    is_valid, detail = _validate_reference_audio(voice_path)
    print(f"  Valid:        {'yes' if is_valid else 'no'}")
    print(f"  Details:     {detail}")

    # ── coqui-tts ──
    print(_separator("coqui-tts Library"))
    try:
        import TTS
        version = getattr(TTS, "__version__", "unknown")
        print(f"  Version:     {version}")
        print(f"  Installed:   yes")
    except ImportError:
        print("  Installed:   no")
        print("  Install:     pip install coqui-tts")

    # ── Model Cache ──
    print(_separator("Model Cache"))
    import os
    if sys.platform == "win32":
        cache_dir = Path(os.environ.get("LOCALAPPDATA", "")) / "tts"
    else:
        cache_dir = Path.home() / ".local" / "share" / "tts"
    print(f"  Cache dir:   {cache_dir}")
    if cache_dir.exists():
        models = list(cache_dir.rglob("*.pth")) + list(cache_dir.rglob("*.json"))
        print(f"  Files found: {len(models)}")
        # Проверяем наличие XTTS-v2
        xtts_files = [f for f in models if "xtts" in str(f).lower()]
        if xtts_files:
            print(f"  XTTS files:  {len(xtts_files)}")
        else:
            print("  XTTS files:  not downloaded yet")
    else:
        print("  Cache dir:   not found (model not downloaded)")

    # ── Overall Status ──
    print(_separator("Status"))
    errors = []

    if config.TTS_ENGINE == "xtts":
        try:
            import TTS
        except ImportError:
            errors.append("coqui-tts не установлен")

        if not is_valid:
            errors.append(f"Reference voice невалиден: {detail}")

    if errors:
        print(f"  Status:      ERROR")
        for e in errors:
            print(f"  [ERROR] {e}")
    else:
        print(f"  Status:      READY")
        print(f"  [OK] Все проверки пройдены")

    print()


if __name__ == "__main__":
    main()
