"""
Конфигурация Агаты.

Все настройки проекта в одном месте — пути, URL, модели, лимиты.
Импортируй этот модуль везде вместо хардкода констант.
"""

from pathlib import Path

# ── Корневые пути ──────────────────────────────────────────────────────────
BASE_DIR: Path = Path(__file__).resolve().parent
DATA_DIR: Path = BASE_DIR / "data"

# ── Ollama ─────────────────────────────────────────────────────────────────
OLLAMA_URL: str = "http://localhost:11434"
LLM_MODEL: str = "agata"                       # Qwen3-30B-A3B + Modelfile-характер
EMBED_MODEL: str = "nomic-embed-text"           # Модель для эмбеддингов

# ── Ассистент ──────────────────────────────────────────────────────────────
ASSISTANT_NAME: str = "Агата"

# ── Память ─────────────────────────────────────────────────────────────────
DB_PATH: Path = DATA_DIR / "agata.db"           # SQLite: дневник + напоминания
WORKING_MEMORY_PATH: Path = DATA_DIR / "working_memory.md"

# ── Agent loop ─────────────────────────────────────────────────────────────
MAX_AGENT_STEPS: int = 5                        # Макс. итераций tool-calling за один запрос

# ── Thinking / Рассуждения ────────────────────────────────────────────────
ENABLE_ADAPTIVE_THINKING: bool = True  # модель сама решает think true/false на каждый запрос
# Используется только если ENABLE_ADAPTIVE_THINKING = True.
# Если False — используется статичный ENABLE_THINKING как раньше.
ENABLE_THINKING: bool = False  # fallback-значение, если адаптивный режим выключен/не сработал

# ── STT (faster-whisper) ───────────────────────────────────────────────────
WHISPER_MODEL_SIZE: str = "small"               # tiny / base / small / medium / large-v3
WHISPER_DEVICE: str = "cuda"                    # "cuda" для RTX 4050, "cpu" как fallback
WHISPER_COMPUTE_TYPE: str = "float16"           # float16 на GPU, int8 на CPU

# ── TTS ────────────────────────────────────────────────────────────────────
TTS_RATE: int = 175                             # Скорость речи pyttsx3 (слов/мин)

# ── Аудиозапись ────────────────────────────────────────────────────────────
SAMPLE_RATE: int = 16000                        # Частота дискретизации (Гц)
SILENCE_THRESHOLD: float = 0.02                 # Порог амплитуды для VAD
SILENCE_DURATION: float = 1.5                   # Сколько секунд тишины = конец фразы
MAX_RECORD_SECONDS: int = 30                    # Максимальная длина записи

# ── Консолидация ───────────────────────────────────────────────────────────
CONSOLIDATION_DAYS: int = 7                     # За сколько дней брать записи для "сна"
SIMILARITY_THRESHOLD: float = 0.75              # Порог косинусной схожести для кластеризации
