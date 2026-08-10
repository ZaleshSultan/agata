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
OLLAMA_KEEP_ALIVE: str = "30m"                  # Время удержания модели в VRAM при простое

# ── Ассистент ──────────────────────────────────────────────────────────────
ASSISTANT_NAME: str = "Агата"

# ── Память ─────────────────────────────────────────────────────────────────
DB_PATH: Path = DATA_DIR / "agata.db"           # SQLite: дневник + напоминания
WORKING_MEMORY_PATH: Path = DATA_DIR / "working_memory.md"

# ── Agent loop ─────────────────────────────────────────────────────────────
MAX_AGENT_STEPS: int = 5                        # Макс. итераций tool-calling за один запрос

# ── Thinking / Рассуждения ────────────────────────────────────────────────
# При True Agent использует только локальные эвристики: сложные запросы могут
# включить think=True, все остальные идут с think=False. Дополнительного LLM-
# классификатора нет.
ENABLE_ADAPTIVE_THINKING: bool = True
# Используется как статичное значение, если локальная эвристика отключена.
ENABLE_THINKING: bool = False

# ── STT (faster-whisper) ───────────────────────────────────────────────────
# large-v3-turbo — оптимальный баланс качество/скорость для русского языка на RTX 4050.
# Альтернативы: tiny / base / small / medium / large-v3 / large-v3-turbo
WHISPER_MODEL_SIZE: str = "large-v3-turbo"
WHISPER_DEVICE: str = "cuda"                    # "cuda" для RTX 4050, "cpu" как fallback
WHISPER_COMPUTE_TYPE: str = "float16"           # float16 на GPU, int8 на CPU

# ── TTS ────────────────────────────────────────────────────────────────────
TTS_RATE: int = 175                             # Скорость речи pyttsx3 (слов/мин)

# ── Аудиозапись ────────────────────────────────────────────────────────────

SAMPLE_RATE: int = 16000

# Порог RMS для определения речи.
SILENCE_THRESHOLD: float = 0.015

# Сколько тишины после речи считать концом фразы.
SILENCE_DURATION: float = 0.8

# Максимальная длина одной голосовой команды.
MAX_RECORD_SECONDS: int = 20

# Сколько аудио сохранять перед обнаружением речи,
# чтобы не съедать первые слоги.
PRE_ROLL_SECONDS: float = 0.4

# Минимальная продолжительность обнаруженной речи.
MIN_SPEECH_SECONDS: float = 0.15

# ── Консолидация ───────────────────────────────────────────────────────────
CONSOLIDATION_DAYS: int = 7                     # За сколько дней брать записи для "сна"
SIMILARITY_THRESHOLD: float = 0.75              # Порог косинусной схожести для кластеризации
