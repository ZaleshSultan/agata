"""
Конфигурация Агаты.

Все настройки проекта в одном месте — пути, URL, модели, лимиты.
Импортируй этот модуль везде вместо хардкода констант.
"""

import os
from pathlib import Path


def _env_bool(name: str, default: bool) -> bool:
    """Read a boolean setting without making imports depend on dotenv."""
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().casefold() in {"1", "true", "yes", "on"}

# ── Корневые пути ──────────────────────────────────────────────────────────
BASE_DIR: Path = Path(__file__).resolve().parent
DATA_DIR: Path = BASE_DIR / "data"

# ── Ollama ─────────────────────────────────────────────────────────────────
OLLAMA_URL: str = "http://localhost:11434"
LLM_MODEL: str = "agata"                       # Qwen3-30B-A3B + Modelfile-характер
EMBED_MODEL: str = "nomic-embed-text"           # Модель для эмбеддингов
OLLAMA_KEEP_ALIVE: str = "30m"                  # Время удержания модели в VRAM при простое

# ── Provider selection (extension points; OSINT is intentionally not implemented) ──
LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "ollama")
VISION_PROVIDER: str = os.getenv("VISION_PROVIDER", "")
OSINT_PROVIDER: str = os.getenv("OSINT_PROVIDER", "")

# ── Reusable browser / vision infrastructure ─────────────────────────────
BROWSER_ENABLED: bool = _env_bool("BROWSER_ENABLED", True)
BROWSER_HEADLESS: bool = _env_bool("BROWSER_HEADLESS", True)
BROWSER_TIMEOUT_MS: int = int(os.getenv("BROWSER_TIMEOUT_MS", "15000"))
BROWSER_MAX_PAGES: int = int(os.getenv("BROWSER_MAX_PAGES", "4"))
BROWSER_MAX_EXTRACT_CHARS: int = int(os.getenv("BROWSER_MAX_EXTRACT_CHARS", "5000"))
BROWSER_MAX_LINKS: int = int(os.getenv("BROWSER_MAX_LINKS", "100"))
VISION_ENABLED: bool = _env_bool("VISION_ENABLED", True)

# ── OSINT engine (public, permissioned sources only) ──────────────────────
OSINT_ENABLED: bool = _env_bool("OSINT_ENABLED", True)
OSINT_STORAGE_DIR: Path = Path(os.getenv("OSINT_STORAGE_DIR", str(DATA_DIR / "investigations")))
OSINT_MAX_RUNTIME_SECONDS: float = float(os.getenv("OSINT_MAX_RUNTIME_SECONDS", "45"))
OSINT_MAX_REQUESTS: int = int(os.getenv("OSINT_MAX_REQUESTS", "24"))
OSINT_MAX_ENTITIES: int = int(os.getenv("OSINT_MAX_ENTITIES", "80"))
OSINT_MAX_PAGES: int = int(os.getenv("OSINT_MAX_PAGES", "8"))
OSINT_MAX_DEPTH: int = int(os.getenv("OSINT_MAX_DEPTH", "2"))
OSINT_MAX_PARALLEL_TASKS: int = int(os.getenv("OSINT_MAX_PARALLEL_TASKS", "4"))
OSINT_REQUEST_TIMEOUT_SECONDS: float = float(os.getenv("OSINT_REQUEST_TIMEOUT_SECONDS", "12"))
OSINT_AI_PROVIDER: str = os.getenv("OSINT_AI_PROVIDER", "")
OSINT_AI_MODEL: str = os.getenv("OSINT_AI_MODEL", "")
OSINT_AI_API_KEY: str = os.getenv("OSINT_AI_API_KEY", "")
OSINT_AI_BASE_URL: str = os.getenv("OSINT_AI_BASE_URL", "")
OSINT_AI_FALLBACK_PROVIDER: str = os.getenv("OSINT_AI_FALLBACK_PROVIDER", "")
OSINT_AI_FALLBACK_MODEL: str = os.getenv("OSINT_AI_FALLBACK_MODEL", "")
OSINT_AI_FALLBACK_API_KEY: str = os.getenv("OSINT_AI_FALLBACK_API_KEY", "")
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")

# ── Ассистент ──────────────────────────────────────────────────────────────
ASSISTANT_NAME: str = "Агата"

# ── Память ─────────────────────────────────────────────────────────────────
DB_PATH: Path = DATA_DIR / "agata.db"           # SQLite: дневник + напоминания
WORKING_MEMORY_PATH: Path = DATA_DIR / "working_memory.md"
USER_TIMEZONE: str = "Asia/Almaty"

# Memory v2 retrieval.  Kept here so ranking can be tuned without touching SQL.
MEMORY_RECALL_LIMIT: int = 5
MEMORY_CONTEXT_CHAR_BUDGET: int = 2_500
MEMORY_WORKING_CONTEXT_CHAR_BUDGET: int = 2_500
MEMORY_RECENT_CONVERSATION_TURNS: int = 5
MEMORY_RECENCY_HALF_LIFE_DAYS: float = 90.0
MEMORY_RANK_SIMILARITY_WEIGHT: float = 0.55
MEMORY_RANK_IMPORTANCE_WEIGHT: float = 0.15
MEMORY_RANK_RECENCY_WEIGHT: float = 0.15
MEMORY_RANK_CONFIDENCE_WEIGHT: float = 0.10
MEMORY_RANK_EXPLICIT_WEIGHT: float = 0.05
MEMORY_DUPLICATE_THRESHOLD: float = 0.92
MEMORY_CONFLICT_SIMILARITY_THRESHOLD: float = 0.70
MEMORY_CONFLICT_MIN_CONFIDENCE: float = 0.70
MEMORY_TEMPORARY_TTL_DAYS: int = 7
MEMORY_LLM_EXTRACTION_ENABLED: bool = True
MEMORY_CONSOLIDATION_CLUSTER_THRESHOLD: float = 0.82
MEMORY_CONSOLIDATION_MIN_CLUSTER_SIZE: int = 2
MEMORY_CONSOLIDATION_MAX_EPISODES: int = 30

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
TTS_COOLDOWN_SECONDS: float = 0.5               # Пауза безопасности после озвучивания до включения микрофона
TTS_ENGINE: str = "xtts"                        # Движок TTS: "xtts" | "pyttsx3"
TTS_ENABLED: bool = True                        # Мастер-переключатель TTS
TTS_MIN_SENTENCE_CHARS: int = 20                # Мин. символов для отправки предложения в TTS
TTS_QUEUE_MAX_SIZE: int = 50                    # Макс. предложений в очереди TTS
TTS_DEVICE: str = "auto"                        # "auto" | "cuda" | "cpu"
TTS_LANGUAGE: str = "ru"                        # Язык синтеза для XTTS
TTS_VOICE_PATH: str = str(BASE_DIR / "voices" / "agata_reference.wav")
TTS_XTTS_MODEL: str = "tts_models/multilingual/multi-dataset/xtts_v2"
TTS_FALLBACK_ENGINE: str = "pyttsx3"            # Fallback движок при ошибке primary
TTS_PHRASE_CACHE_ENABLED: bool = True           # RAM-кэширование коротких фраз

# ── Аудиозапись / STT VAD ──────────────────────────────────────────────────

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

# Минимальная общая длительность звучащих кадров (RMS >= SILENCE_THRESHOLD) для защиты от тишины.
MIN_VOICED_AUDIO_SECONDS: float = 0.20

# Метрики отбраковки галлюцинаций / молчания у Whisper:
WHISPER_MAX_NO_SPEECH_PROB: float = 0.45        # Порог no_speech_prob
WHISPER_MIN_AVG_LOGPROB: float = -1.1           # Порог logprob (уверенность)

# ── Консолидация ───────────────────────────────────────────────────────────
CONSOLIDATION_DAYS: int = 7                     # За сколько дней брать записи для "сна"
SIMILARITY_THRESHOLD: float = 0.75              # Порог косинусной схожести для кластеризации
