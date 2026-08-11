"""
Дневник Агаты — долгосрочная память.

SQLite-таблица diary(id, text, embedding BLOB, created_at, tags, consolidated).
Семантический поиск через косинусную схожесть numpy-эмбеддингов.
Эмбеддинги получаются через Ollama nomic-embed-text.

Побочные эффекты: чтение/запись SQLite-базы data/agata.db.
"""

import json
import logging
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np

import config

logger = logging.getLogger(__name__)


def _get_connection() -> sqlite3.Connection:
    """Получить соединение с SQLite и создать таблицу, если её нет."""
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(config.DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS diary (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            text        TEXT    NOT NULL,
            embedding   BLOB,
            created_at  TEXT    NOT NULL DEFAULT (datetime('now')),
            tags        TEXT    DEFAULT '',
            consolidated INTEGER DEFAULT 0
        )
    """)
    conn.commit()
    return conn


def _serialize_embedding(emb: list[float]) -> bytes:
    """Сериализовать эмбеддинг в bytes для BLOB-колонки."""
    return np.array(emb, dtype=np.float32).tobytes()


def _deserialize_embedding(blob: bytes) -> np.ndarray:
    """Десериализовать BLOB обратно в numpy-массив."""
    return np.frombuffer(blob, dtype=np.float32)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Косинусная схожесть между двумя векторами."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def add_entry(text: str, embedding: list[float], tags: str = "") -> int:
    """
    Добавить запись в дневник.

    Args:
        text: Текст записи.
        embedding: Эмбеддинг текста (list[float]).
        tags: Теги через запятую (опционально).

    Returns:
        id новой записи.
    """
    conn = _get_connection()
    try:
        cursor = conn.execute(
            "INSERT INTO diary (text, embedding, tags) VALUES (?, ?, ?)",
            (text, _serialize_embedding(embedding), tags),
        )
        conn.commit()
        entry_id: int = cursor.lastrowid  # type: ignore[assignment]
        logger.info("Запись добавлена в дневник: id=%d, len=%d", entry_id, len(text))
        return entry_id
    finally:
        conn.close()


def search(query_embedding: list[float], top_k: int = 5) -> list[dict]:
    """
    Семантический поиск по дневнику.

    Args:
        query_embedding: Эмбеддинг поискового запроса.
        top_k: Сколько ближайших записей вернуть.

    Returns:
        Список dict с ключами: id, text, created_at, tags, similarity.
    """
    conn = _get_connection()
    try:
        rows = conn.execute(
            "SELECT id, text, embedding, created_at, tags FROM diary"
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        return []

    query_vec = np.array(query_embedding, dtype=np.float32)
    results: list[tuple[float, dict]] = []

    for row_id, text, emb_blob, created_at, tags in rows:
        if emb_blob is None:
            continue
        entry_vec = _deserialize_embedding(emb_blob)
        sim = _cosine_similarity(query_vec, entry_vec)
        results.append((sim, {
            "id": row_id,
            "text": text,
            "created_at": created_at,
            "tags": tags,
            "similarity": round(sim, 4),
        }))

    # Сортировка по убыванию схожести
    results.sort(key=lambda x: x[0], reverse=True)
    return [item for _, item in results[:top_k]]


def all_entries_since(days: int = 7) -> list[dict]:
    """
    Получить все неконсолидированные записи за последние N дней.

    Args:
        days: За сколько дней назад брать записи.

    Returns:
        Список dict с ключами: id, text, embedding, created_at, tags.
    """
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    conn = _get_connection()
    try:
        rows = conn.execute(
            """SELECT id, text, embedding, created_at, tags
               FROM diary
               WHERE consolidated = 0 AND created_at >= ?
               ORDER BY created_at""",
            (cutoff,),
        ).fetchall()
    finally:
        conn.close()

    entries: list[dict] = []
    for row_id, text, emb_blob, created_at, tags in rows:
        entry: dict = {
            "id": row_id,
            "text": text,
            "created_at": created_at,
            "tags": tags,
        }
        if emb_blob is not None:
            entry["embedding"] = _deserialize_embedding(emb_blob)
        entries.append(entry)
    return entries


def mark_consolidated(ids: list[int]) -> None:
    """Пометить записи как консолидированные (обработаны во время "сна")."""
    if not ids:
        return
    conn = _get_connection()
    try:
        placeholders = ",".join("?" for _ in ids)
        conn.execute(
            f"UPDATE diary SET consolidated = 1 WHERE id IN ({placeholders})",
            ids,
        )
        conn.commit()
        logger.info("Помечены как консолидированные: %d записей", len(ids))
    finally:
        conn.close()


def count_entries() -> int:
    """Общее количество записей в дневнике (для диагностики)."""
    conn = _get_connection()
    try:
        row = conn.execute("SELECT COUNT(*) FROM diary").fetchone()
        return row[0] if row else 0
    finally:
        conn.close()
