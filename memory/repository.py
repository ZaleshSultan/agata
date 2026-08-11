"""SQLite persistence and migration for Memory v2."""

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Iterable
from zoneinfo import ZoneInfo

import numpy as np

import config
from memory.models import Memory, MemorySource, MemoryStatus, MemoryType


def user_timezone():
    """Configured timezone with a Windows-safe fallback when tzdata is absent."""
    try:
        return ZoneInfo(config.USER_TIMEZONE)
    except Exception:
        # Kazakhstan uses UTC+05:00.  This keeps all new timestamps aware even
        # in Python environments without the IANA timezone database.
        return timezone(timedelta(hours=5), config.USER_TIMEZONE)


def _now() -> str:
    return datetime.now(user_timezone()).isoformat()


def _pack(embedding: list[float] | None) -> bytes | None:
    return None if embedding is None else np.asarray(embedding, dtype=np.float32).tobytes()


def _unpack(blob: bytes | None) -> list[float] | None:
    return None if blob is None else np.frombuffer(blob, dtype=np.float32).tolist()


class MemoryRepository:
    def __init__(self, db_path=None) -> None:
        self.db_path = db_path or config.DB_PATH
        self.initialize()

    def _connect(self) -> sqlite3.Connection:
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    @contextmanager
    def _connection(self):
        conn = self._connect()
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def initialize(self) -> None:
        with self._connection() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content TEXT NOT NULL,
                    memory_type TEXT NOT NULL,
                    importance REAL NOT NULL,
                    confidence REAL NOT NULL,
                    status TEXT NOT NULL,
                    source TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    last_recalled_at TEXT,
                    expires_at TEXT,
                    consolidated_at TEXT,
                    supersedes_id INTEGER,
                    embedding BLOB,
                    metadata TEXT NOT NULL DEFAULT '{}',
                    FOREIGN KEY(supersedes_id) REFERENCES memories(id)
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_status ON memories(status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at)")
            self._migrate_diary(conn)

    def _migrate_diary(self, conn: sqlite3.Connection) -> None:
        exists = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='diary'").fetchone()
        if not exists:
            return
        cols = {row[1] for row in conn.execute("PRAGMA table_info(diary)")}
        required = {"id", "text", "embedding", "created_at", "tags", "consolidated"}
        if not required.issubset(cols):
            return
        rows = conn.execute("SELECT id, text, embedding, created_at, tags, consolidated FROM diary").fetchall()
        for row in rows:
            # Negative ids prevent collisions while retaining a stable legacy reference.
            legacy_id = -int(row["id"])
            already = conn.execute("SELECT 1 FROM memories WHERE id = ?", (legacy_id,)).fetchone()
            if already:
                continue
            created = row["created_at"] or _now()
            conn.execute("""INSERT INTO memories
                (id, content, memory_type, importance, confidence, status, source,
                 created_at, updated_at, consolidated_at, embedding, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", (
                legacy_id, row["text"], MemoryType.EVENT.value, 0.5, 0.7,
                MemoryStatus.ACTIVE.value, MemorySource.IMPORT.value, created, created,
                created if row["consolidated"] else None, row["embedding"],
                json.dumps({"legacy_diary_id": row["id"], "tags": row["tags"] or ""}),
            ))

    @staticmethod
    def _row(row: sqlite3.Row) -> Memory:
        return Memory(
            id=row["id"], content=row["content"], memory_type=MemoryType(row["memory_type"]),
            importance=row["importance"], confidence=row["confidence"], status=MemoryStatus(row["status"]),
            source=MemorySource(row["source"]), created_at=row["created_at"], updated_at=row["updated_at"],
            last_recalled_at=row["last_recalled_at"], expires_at=row["expires_at"],
            consolidated_at=row["consolidated_at"], supersedes_id=row["supersedes_id"],
            embedding=_unpack(row["embedding"]), metadata=json.loads(row["metadata"] or "{}"),
        )

    def create(self, memory: Memory) -> Memory:
        now = _now()
        with self._connection() as conn:
            cursor = conn.execute("""INSERT INTO memories
                (content,memory_type,importance,confidence,status,source,created_at,updated_at,
                 last_recalled_at,expires_at,consolidated_at,supersedes_id,embedding,metadata)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", (
                memory.content, memory.memory_type.value, memory.importance, memory.confidence,
                memory.status.value, memory.source.value, memory.created_at or now, memory.updated_at or now,
                memory.last_recalled_at, memory.expires_at, memory.consolidated_at, memory.supersedes_id,
                _pack(memory.embedding), json.dumps(memory.metadata, ensure_ascii=False),
            ))
            memory.id = cursor.lastrowid
            memory.created_at = memory.created_at or now
            memory.updated_at = memory.updated_at or now
        return memory

    def get(self, memory_id: int) -> Memory | None:
        with self._connection() as conn:
            row = conn.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
        return self._row(row) if row else None

    def list_active(self) -> list[Memory]:
        now = _now()
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT * FROM memories WHERE status = ? AND (expires_at IS NULL OR expires_at > ?)",
                (MemoryStatus.ACTIVE.value, now),
            ).fetchall()
        return [self._row(row) for row in rows]

    def update(self, memory: Memory) -> Memory:
        if memory.id is None:
            raise ValueError("memory id is required for update")
        memory.updated_at = _now()
        with self._connection() as conn:
            conn.execute("""UPDATE memories SET content=?, memory_type=?, importance=?, confidence=?, status=?, source=?,
                updated_at=?, last_recalled_at=?, expires_at=?, consolidated_at=?, supersedes_id=?, embedding=?, metadata=?
                WHERE id=?""", (
                memory.content, memory.memory_type.value, memory.importance, memory.confidence, memory.status.value,
                memory.source.value, memory.updated_at, memory.last_recalled_at, memory.expires_at,
                memory.consolidated_at, memory.supersedes_id, _pack(memory.embedding),
                json.dumps(memory.metadata, ensure_ascii=False), memory.id,
            ))
        return memory

    def mark_recalled(self, ids: Iterable[int]) -> None:
        values = list(ids)
        if not values:
            return
        with self._connection() as conn:
            conn.executemany("UPDATE memories SET last_recalled_at=?, updated_at=? WHERE id=?", [(_now(), _now(), i) for i in values])

    def mark_consolidated(self, ids: Iterable[int]) -> None:
        values = list(ids)
        if values:
            with self._connection() as conn:
                conn.executemany("UPDATE memories SET consolidated_at=?, updated_at=? WHERE id=?", [(_now(), _now(), i) for i in values])
