"""
Инструмент напоминаний.

SQLite-таблица reminders(id, title, datetime_iso, created_at, done).
create_reminder — создаёт напоминание (с подтверждением пользователя).
list_reminders — показывает активные напоминания (read-only).

Побочные эффекты: чтение/запись SQLite-базы data/agata.db.
"""

import logging
import sqlite3
from datetime import datetime

import config

logger = logging.getLogger(__name__)


def _get_connection() -> sqlite3.Connection:
    """Получить соединение и создать таблицу, если её нет."""
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(config.DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS reminders (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            title       TEXT    NOT NULL,
            datetime_iso TEXT   NOT NULL,
            created_at  TEXT    NOT NULL DEFAULT (datetime('now')),
            done        INTEGER DEFAULT 0
        )
    """)
    conn.commit()
    return conn


def create_reminder(title: str, datetime_iso: str) -> str:
    """
    Создать напоминание.

    Args:
        title: Описание напоминания.
        datetime_iso: Дата и время в формате ISO 8601 (например, "2026-08-10T12:00").

    Returns:
        Подтверждение создания.
    """
    conn = _get_connection()
    try:
        cursor = conn.execute(
            "INSERT INTO reminders (title, datetime_iso) VALUES (?, ?)",
            (title, datetime_iso),
        )
        conn.commit()
        reminder_id: int = cursor.lastrowid  # type: ignore[assignment]
        result = f"Напоминание создано (id={reminder_id}): «{title}» на {datetime_iso}"
        logger.info(result)
        return result
    except Exception as exc:
        error_msg = f"Ошибка создания напоминания: {exc}"
        logger.error(error_msg)
        return error_msg
    finally:
        conn.close()


def list_reminders() -> str:
    """
    Показать все активные (невыполненные) напоминания.

    Returns:
        Список напоминаний или "нет активных напоминаний".
    """
    conn = _get_connection()
    try:
        rows = conn.execute(
            "SELECT id, title, datetime_iso FROM reminders WHERE done = 0 ORDER BY datetime_iso"
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        return "Нет активных напоминаний."

    lines: list[str] = []
    for row_id, title, dt_iso in rows:
        lines.append(f"• [{dt_iso}] {title} (id={row_id})")
    result = "\n".join(lines)
    logger.info("Активных напоминаний: %d", len(rows))
    return result


# ── JSON Schema для tool calling ──────────────────────────────────────────

CREATE_REMINDER_DEFINITION: dict = {
    "function": {
        "name": "create_reminder",
        "description": (
            "Создать напоминание на определённую дату и время. "
            "ВАЖНО: перед использованием вызови get_current_datetime, "
            "чтобы правильно вычислить абсолютную дату из относительной "
            "(\"завтра\", \"в понедельник\", \"через 2 часа\")."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Описание напоминания, например 'Поликлиника'.",
                },
                "datetime_iso": {
                    "type": "string",
                    "description": "Дата и время в формате ISO 8601, например '2026-08-12T14:00'.",
                },
            },
            "required": ["title", "datetime_iso"],
        },
    },
    "requires_confirmation": True,
}

LIST_REMINDERS_DEFINITION: dict = {
    "function": {
        "name": "list_reminders",
        "description": "Показать все активные (невыполненные) напоминания пользователя.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    "requires_confirmation": False,
}
