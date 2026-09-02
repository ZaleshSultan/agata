"""
Инструмент напоминаний.

SQLite-таблица reminders(id, title, due_datetime, is_done, datetime_iso, created_at, done).
create_reminder — создаёт напоминание (с подтверждением пользователя).
list_reminders — показывает активные напоминания (read-only).

Побочные эффекты: чтение/запись SQLite-базы data/agata.db.
"""

import logging
import sqlite3
from datetime import datetime

import config

logger = logging.getLogger(__name__)


def _reminders_table_exists(conn: sqlite3.Connection) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'reminders'"
    ).fetchone() is not None


def _create_reminders_table(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE reminders (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            title        TEXT NOT NULL,
            due_datetime TEXT NOT NULL,
            is_done      INTEGER DEFAULT 0,
            datetime_iso TEXT,
            created_at   TEXT NOT NULL DEFAULT (datetime('now')),
            done         INTEGER DEFAULT 0
        )
    """)


def _get_column_info(conn: sqlite3.Connection) -> dict[str, sqlite3.Row]:
    old_factory = conn.row_factory
    conn.row_factory = sqlite3.Row
    try:
        return {row["name"]: row for row in conn.execute("PRAGMA table_info(reminders)").fetchall()}
    finally:
        conn.row_factory = old_factory


def _get_columns(conn: sqlite3.Connection) -> set[str]:
    return {row[1] for row in conn.execute("PRAGMA table_info(reminders)").fetchall()}


def _select_existing(columns: set[str], name: str, fallback: str) -> str:
    return name if name in columns else fallback


def _rebuild_reminders_table(conn: sqlite3.Connection, columns: set[str]) -> None:
    legacy_table = "reminders_legacy_migration"
    conn.execute(f"DROP TABLE IF EXISTS {legacy_table}")
    conn.execute(f"ALTER TABLE reminders RENAME TO {legacy_table}")
    _create_reminders_table(conn)

    title_expr = _select_existing(columns, "title", "'Напоминание'")
    due_expr = "COALESCE(" + ", ".join(
        expr for expr in (
            _select_existing(columns, "due_datetime", "NULL"),
            _select_existing(columns, "datetime_iso", "NULL"),
            _select_existing(columns, "created_at", "NULL"),
            "datetime('now')",
        )
    ) + ")"
    done_expr = "COALESCE(" + ", ".join(
        expr for expr in (
            _select_existing(columns, "is_done", "NULL"),
            _select_existing(columns, "done", "NULL"),
            "0",
        )
    ) + ")"
    datetime_iso_expr = _select_existing(columns, "datetime_iso", due_expr)
    created_at_expr = "COALESCE(" + _select_existing(columns, "created_at", "NULL") + ", datetime('now'))"
    id_expr = _select_existing(columns, "id", "NULL")

    conn.execute(
        f"""
        INSERT INTO reminders (id, title, due_datetime, is_done, datetime_iso, created_at, done)
        SELECT {id_expr}, {title_expr}, {due_expr}, {done_expr}, {datetime_iso_expr}, {created_at_expr}, {done_expr}
        FROM {legacy_table}
        """
    )
    conn.execute(f"DROP TABLE {legacy_table}")


def _ensure_reminders_schema(conn: sqlite3.Connection) -> None:
    """Create or migrate reminders so scheduler and tool code share one table."""
    if not _reminders_table_exists(conn):
        _create_reminders_table(conn)
        conn.commit()
        return

    # Existing installations used datetime_iso/done. Add the scheduler-facing
    # names without dropping old columns that other code may still reference.
    column_info = _get_column_info(conn)
    datetime_iso_info = column_info.get("datetime_iso")
    datetime_iso_required = datetime_iso_info is not None and bool(datetime_iso_info["notnull"])
    if "due_datetime" not in column_info or datetime_iso_required:
        _rebuild_reminders_table(conn, set(column_info))
        conn.commit()
        return

    columns = _get_columns(conn)
    migrations = {
        "due_datetime": "ALTER TABLE reminders ADD COLUMN due_datetime TEXT",
        "is_done": "ALTER TABLE reminders ADD COLUMN is_done INTEGER DEFAULT 0",
        "datetime_iso": "ALTER TABLE reminders ADD COLUMN datetime_iso TEXT",
        "created_at": "ALTER TABLE reminders ADD COLUMN created_at TEXT",
        "done": "ALTER TABLE reminders ADD COLUMN done INTEGER DEFAULT 0",
    }
    for column, statement in migrations.items():
        if column not in columns:
            conn.execute(statement)

    columns = _get_columns(conn)
    if {"due_datetime", "datetime_iso"}.issubset(columns):
        conn.execute(
            "UPDATE reminders SET due_datetime = datetime_iso "
            "WHERE due_datetime IS NULL AND datetime_iso IS NOT NULL"
        )
        conn.execute(
            "UPDATE reminders SET datetime_iso = due_datetime "
            "WHERE datetime_iso IS NULL AND due_datetime IS NOT NULL"
        )
    if {"is_done", "done"}.issubset(columns):
        conn.execute("UPDATE reminders SET is_done = done WHERE is_done IS NULL AND done IS NOT NULL")
        conn.execute("UPDATE reminders SET done = is_done WHERE done IS NULL AND is_done IS NOT NULL")
    if "created_at" in columns:
        conn.execute("UPDATE reminders SET created_at = datetime('now') WHERE created_at IS NULL")
    conn.commit()


def ensure_reminders_table() -> None:
    """Ensure the reminders table exists for startup code and background schedulers."""
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(config.DB_PATH))
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        _ensure_reminders_schema(conn)
    finally:
        conn.close()


def _get_connection() -> sqlite3.Connection:
    """Получить соединение и создать таблицу, если её нет."""
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(config.DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")
    _ensure_reminders_schema(conn)
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
            "INSERT INTO reminders (title, due_datetime, datetime_iso, is_done, done) VALUES (?, ?, ?, 0, 0)",
            (title, datetime_iso, datetime_iso),
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
            """
            SELECT id, title, COALESCE(due_datetime, datetime_iso) AS reminder_datetime
            FROM reminders
            WHERE COALESCE(is_done, done, 0) = 0
            ORDER BY reminder_datetime
            """
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
