"""
Инструмент для получения текущих даты и времени.

Использует системные часы Python/Windows с учётом временной зоны Казахстан (Asia/Almaty).
Побочные эффекты: нет.
"""

import logging
from datetime import datetime
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

TIMEZONE_NAME = "Asia/Almaty"

# Названия дней недели на русском
_WEEKDAYS_RU: list[str] = [
    "понедельник", "вторник", "среда", "четверг",
    "пятница", "суббота", "воскресенье",
]

# Названия месяцев на русском
_MONTHS_RU: list[str] = [
    "января", "февраля", "марта", "апреля",
    "мая", "июня", "июля", "августа",
    "сентября", "октября", "ноября", "декабря",
]


def get_current_datetime() -> str:
    """
    Получить текущую дату, время и день недели в часовом поясе Asia/Almaty.

    Returns:
        Строка вида: "Сейчас Понедельник, 10 августа 2026, 17:25."
    """
    try:
        tz = ZoneInfo(TIMEZONE_NAME)
        now = datetime.now(tz)
    except Exception:
        now = datetime.now()

    weekday = _WEEKDAYS_RU[now.weekday()].capitalize()
    month = _MONTHS_RU[now.month - 1]
    formatted = f"{weekday}, {now.day} {month} {now.year}, {now.strftime('%H:%M')}"
    logger.info("[DATETIME] now='%s' timezone='%s'", formatted, TIMEZONE_NAME)
    return f"Сейчас {formatted}."


# ── JSON Schema для tool calling ──────────────────────────────────────────

TOOL_DEFINITION: dict = {
    "function": {
        "name": "get_current_datetime",
        "description": (
            "Получить текущую дату, время и день недели в реальном времени. "
            "Вызывай всегда, когда пользователь спрашивает дату, время, день недели "
            "или относительные даты (завтра, в понедельник, через час и т.д.)."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    "requires_confirmation": False,
}
