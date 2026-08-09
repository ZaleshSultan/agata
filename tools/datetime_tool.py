"""
Инструмент для получения текущих даты и времени.

Обязателен для корректной обработки относительных дат
("завтра", "в понедельник", "через 2 часа").
Побочные эффекты: нет.
"""

import logging
from datetime import datetime

logger = logging.getLogger(__name__)

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
    Получить текущую дату, время и день недели.

    Returns:
        Строка вида: "Суббота, 9 августа 2026, 12:41 (UTC+5)"
    """
    now = datetime.now()
    weekday = _WEEKDAYS_RU[now.weekday()].capitalize()
    month = _MONTHS_RU[now.month - 1]
    result = f"{weekday}, {now.day} {month} {now.year}, {now.strftime('%H:%M')}"
    logger.info("Текущая дата/время: %s", result)
    return result


# ── JSON Schema для tool calling ──────────────────────────────────────────

TOOL_DEFINITION: dict = {
    "function": {
        "name": "get_current_datetime",
        "description": (
            "Получить текущую дату, время и день недели. "
            "Вызывай ПЕРЕД созданием напоминаний с относительными датами "
            "(завтра, в понедельник, через час и т.д.)."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    "requires_confirmation": False,
}
