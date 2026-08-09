"""
Реестр инструментов Агаты.

Централизованное место для регистрации всех tool definitions.
Функции:
- get_tool_schemas() — вернуть JSON Schema для Ollama /api/chat
- execute_tool(name, arguments) — выполнить инструмент по имени

Чтобы добавить новый инструмент:
1. Создай файл tools/my_tool.py с функцией и TOOL_DEFINITION.
2. Зарегистрируй его в _REGISTRY ниже.
"""

import json
import logging
from typing import Any, Callable

from tools import calc
from tools import datetime_tool
from tools import memory as memory_tools
from tools import reminder
from tools import system_control
from tools import web_search

logger = logging.getLogger(__name__)


# ── Реестр инструментов ───────────────────────────────────────────────────
# Формат: name -> {"function": callable, "schema": dict, "requires_confirmation": bool}

_REGISTRY: dict[str, dict[str, Any]] = {
    # Калькулятор
    "calculate": {
        "function": calc.calculate,
        "schema": calc.TOOL_DEFINITION,
        "requires_confirmation": calc.TOOL_DEFINITION["requires_confirmation"],
    },
    # Дата/время
    "get_current_datetime": {
        "function": lambda **_: datetime_tool.get_current_datetime(),
        "schema": datetime_tool.TOOL_DEFINITION,
        "requires_confirmation": datetime_tool.TOOL_DEFINITION["requires_confirmation"],
    },
    # Дневник — запись
    "write_diary": {
        "function": memory_tools.write_diary,
        "schema": memory_tools.WRITE_DIARY_DEFINITION,
        "requires_confirmation": memory_tools.WRITE_DIARY_DEFINITION["requires_confirmation"],
    },
    # Дневник — поиск
    "search_diary": {
        "function": memory_tools.search_diary,
        "schema": memory_tools.SEARCH_DIARY_DEFINITION,
        "requires_confirmation": memory_tools.SEARCH_DIARY_DEFINITION["requires_confirmation"],
    },
    # Напоминания — создать
    "create_reminder": {
        "function": reminder.create_reminder,
        "schema": reminder.CREATE_REMINDER_DEFINITION,
        "requires_confirmation": reminder.CREATE_REMINDER_DEFINITION["requires_confirmation"],
    },
    # Напоминания — список
    "list_reminders": {
        "function": lambda **_: reminder.list_reminders(),
        "schema": reminder.LIST_REMINDERS_DEFINITION,
        "requires_confirmation": reminder.LIST_REMINDERS_DEFINITION["requires_confirmation"],
    },
    # Веб-поиск
    "web_search": {
        "function": web_search.web_search,
        "schema": web_search.TOOL_DEFINITION,
        "requires_confirmation": web_search.TOOL_DEFINITION["requires_confirmation"],
    },
    # Системный контроль — браузер
    "open_browser": {
        "function": system_control.open_browser,
        "schema": system_control.OPEN_BROWSER_DEFINITION,
        "requires_confirmation": system_control.OPEN_BROWSER_DEFINITION["requires_confirmation"],
    },
    # Системный контроль — приложение из белого списка
    "open_app": {
        "function": system_control.open_app,
        "schema": system_control.OPEN_APP_DEFINITION,
        "requires_confirmation": system_control.OPEN_APP_DEFINITION["requires_confirmation"],
    },
    # Системный контроль — список съёмных дисков
    "list_removable_drives": {
        "function": lambda **_: system_control.list_removable_drives(),
        "schema": system_control.LIST_REMOVABLE_DRIVES_DEFINITION,
        "requires_confirmation": system_control.LIST_REMOVABLE_DRIVES_DEFINITION["requires_confirmation"],
    },
    # Системный контроль — копирование снимка проекта
    "copy_project_snapshot": {
        "function": system_control.copy_project_snapshot,
        "schema": system_control.COPY_PROJECT_SNAPSHOT_DEFINITION,
        "requires_confirmation": system_control.COPY_PROJECT_SNAPSHOT_DEFINITION["requires_confirmation"],
    },
}



def get_tool_schemas() -> list[dict]:
    """
    Получить JSON Schema всех инструментов для передачи в Ollama.

    Returns:
        Список tool definitions в формате Ollama API.
    """
    schemas: list[dict] = []
    for tool_info in _REGISTRY.values():
        schemas.append({
            "type": "function",
            "function": tool_info["schema"]["function"],
        })
    return schemas


def execute_tool(name: str, arguments: dict[str, Any]) -> str:
    """
    Выполнить инструмент по имени.

    Args:
        name: Имя инструмента (должно быть в реестре).
        arguments: Аргументы для функции.

    Returns:
        Результат выполнения (строка).

    Raises:
        KeyError: Инструмент не найден в реестре.
    """
    if name not in _REGISTRY:
        available = ", ".join(_REGISTRY.keys())
        error_msg = f"Неизвестный инструмент '{name}'. Доступные: {available}"
        logger.error(error_msg)
        return error_msg

    tool = _REGISTRY[name]
    fn: Callable = tool["function"]

    try:
        result = fn(**arguments)
        return str(result)
    except TypeError as exc:
        # Неправильные аргументы
        error_msg = f"Ошибка аргументов для '{name}': {exc}"
        logger.error(error_msg)
        return error_msg
    except Exception as exc:
        # Любая другая ошибка — не роняем программу
        error_msg = f"Ошибка при выполнении '{name}': {exc}"
        logger.error(error_msg, exc_info=True)
        return error_msg


def requires_confirmation(name: str) -> bool:
    """Требует ли инструмент подтверждения пользователя перед выполнением."""
    tool = _REGISTRY.get(name)
    if tool is None:
        return False
    return tool["requires_confirmation"]


def list_tool_names() -> list[str]:
    """Список имён всех зарегистрированных инструментов."""
    return list(_REGISTRY.keys())
