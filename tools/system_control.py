"""
Ограниченный системный контроль: браузер, белый список приложений,
съёмные диски. НЕ включает произвольное выполнение команд или
доступ к файловой системе за пределами явно описанных функций.
"""

import json
import logging
import os
import shutil
import string
import subprocess
import webbrowser
from pathlib import Path

import config

logger = logging.getLogger(__name__)

# Белый список приложений — добавляй сюда вручную, не через параметр модели
ALLOWED_APPS = {
    "chrome": (
        r"%ProgramFiles%\Google\Chrome\Application\chrome.exe",
        r"%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe",
        r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe",
    ),
    "vscode": (
        r"%LOCALAPPDATA%\Programs\Microsoft VS Code\Code.exe",
        r"%ProgramFiles%\Microsoft VS Code\Code.exe",
    ),
    "notepad": ("notepad.exe",),
    "explorer": ("explorer.exe",),
    # Добавь свои пути к нужным приложениям
}


def _tool_result(success: bool, **details: str) -> str:
    """Serialize an execution result so callers can treat it as authority."""
    return json.dumps({"success": success, **details}, ensure_ascii=False)


def _resolve_allowed_app(app_name: str) -> str | None:
    """Resolve only known application candidates; never accept user paths."""
    for candidate in ALLOWED_APPS[app_name]:
        expanded = os.path.expandvars(candidate)
        # Bare Windows system commands are intentionally whitelisted above.
        if not Path(expanded).is_absolute() or Path(expanded).is_file():
            return expanded
    return None


def open_browser(url: str) -> str:
    """
    Открывает URL в браузере по умолчанию.
    url: полный адрес, например https://youtube.com
    """
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    try:
        opened = webbrowser.open(url)
        if not opened:
            return _tool_result(False, url=url, error="Браузер не подтвердил открытие URL.")
        return _tool_result(True, url=url)
    except Exception as e:
        logger.error(f"Ошибка открытия браузера: {e}")
        return _tool_result(False, url=url, error=f"Не удалось открыть браузер: {e}")


def open_app(app_name: str) -> str:
    """
    Запускает приложение из белого списка (chrome, notepad, explorer).
    Приложений вне списка запустить нельзя — это осознанное ограничение.
    """
    app_name = app_name.lower().strip()
    if app_name not in ALLOWED_APPS:
        return f"Приложение '{app_name}' не в белом списке. Доступны: {', '.join(ALLOWED_APPS.keys())}"
    command = _resolve_allowed_app(app_name)
    if command is None:
        error = f"Исполняемый файл для '{app_name}' не найден в разрешённых путях."
        logger.error(error)
        return _tool_result(False, application=app_name, error=error)
    try:
        subprocess.Popen(command)
        return _tool_result(True, application=app_name)
    except Exception as e:
        logger.error(f"Ошибка запуска приложения {app_name}: {e}")
        return _tool_result(False, application=app_name, error=f"Не удалось запустить {app_name}: {e}")


def list_removable_drives() -> str:
    """
    Находит подключённые съёмные диски (флешки) в системе.
    """
    drives = []
    for letter in string.ascii_uppercase:
        drive_path = f"{letter}:\\"
        if Path(drive_path).exists():
            try:
                usage = shutil.disk_usage(drive_path)
                # Простая эвристика: съёмные диски обычно небольшого объёма
                # Более надёжно — через win32api, но это упрощённый MVP-вариант
                drives.append(f"{drive_path} (свободно: {usage.free // (1024**3)} ГБ)")
            except Exception:
                continue
    if not drives:
        return "Съёмных дисков не найдено."
    return "Найдены диски: " + ", ".join(drives)


def copy_project_snapshot(destination_drive: str) -> str:
    """
    Копирует текущую папку проекта (BASE_DIR) на указанный диск.
    destination_drive: буква диска, например "E:\\"
    requires_confirmation = True — обязательно подтверждение перед копированием.
    """
    dest = Path(destination_drive) / "agata_snapshot"
    try:
        if not Path(destination_drive).exists():
            return f"Диск {destination_drive} не найден."
        shutil.copytree(
            config.BASE_DIR,
            dest,
            ignore=shutil.ignore_patterns(".git", "__pycache__", "*.db", "data"),
            dirs_exist_ok=True,
        )
        return f"Проект скопирован в {dest}."
    except Exception as e:
        logger.error(f"Ошибка копирования проекта: {e}")
        return f"Не удалось скопировать проект: {e}"


# ── JSON Schema для tool calling ──────────────────────────────────────────

OPEN_BROWSER_DEFINITION: dict = {
    "function": {
        "name": "open_browser",
        "description": "Открыть указанный сайт/URL в веб-браузере по умолчанию.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "Веб-адрес для открытия, например 'https://youtube.com' или 'google.com'.",
                },
            },
            "required": ["url"],
        },
    },
    "requires_confirmation": False,
}

OPEN_APP_DEFINITION: dict = {
    "function": {
        "name": "open_app",
        "description": "Запустить разрешённую программу из белого списка (chrome, vscode, notepad, explorer).",
        "parameters": {
            "type": "object",
            "properties": {
                "app_name": {
                    "type": "string",
                    "description": "Имя программы для запуска из белого списка (например, 'chrome', 'vscode', 'notepad', 'explorer').",
                },
            },
            "required": ["app_name"],
        },
    },
    "requires_confirmation": False,
}

LIST_REMOVABLE_DRIVES_DEFINITION: dict = {
    "function": {
        "name": "list_removable_drives",
        "description": "Получить список доступных дисков и флешек в системе.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    "requires_confirmation": False,
}

COPY_PROJECT_SNAPSHOT_DEFINITION: dict = {
    "function": {
        "name": "copy_project_snapshot",
        "description": "Скопировать снимок проекта Агаты на указанный целевой диск (например, 'E:\\').",
        "parameters": {
            "type": "object",
            "properties": {
                "destination_drive": {
                    "type": "string",
                    "description": "Буква целевого диска, например 'E:\\' или 'F:\\'.",
                },
            },
            "required": ["destination_drive"],
        },
    },
    "requires_confirmation": True,
}
