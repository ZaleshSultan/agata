"""
Ограниченный системный контроль: браузер, белый список приложений,
съёмные диски. НЕ включает произвольное выполнение команд или
доступ к файловой системе за пределами явно описанных функций.
"""

import logging
import shutil
import string
import subprocess
import webbrowser
from pathlib import Path

import config

logger = logging.getLogger(__name__)

# Белый список приложений — добавляй сюда вручную, не через параметр модели
ALLOWED_APPS = {
    "chrome": r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    "notepad": "notepad.exe",
    "explorer": "explorer.exe",
    # Добавь свои пути к нужным приложениям
}


def open_browser(url: str) -> str:
    """
    Открывает URL в браузере по умолчанию.
    url: полный адрес, например https://youtube.com
    """
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    try:
        webbrowser.open(url)
        return f"Открыла {url} в браузере."
    except Exception as e:
        logger.error(f"Ошибка открытия браузера: {e}")
        return f"Не удалось открыть браузер: {e}"


def open_app(app_name: str) -> str:
    """
    Запускает приложение из белого списка (chrome, notepad, explorer).
    Приложений вне списка запустить нельзя — это осознанное ограничение.
    """
    app_name = app_name.lower().strip()
    if app_name not in ALLOWED_APPS:
        return f"Приложение '{app_name}' не в белом списке. Доступны: {', '.join(ALLOWED_APPS.keys())}"
    try:
        subprocess.Popen(ALLOWED_APPS[app_name])
        return f"Запустила {app_name}."
    except Exception as e:
        logger.error(f"Ошибка запуска приложения {app_name}: {e}")
        return f"Не удалось запустить {app_name}: {e}"


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
        "description": "Запустить разрешённую программу из белого списка (chrome, notepad, explorer).",
        "parameters": {
            "type": "object",
            "properties": {
                "app_name": {
                    "type": "string",
                    "description": "Имя программы для запуска из белого списка (например, 'chrome', 'notepad', 'explorer').",
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
