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
from typing import TypeAlias

import config

logger = logging.getLogger(__name__)

# Белый список приложений — добавляй сюда вручную, не через параметр модели.
# Каждая команда полностью статична: аргументы пользователя сюда не попадают.
AppCommand: TypeAlias = tuple[str, ...]
ALLOWED_APPS: dict[str, tuple[AppCommand, ...]] = {
    "chrome": (
        (r"%ProgramFiles%\Google\Chrome\Application\chrome.exe",),
        (r"%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe",),
        (r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe",),
    ),
    "vscode": (
        (r"%LOCALAPPDATA%\Programs\Microsoft VS Code\Code.exe",),
        (r"%ProgramFiles%\Microsoft VS Code\Code.exe",),
    ),
    "discord": (
        (r"%LOCALAPPDATA%\Discord\Update.exe", "--processStart", "Discord.exe"),
    ),
    "telegram": (
        (r"%APPDATA%\Telegram Desktop\Telegram.exe",),
        (r"%LOCALAPPDATA%\Programs\Telegram Desktop\Telegram.exe",),
    ),
    "notepad": (("notepad.exe",),),
    "explorer": (("explorer.exe",),),
    # Добавь свои пути к нужным приложениям
}

_PROCESS_NAMES = {
    "chrome": "chrome.exe",
    "google chrome": "chrome.exe",
    "vscode": "Code.exe",
    "vs code": "Code.exe",
    "visual studio code": "Code.exe",
    "code": "Code.exe",
    "discord": "Discord.exe",
    "telegram": "Telegram.exe",
    "notepad": "notepad.exe",
    "explorer": "explorer.exe",
}


def _tool_result(success: bool, **details: str) -> str:
    """Serialize an execution result so callers can treat it as authority."""
    return json.dumps({"success": success, **details}, ensure_ascii=False)


def _resolve_allowed_app(app_name: str) -> list[str] | None:
    """Resolve only known application candidates; never accept user paths."""
    for candidate in ALLOWED_APPS[app_name]:
        expanded = os.path.expandvars(candidate[0])
        # Bare Windows system commands are intentionally whitelisted above.
        if not Path(expanded).is_absolute() or Path(expanded).is_file():
            return [expanded, *candidate[1:]]
    return None


def open_browser(url: str) -> str:
    """
    Открывает URL в браузере по умолчанию.
    url: полный адрес, например https://youtube.com
    """
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    logger.info("[TOOL] open_browser('%s')", url)
    try:
        opened = webbrowser.open(url)
        if not opened:
            logger.info("[TOOL] failed: Browser failed to open URL")
            return _tool_result(False, url=url, error="Браузер не подтвердил открытие URL.")
        logger.info("[TOOL] success")
        return _tool_result(True, url=url)
    except Exception as e:
        logger.error(f"[TOOL] failed: {e}")
        return _tool_result(False, url=url, error=f"Не удалось открыть браузер: {e}")


def open_app(app_name: str, resolved_target: str | None = None) -> str:
    """
    Запускает приложение по имени из списка или по динамически найденному пути/ярлыку.

    Args:
        app_name: Имя приложения (каноническое или display name).
        resolved_target: Доверенный путь к exe/lnk от target_resolver (необязательно).
                         Если передан и файл существует — запуск без повторного поиска.
    """
    app_key = app_name.lower().strip()
    logger.info("[TOOL] open_app('%s', resolved_target='%s')", app_name, resolved_target)

    # 0. Если путь уже найден target_resolver'ом — используем его напрямую
    if resolved_target:
        target_path = Path(resolved_target)
        if target_path.exists():
            try:
                if target_path.suffix.lower() == ".lnk":
                    os.startfile(str(target_path))
                else:
                    subprocess.Popen([str(target_path)])
                logger.info("[TOOL] success (resolved_target)")
                return _tool_result(True, application=app_name)
            except Exception as e:
                try:
                    os.startfile(str(target_path))
                    logger.info("[TOOL] success (resolved_target via startfile)")
                    return _tool_result(True, application=app_name)
                except Exception as inner_e:
                    logger.error("[TOOL] failed (resolved_target): %s", inner_e)
                    return _tool_result(False, application=app_name, error=f"Не удалось запустить {app_name}: {inner_e}")
        else:
            logger.warning("[TOOL] resolved_target path does not exist: '%s', falling through", resolved_target)

    # 1. Если передано известное приложение из белого списка
    if app_key in ALLOWED_APPS:
        command = _resolve_allowed_app(app_key)
        if command is not None:
            try:
                subprocess.Popen(command)
                logger.info("[TOOL] success")
                return _tool_result(True, application=app_name)
            except Exception as e:
                logger.error(f"[TOOL] failed: {e}")
                return _tool_result(False, application=app_name, error=f"Не удалось запустить {app_name}: {e}")

    # 2. Если передан прямой путь или ярлык (.lnk / .exe)
    target_path = Path(app_name)
    if target_path.exists():
        try:
            os.startfile(str(target_path))
            logger.info("[TOOL] success")
            return _tool_result(True, application=target_path.stem)
        except Exception as e:
            logger.error(f"[TOOL] failed: {e}")
            return _tool_result(False, application=app_name, error=f"Не удалось запустить {app_name}: {e}")

    # 3. Динамический поиск через target_resolver
    from target_resolver import resolve_target
    resolved = resolve_target(app_name)
    if resolved is not None:
        if resolved.kind == "app":
            try:
                os.startfile(resolved.target)
                logger.info("[TOOL] success")
                return _tool_result(True, application=resolved.display_name)
            except Exception as e:
                try:
                    subprocess.Popen([resolved.target])
                    logger.info("[TOOL] success")
                    return _tool_result(True, application=resolved.display_name)
                except Exception as inner_e:
                    logger.error("[TOOL] failed: %s", inner_e)
                    return _tool_result(False, application=app_name, error=f"Не удалось открыть {app_name}: {inner_e}")
        elif resolved.kind == "browser":
            return open_browser(resolved.target)

    error = f"Приложение '{app_name}' не найдено на устройстве."
    logger.error("[TOOL] failed: %s", error)
    return _tool_result(False, application=app_name, error=error)


def close_app(app_name: str, resolved_target: str | None = None) -> str:
    """Закрыть приложение по имени процесса или найденному пути."""
    app_key = app_name.lower().strip()
    logger.info("[CLOSE] target='%s'", app_name)

    process_name = _PROCESS_NAMES.get(app_key)
    if not process_name and resolved_target:
        stem = Path(resolved_target).stem.lower()
        process_name = _PROCESS_NAMES.get(stem, f"{stem}.exe" if not stem.endswith(".exe") else stem)

    if not process_name:
        process_name = f"{app_key}.exe"

    try:
        completed = subprocess.run(
            ["taskkill", "/F", "/T", "/IM", process_name],
            capture_output=True,
            text=True,
            encoding="cp866",   # Windows OEM-кодировка консольных утилит вроде taskkill
            errors="replace",   # не падать, если попадётся byte вне cp866
            timeout=15,
            check=False,
        )
    except Exception as exc:
        logger.error("[CLOSE] target='%s' result=failed (%s)", app_name, exc)
        return _tool_result(False, application=app_name, error=f"Не удалось закрыть {app_name}: {exc}")

    if completed.returncode == 0:
        logger.info("[CLOSE] target='%s' result=success", app_name)
        return _tool_result(True, application=app_name)

    details = (completed.stderr or completed.stdout).strip() or "Процесс не найден или не завершился."
    logger.error("[CLOSE] target='%s' result=failed (%s)", app_name, details)
    return _tool_result(False, application=app_name, error=details)


def close_target(target: str, kind: str = "browser", url: str = "") -> str:
    """
    Безопасное закрытие цели.
    Если это веб-сайт в браузере и отдельную вкладку закрыть нельзя —
    возвращает результат target_not_individually_closeable.
    """
    logger.info("[CLOSE] target='%s'", target)
    if kind == "browser":
        logger.info("[CLOSE] target='%s' result=failed (target_not_individually_closeable)", target)
        return _tool_result(
            False,
            target=target,
            error="target_not_individually_closeable",
            details="Управление отдельными вкладками браузера не поддерживается. Вы можете закрыть браузер целиком командой 'закрой Chrome'.",
        )
    return close_app(target)


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
        "description": "Запустить разрешённую программу из белого списка (chrome, vscode, discord, telegram, notepad, explorer).",
        "parameters": {
            "type": "object",
            "properties": {
                "app_name": {
                    "type": "string",
                    "description": "Имя программы для запуска из белого списка (например, 'chrome', 'vscode', 'discord', 'telegram').",
                },
                "resolved_target": {
                    "type": "string",
                    "description": "Доверенный путь к exe/lnk от target_resolver. Используется только action router'ом.",
                },
            },
            "required": ["app_name"],
        },
    },
    "requires_confirmation": False,
}

CLOSE_APP_DEFINITION: dict = {
    "function": {
        "name": "close_app",
        "description": "Закрыть разрешённую программу из белого списка после подтверждения пользователя.",
        "parameters": {
            "type": "object",
            "properties": {
                "app_name": {
                    "type": "string",
                    "description": "Имя приложения из белого списка, например 'chrome' или 'vscode'.",
                },
            },
            "required": ["app_name"],
        },
    },
    "requires_confirmation": True,
}

CLOSE_TARGET_DEFINITION: dict = {
    "function": {
        "name": "close_target",
        "description": "Закрыть целевую программу или сайт.",
        "parameters": {
            "type": "object",
            "properties": {
                "target": {
                    "type": "string",
                    "description": "Имя приложения или сайта для закрытия.",
                },
                "kind": {
                    "type": "string",
                    "description": "Тип цели ('browser' или 'app').",
                },
                "url": {
                    "type": "string",
                    "description": "URL веб-сайта если доступен.",
                },
            },
            "required": ["target"],
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
