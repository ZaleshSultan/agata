"""
Универсальный слой разрешения целей (Target Resolver) для открытия приложений и веб-сайтов.

Иерархия разрешения целей (Tier 1 - Tier 4):
1. Прямые URL и домены (https://..., example.com).
2. Веб-сервисы (Spotify, YouTube, GitHub, ChatGPT, Reddit, Telegram и др.):
   - Если установлено локальное приложение — запустить его;
   - Если локального приложения нет — открыть URL веб-сервиса без ошибки.
3. Поиск локально установленных приложений Windows:
   - Ярлыки меню Пуск (.lnk) в пользовательских и системных каталогах (с учётом алиасов);
   - Известные прямые пути установки (VS Code, Chrome, Discord, Telegram);
   - Раздел реестра App Paths;
   - Системный PATH и WindowsApp aliases (%LOCALAPPDATA%\\Microsoft\\WindowsApps).
4. Если цель не найдена — вернуть None (для передачи LLM / уточнения).

Возвращает `ResolvedTarget`:
- `kind`: "app" или "browser"
- `target`: путь к исполняемому файлу / ярлыку или URL
- `display_name`: Понятное отображаемое имя
- `source`: Источник решения ("url", "web_service", "start_menu", "known_path", "registry", "path")
"""

import logging
import os
import re
import shutil
import winreg
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ResolvedTarget:
    kind: str  # "app" или "browser"
    target: str  # Путь к exe/lnk или URL
    display_name: str  # Понятное имя для ответа пользователю
    source: str = ""  # "url" | "web_service" | "start_menu" | "known_path" | "registry" | "path"


# ── Схема доменов / URL ───────────────────────────────────────────────────
_URL_PATTERN = re.compile(
    r"^(?:https?://)?(?:[a-z0-9-]+\.)+[a-z]{2,}(?:/[^\s]*)?$",
    re.IGNORECASE,
)

# ── Карты веб-сервисов ───────────────────────────────────────────────────
_WEB_SERVICES: dict[str, tuple[str, str]] = {
    "spotify": ("https://open.spotify.com", "Spotify"),
    "спотифай": ("https://open.spotify.com", "Spotify"),
    "youtube": ("https://www.youtube.com", "YouTube"),
    "ютуб": ("https://www.youtube.com", "YouTube"),
    "ютьюб": ("https://www.youtube.com", "YouTube"),
    "github": ("https://github.com", "GitHub"),
    "гитхаб": ("https://github.com", "GitHub"),
    "discord": ("https://discord.com/app", "Discord"),
    "дискорд": ("https://discord.com/app", "Discord"),
    "telegram": ("https://web.telegram.org", "Telegram"),
    "телеграм": ("https://web.telegram.org", "Telegram"),
    "телега": ("https://web.telegram.org", "Telegram"),
    "reddit": ("https://www.reddit.com", "Reddit"),
    "реддит": ("https://www.reddit.com", "Reddit"),
    "google": ("https://www.google.com", "Google"),
    "гугл": ("https://www.google.com", "Google"),
    "chatgpt": ("https://chatgpt.com", "ChatGPT"),
    "чатгпт": ("https://chatgpt.com", "ChatGPT"),
    "chat gpt": ("https://chatgpt.com", "ChatGPT"),
    "twitch": ("https://www.twitch.tv", "Twitch"),
    "твич": ("https://www.twitch.tv", "Twitch"),
    "vk": ("https://vk.com", "ВКонтакте"),
    "вконтакте": ("https://vk.com", "ВКонтакте"),
    "вк": ("https://vk.com", "ВКонтакте"),
    "yandex": ("https://yandex.ru", "Яндекс"),
    "янндекс": ("https://yandex.ru", "Яндекс"),
    "steam": ("https://store.steampowered.com", "Steam"),
    "стим": ("https://store.steampowered.com", "Steam"),
}

# Синонимы имён программ для расширенного поиска
_APP_ALIASES: dict[str, list[str]] = {
    "vscode": ["code", "vscode", "vs code", "visual studio code", "вс код", "vs код"],
    "chrome": ["chrome", "google chrome", "гугл хром", "гугл-хром", "хром"],
    "discord": ["discord", "дискорд"],
    "telegram": ["telegram", "telegram desktop", "телеграм", "телега"],
    "spotify": ["spotify", "спотифай"],
    "steam": ["steam", "стим"],
    "notepad": ["notepad", "блокнот"],
    "explorer": ["explorer", "проводник"],
    "calculator": ["calc", "calculator", "калькулятор"],
}

# Известные стандартные пути установки приложений на Windows
_KNOWN_APP_PATHS: dict[str, list[str]] = {
    "vscode": [
        r"%LOCALAPPDATA%\Programs\Microsoft VS Code\Code.exe",
        r"%ProgramFiles%\Microsoft VS Code\Code.exe",
        r"%ProgramFiles(x86)%\Microsoft VS Code\Code.exe",
    ],
    "chrome": [
        r"%ProgramFiles%\Google\Chrome\Application\chrome.exe",
        r"%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe",
        r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe",
    ],
    "discord": [
        r"%LOCALAPPDATA%\Discord\Update.exe",
        r"%LOCALAPPDATA%\Programs\Discord\Discord.exe",
    ],
    "telegram": [
        r"%APPDATA%\Telegram Desktop\Telegram.exe",
        r"%LOCALAPPDATA%\Programs\Telegram Desktop\Telegram.exe",
    ],
}


def _clean_name(name: str) -> str:
    """Нормализация строки для сопоставления файлов и ярлыков."""
    return re.sub(r"[^\w\s]", "", name.lower()).strip()


def _get_search_queries(target_raw: str) -> list[str]:
    """Получить список всех вариантов имён для поиска программы."""
    raw_clean = _clean_name(target_raw)
    queries = [raw_clean]

    for key, aliases in _APP_ALIASES.items():
        aliases_clean = [_clean_name(a) for a in aliases]
        if raw_clean in aliases_clean or raw_clean == key:
            for a in aliases_clean:
                if a not in queries:
                    queries.append(a)
            if key not in queries:
                queries.append(key)

    return queries


# ── 1. Поиск ярлыков в Меню Пуск Windows ───────────────────────────────────
def _find_start_menu_shortcut(target_raw: str, exact_only: bool = False) -> Optional[Path]:
    """Сканировать директории меню Пуск в поисках .lnk файла приложения."""
    queries = _get_search_queries(target_raw)
    start_menu_dirs = [
        Path(os.path.expandvars(r"%APPDATA%\Microsoft\Windows\Start Menu\Programs")),
        Path(os.path.expandvars(r"%ProgramData%\Microsoft\Windows\Start Menu\Programs")),
    ]

    candidates: list[tuple[int, Path]] = []

    for base_dir in start_menu_dirs:
        if not base_dir.exists():
            continue
        try:
            for path in base_dir.rglob("*.lnk"):
                stem_clean = _clean_name(path.stem)

                if any(w in stem_clean for w in ["uninstall", "удалить", "help", "readme", "documentation", "справка"]):
                    continue

                for q in queries:
                    if q == stem_clean:
                        return path
                    if not exact_only and len(q) >= 3 and (q in stem_clean or stem_clean in q):
                        score = abs(len(stem_clean) - len(q))
                        candidates.append((score, path))
        except Exception as exc:
            logger.debug("Ошибка чтения каталога меню пуск %s: %s", base_dir, exc)

    if not exact_only and candidates:
        candidates.sort(key=lambda x: x[0])
        return candidates[0][1]

    return None


# ── 2. Поиск по известным прямым путям установки ─────────────────────────
def _find_known_path(target_raw: str) -> Optional[str]:
    """Проверить известную таблицу прямых путей установки."""
    queries = _get_search_queries(target_raw)
    for app_key, paths in _KNOWN_APP_PATHS.items():
        if app_key in queries or any(q in queries for q in _APP_ALIASES.get(app_key, [])):
            for p in paths:
                expanded = os.path.expandvars(p)
                if Path(expanded).exists():
                    return expanded
    return None


# ── 3. Поиск в реестре Windows (App Paths) ─────────────────────────────────
def _find_in_registry_app_paths(target_raw: str) -> Optional[str]:
    """Поиск исполняемого файла в ветках реестра App Paths."""
    queries = _get_search_queries(target_raw)
    executable_names = [f"{q}.exe" for q in queries] + queries

    registry_roots = [
        (winreg.HKEY_CURRENT_USER, r"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths"),
        (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths"),
    ]

    for root, subkey_path in registry_roots:
        try:
            with winreg.OpenKey(root, subkey_path) as parent_key:
                info = winreg.QueryInfoKey(parent_key)
                for i in range(info[0]):
                    subkey_name = winreg.EnumKey(parent_key, i)
                    name_without_ext = _clean_name(subkey_name.replace(".exe", ""))
                    if subkey_name.lower() in [e.lower() for e in executable_names] or name_without_ext in queries:
                        with winreg.OpenKey(parent_key, subkey_name) as app_key:
                            exe_path, _ = winreg.QueryValueEx(app_key, "")
                            expanded = os.path.expandvars(exe_path).strip('"')
                            if Path(expanded).exists():
                                return expanded
        except Exception:
            continue

    return None


# ── 4. Поиск через PATH и App Execution Aliases ───────────────────────────
def _find_in_path(target_raw: str) -> Optional[str]:
    """Проверка наличия файла в PATH или WindowsApps."""
    queries = _get_search_queries(target_raw)
    names_to_try = queries + [f"{q}.exe" for q in queries]

    for name in names_to_try:
        found = shutil.which(name)
        if found:
            return found

    windows_apps_dir = Path(os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WindowsApps"))
    if windows_apps_dir.exists():
        for name in names_to_try:
            exe_file = windows_apps_dir / (name if name.endswith(".exe") else f"{name}.exe")
            if exe_file.exists():
                return str(exe_file)

    return None


# ── Поиск локального приложения на диске ──────────────────────────────────
def _find_local_app(target_raw: str, exact_only: bool = False) -> Optional[ResolvedTarget]:
    """Попытаться найти локальное Windows приложение через все дисковые источники."""
    # 1. Меню Пуск (.lnk)
    shortcut_path = _find_start_menu_shortcut(target_raw, exact_only=exact_only)
    if shortcut_path is not None:
        display_name = shortcut_path.stem
        logger.info("[TARGET] found kind=app source=start_menu path='%s'", shortcut_path)
        return ResolvedTarget(kind="app", target=str(shortcut_path), display_name=display_name, source="start_menu")

    # 2. Известные пути установки
    known_exe = _find_known_path(target_raw)
    if known_exe is not None:
        display_name = Path(known_exe).stem
        if display_name.lower() == "code":
            display_name = "VS Code"
        else:
            display_name = display_name.capitalize()
        logger.info("[TARGET] found kind=app source=known_path path='%s'", known_exe)
        return ResolvedTarget(kind="app", target=known_exe, display_name=display_name, source="known_path")

    # 3. Реестр App Paths
    reg_exe = _find_in_registry_app_paths(target_raw)
    if reg_exe is not None:
        display_name = Path(reg_exe).stem.capitalize()
        logger.info("[TARGET] found kind=app source=registry path='%s'", reg_exe)
        return ResolvedTarget(kind="app", target=reg_exe, display_name=display_name, source="registry")

    # 4. PATH / WindowsApps
    path_exe = _find_in_path(target_raw)
    if path_exe is not None:
        display_name = Path(path_exe).stem.capitalize()
        logger.info("[TARGET] found kind=app source=path path='%s'", path_exe)
        return ResolvedTarget(kind="app", target=path_exe, display_name=display_name, source="path")

    return None


# ── Главная функция разрешения цели ────────────────────────────────────────
def resolve_target(target_raw: str) -> Optional[ResolvedTarget]:
    """
    Универсальное разрешение цели для открывающих команд.

    Args:
        target_raw: Имя программы, название сайта или URL (например "VS Code", "Spotify", "https://youtube.com").

    Returns:
        ResolvedTarget с типом ("app" или "browser") или None, если определить невозможно.
    """
    if not target_raw or not target_raw.strip():
        return None

    target_str = target_raw.strip().rstrip(".,!?:;'\"")
    target_lower = target_str.lower()
    logger.info("[TARGET] resolving '%s'", target_str)

    # Tier 1: Проверка на явный URL или домен (например https://..., example.com)
    if target_lower.startswith(("http://", "https://")) or ("." in target_lower and " " not in target_lower and _URL_PATTERN.match(target_lower)):
        url = target_str if target_lower.startswith(("http://", "https://")) else f"https://{target_str}"
        logger.info("[TARGET] explicit URL -> %s", url)
        return ResolvedTarget(kind="browser", target=url, display_name=target_str, source="url")

    # Tier 2: Веб-сервисы (Spotify, YouTube, GitHub, ChatGPT, Reddit, Telegram и т.д.)
    if target_lower in _WEB_SERVICES:
        # Проверяем, есть ли явный ярлык рабочего стола / меню Пуск для этого сервиса
        shortcut = _find_start_menu_shortcut(target_str, exact_only=True)
        if shortcut is not None:
            logger.info("[TARGET] found desktop app shortcut for web service: '%s'", shortcut)
            return ResolvedTarget(kind="app", target=str(shortcut), display_name=shortcut.stem, source="start_menu")

        # Если локального десктоп-ярлыка нет — открываем веб-версию сервиса без ошибки
        service_url, display_name = _WEB_SERVICES[target_lower]
        logger.info("[TARGET] web service '%s' -> %s", target_str, service_url)
        return ResolvedTarget(kind="browser", target=service_url, display_name=display_name, source="web_service")

    # Tier 3: Ищем локальное приложение на диске Windows
    local_app = _find_local_app(target_str)
    if local_app is not None:
        return local_app

    # Tier 4: Цель не найдена
    logger.info("[TARGET] target '%s' could not be resolved", target_str)
    return None
