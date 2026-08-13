"""
Универсальный слой разрешения целей (Target Resolver) для открытия приложений и веб-сайтов.

Иерархия разрешения целей (Tier 1 - Tier 5):
1. Прямые URL и домены (https://..., example.com).
2. Веб-сервисы и приложения по расширенным алиасам и формам падежей:
   - Если установлено локальное приложение — запустить его;
   - Если локального приложения нет — открыть URL веб-сервиса без ошибки.
3. Нормализация русских падежей и окончаний (хрому -> хром, телеграмму -> телеграм, спотик -> spotify).
4. Fuzzy matching (Levenshtein) по известным алиасам и сервисам.
5. Поиск локально установленных приложений Windows:
   - Ярлыки меню Пуск (.lnk) с учётом fuzzy matching;
   - Известные пути установки (VS Code, Chrome, Discord, Telegram, Spotify);
   - Реестр App Paths;
   - Системный PATH и WindowsApps.

Возвращает `ResolvedTarget`:
- `kind`: "app" или "browser"
- `target`: путь к исполняемому файлу / ярлыку или URL
- `display_name`: Понятное отображаемое имя
- `source`: Источник решения ("url", "web_service", "start_menu", "known_path", "registry", "path", "fuzzy")
"""

import logging
import os
import re
import shutil
import winreg
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from fuzzy_resolver import fuzzy_resolve

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ResolvedTarget:
    kind: str  # "app" или "browser"
    target: str  # Путь к exe/lnk или URL
    display_name: str  # Понятное имя для ответа пользователю
    source: str = ""  # "url" | "web_service" | "start_menu" | "known_path" | "registry" | "path" | "fuzzy"


# ── Схема доменов / URL ───────────────────────────────────────────────────
_URL_PATTERN = re.compile(
    r"^(?:https?://)?(?:[a-z0-9-]+\.)+[a-z]{2,}(?:/[^\s]*)?$",
    re.IGNORECASE,
)

# ── Карты веб-сервисов (канонический URL и имя) ──────────────────────────
_WEB_SERVICES: dict[str, tuple[str, str]] = {
    # Spotify
    "spotify": ("https://open.spotify.com", "Spotify"),
    "спотифай": ("https://open.spotify.com", "Spotify"),
    "споти": ("https://open.spotify.com", "Spotify"),
    "спотик": ("https://open.spotify.com", "Spotify"),
    "спотифаю": ("https://open.spotify.com", "Spotify"),
    "спотифаем": ("https://open.spotify.com", "Spotify"),
    # YouTube
    "youtube": ("https://www.youtube.com", "YouTube"),
    "ютуб": ("https://www.youtube.com", "YouTube"),
    "ютуба": ("https://www.youtube.com", "YouTube"),
    "ютубу": ("https://www.youtube.com", "YouTube"),
    "ютубом": ("https://www.youtube.com", "YouTube"),
    "ютубчик": ("https://www.youtube.com", "YouTube"),
    "ютубчика": ("https://www.youtube.com", "YouTube"),
    "ютубчику": ("https://www.youtube.com", "YouTube"),
    "ютьюб": ("https://www.youtube.com", "YouTube"),
    # GitHub
    "github": ("https://github.com", "GitHub"),
    "гитхаб": ("https://github.com", "GitHub"),
    "гит": ("https://github.com", "GitHub"),
    # Reddit
    "reddit": ("https://www.reddit.com", "Reddit"),
    "реддит": ("https://www.reddit.com", "Reddit"),
    # Google
    "google": ("https://www.google.com", "Google"),
    "гугл": ("https://www.google.com", "Google"),
    # ChatGPT
    "chatgpt": ("https://chatgpt.com", "ChatGPT"),
    "чатгпт": ("https://chatgpt.com", "ChatGPT"),
    "чат гпт": ("https://chatgpt.com", "ChatGPT"),
    # Twitch
    "twitch": ("https://www.twitch.tv", "Twitch"),
    "твич": ("https://www.twitch.tv", "Twitch"),
    # VK
    "vk": ("https://vk.com", "ВКонтакте"),
    "вконтакте": ("https://vk.com", "ВКонтакте"),
    "вк": ("https://vk.com", "ВКонтакте"),
    # Yandex
    "yandex": ("https://yandex.ru", "Яндекс"),
    "яндекс": ("https://yandex.ru", "Яндекс"),
}

# Синонимы имён программ для расширенного поиска на локальном ПК
_APP_ALIASES: dict[str, list[str]] = {
    "chrome": [
        "chrome", "google chrome", "гугл хром", "гугл-хром", "хром",
        "хрому", "хромом", "хроме", "хромчик", "гугл хрому", "гугл",
    ],
    "vscode": [
        "code", "vscode", "vs code", "visual studio code", "вс код", "vs код",
        "вскод", "вижуал студио", "вижуал студио код", "визуал студио",
    ],
    "discord": [
        "discord", "дискорд", "дискорда", "дискорду", "дискордом", "дискордик", "диск",
    ],
    "telegram": [
        "telegram", "telegram desktop", "телеграм", "телеграмм", "телеграмма",
        "телеграмму", "телеграме", "телеграмом", "телега", "телегу", "телеге",
        "телегой", "тележка", "тг",
    ],
    "spotify": [
        "spotify", "спотифай", "споти", "спотик", "спотифаю", "спотифаем",
    ],
    "steam": [
        "steam", "стим", "стима", "стиму", "стимом",
    ],
    "notepad": [
        "notepad", "блокнот", "блокнота", "блокноту", "блокнотом", "блокнотик",
    ],
    "explorer": [
        "explorer", "проводник", "проводника", "проводнику", "проводником",
    ],
    "calculator": [
        "calc", "calculator", "калькулятор", "калькулятора", "калькулятору", "кальк",
    ],
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
    "spotify": [
        r"%APPDATA%\Spotify\Spotify.exe",
        r"%LOCALAPPDATA%\Microsoft\WindowsApps\Spotify.exe",
    ],
}

# Карта канонических отображаемых имен
_CANONICAL_APP_NAMES: dict[str, str] = {
    "chrome": "Chrome",
    "vscode": "VS Code",
    "discord": "Discord",
    "telegram": "Telegram",
    "spotify": "Spotify",
    "steam": "Steam",
    "notepad": "Блокнот",
    "explorer": "Проводник",
    "calculator": "Калькулятор",
}


def _clean_name(name: str) -> str:
    """Нормализация строки для сопоставления файлов и ярлыков."""
    return re.sub(r"[^\w\s]", "", name.lower()).strip()


def _generate_stemmed_variants(raw_name: str) -> list[str]:
    """
    Генерирует базовые формы слова путём нормализации частых падежных окончаний и суффиксов.
    """
    clean = _clean_name(raw_name)
    variants = [clean]

    # Снятие уменьшительно-ласкательных суффиксов: хромчик -> хром, ютубчик -> ютуб, спотик -> споти
    if clean.endswith("чик"):
        variants.append(clean[:-3])
    elif clean.endswith("ик"):
        variants.append(clean[:-2])
        variants.append(clean[:-1])  # спотик -> споти

    # Снятие падежных окончаний русского языка (-у, -е, -а, -ом, -ем, -ой, -ей, -ы, -и)
    for suffix in ("у", "е", "а", "ом", "ем", "ой", "ей", "ы", "и", "ой"):
        if len(clean) > 3 and clean.endswith(suffix):
            stem = clean[:-len(suffix)]
            if len(stem) >= 3:
                variants.append(stem)

    # Двойные согласные: телеграмм -> телеграм
    if "мм" in clean:
        variants.append(clean.replace("мм", "м"))
    if "пп" in clean:
        variants.append(clean.replace("пп", "п"))
    if "сс" in clean:
        variants.append(clean.replace("сс", "с"))

    # Уникализация с сохранением порядка
    seen = set()
    result = []
    for v in variants:
        if v and v not in seen:
            seen.add(v)
            result.append(v)
    return result


def _get_search_queries(target_raw: str) -> list[str]:
    """Получить список всех вариантов имён для поиска программы."""
    stemmed = _generate_stemmed_variants(target_raw)
    queries = list(stemmed)

    for stem in stemmed:
        for app_key, aliases in _APP_ALIASES.items():
            aliases_clean = [_clean_name(a) for a in aliases]
            if stem in aliases_clean or stem == app_key:
                for a in aliases_clean:
                    if a not in queries:
                        queries.append(a)
                if app_key not in queries:
                    queries.append(app_key)

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
    all_shortcuts: dict[str, Path] = {}

    for base_dir in start_menu_dirs:
        if not base_dir.exists():
            continue
        try:
            for path in base_dir.rglob("*.lnk"):
                stem_clean = _clean_name(path.stem)

                if any(w in stem_clean for w in ["uninstall", "удалить", "help", "readme", "documentation", "справка"]):
                    continue

                all_shortcuts[stem_clean] = path

                for q in queries:
                    if q == stem_clean:
                        return path
                    if not exact_only and len(q) >= 3 and (q in stem_clean or stem_clean in q):
                        score = abs(len(stem_clean) - len(q))
                        candidates.append((score, path))
        except Exception as exc:
            logger.debug("Ошибка чтения каталога меню пуск %s: %s", base_dir, exc)

    if candidates and not exact_only:
        candidates.sort(key=lambda x: x[0])
        return candidates[0][1]

    # Fuzzy match on shortcut names
    if not exact_only and all_shortcuts:
        for q in queries:
            match = fuzzy_resolve(q, all_shortcuts, threshold=0.75)
            if match:
                logger.info("[TARGET] Start menu fuzzy match: '%s' -> '%s'", q, match.matched_candidate)
                return match.target_value

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
        stem = Path(known_exe).stem.lower()
        display_name = _CANONICAL_APP_NAMES.get(stem, stem.capitalize())
        logger.info("[TARGET] found kind=app source=known_path path='%s'", known_exe)
        return ResolvedTarget(kind="app", target=known_exe, display_name=display_name, source="known_path")

    # 3. Реестр App Paths
    reg_exe = _find_in_registry_app_paths(target_raw)
    if reg_exe is not None:
        stem = Path(reg_exe).stem.lower()
        display_name = _CANONICAL_APP_NAMES.get(stem, stem.capitalize())
        logger.info("[TARGET] found kind=app source=registry path='%s'", reg_exe)
        return ResolvedTarget(kind="app", target=reg_exe, display_name=display_name, source="registry")

    # 4. PATH / WindowsApps
    path_exe = _find_in_path(target_raw)
    if path_exe is not None:
        stem = Path(path_exe).stem.lower()
        display_name = _CANONICAL_APP_NAMES.get(stem, stem.capitalize())
        logger.info("[TARGET] found kind=app source=path path='%s'", path_exe)
        return ResolvedTarget(kind="app", target=path_exe, display_name=display_name, source="path")

    return None


# ── Главная функция разрешения цели ────────────────────────────────────────
def resolve_target(target_raw: str) -> Optional[ResolvedTarget]:
    """
    Универсальное разрешение цели для открывающих и закрывающих команд.

    Args:
        target_raw: Имя программы, название сайта или URL (например "VS Code", "Spotify", "спотик", "телеграмму").

    Returns:
        ResolvedTarget с типом ("app" или "browser") или None, если определить невозможно.
    """
    if not target_raw or not target_raw.strip():
        return None

    target_str = target_raw.strip().rstrip(".,!?:;'\"")
    target_lower = _clean_name(target_str)
    logger.info("[TARGET] resolving '%s'", target_str)

    # Tier 1: Проверка на явный URL или домен (например https://..., example.com)
    if target_str.lower().startswith(("http://", "https://")) or ("." in target_str.lower() and " " not in target_str and _URL_PATTERN.match(target_str.lower())):
        url = target_str if target_str.lower().startswith(("http://", "https://")) else f"https://{target_str}"
        logger.info("[TARGET] explicit URL -> %s", url)
        return ResolvedTarget(kind="browser", target=url, display_name=target_str, source="url")

    # Tier 2: Точное совпадение в веб-сервисах
    if target_lower in _WEB_SERVICES:
        service_url, display_name = _WEB_SERVICES[target_lower]
        logger.info("[TARGET] web service '%s' -> %s", target_str, service_url)
        return ResolvedTarget(kind="browser", target=service_url, display_name=display_name, source="web_service")

    # Tier 3: Сопоставление через формы падежей и стемминг (_generate_stemmed_variants)
    stems = _generate_stemmed_variants(target_str)
    for stem in stems:
        if stem in _WEB_SERVICES:
            service_url, display_name = _WEB_SERVICES[stem]
            logger.info("[TARGET] stemmed match '%s' -> web service %s", stem, service_url)
            return ResolvedTarget(kind="browser", target=service_url, display_name=display_name, source="web_service")

        # Проверка среди известных ключей приложений
        for app_key, aliases in _APP_ALIASES.items():
            if stem == app_key or stem in [_clean_name(a) for a in aliases]:
                local_app = _find_local_app(app_key, exact_only=False)
                if local_app is not None:
                    return local_app
                canonical_name = _CANONICAL_APP_NAMES.get(app_key, app_key.capitalize())
                return ResolvedTarget(kind="app", target=app_key, display_name=canonical_name, source="alias_table")

    # Tier 4: Fuzzy resolve среди известных сервисов и алиасов
    fuzzy_match = fuzzy_resolve(target_lower, _WEB_SERVICES, threshold=0.75)
    if fuzzy_match:
        matched_key = fuzzy_match.matched_candidate
        service_url, display_name = fuzzy_match.target_value
        logger.info("[TARGET] fuzzy matched web service: '%s' -> '%s' (score=%.2f)", target_lower, matched_key, fuzzy_match.score)
        local_app = _find_local_app(matched_key, exact_only=False)
        if local_app is not None:
            return local_app
        return ResolvedTarget(kind="browser", target=service_url, display_name=display_name, source="fuzzy_service")

    # Tier 5: Поиск локального приложения на диске Windows (ярлыки, реестр, PATH)
    local_app = _find_local_app(target_str, exact_only=False)
    if local_app is not None:
        return local_app

    # Tier 6: Цель не найдена
    logger.info("[TARGET] target '%s' could not be resolved", target_str)
    return None
