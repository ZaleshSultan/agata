"""
Рабочая память Агаты.

Хранит актуальный контекст (последние 1–3 дня) в виде Markdown-файла.
Всегда подаётся в системный промпт — модель видит его каждый раз.
Обновляется вручную или через процесс консолидации ("сон").

Побочные эффекты: чтение/запись файла data/working_memory.md.
"""

import logging
from pathlib import Path

import config

logger = logging.getLogger(__name__)

# Шаблон для первого запуска
_TEMPLATE: str = """\
# Рабочая память Агаты

## О пользователе
- (пока ничего не известно)

## Текущие задачи
- (нет активных задач)

## Важные факты
- (пусто)
"""


def read() -> str:
    """Прочитать рабочую память. Создаёт шаблон, если файл не существует."""
    path: Path = config.WORKING_MEMORY_PATH
    if not path.exists():
        logger.info("Файл рабочей памяти не найден — создаю шаблон: %s", path)
        write(_TEMPLATE)
        return _TEMPLATE
    content: str = path.read_text(encoding="utf-8")
    return content


def write(content: str) -> None:
    """Перезаписать рабочую память новым содержимым."""
    path: Path = config.WORKING_MEMORY_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    logger.info("Рабочая память обновлена (%d символов)", len(content))
