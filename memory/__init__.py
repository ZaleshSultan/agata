"""
Пакет памяти Агаты.

Содержит:
- diary — долгосрочный дневник (SQLite + семантический поиск)
- working_memory — рабочая память (Markdown-файл)
"""

from memory.manager import MemoryManager

__all__ = ["MemoryManager"]
