"""
Потоковый разбиватель текста на предложения (Sentence Chunker).

Накапливает токены от LLM-стрима и разбивает их на естественные
речевые фрагменты для TTS.  Не отправляет каждый токен отдельно —
собирает предложения до знаков препинания, учитывая минимальную длину.

Фильтрует теги <think>...</think> (Qwen3) и markdown-разметку.
"""

import logging
import re
from typing import Optional

import config

logger = logging.getLogger(__name__)

# Знаки, по которым разрешено разбивать на предложения
_SENTENCE_TERMINATORS = frozenset(".!?")
_CLAUSE_SEPARATORS = frozenset(";:\n")

# Защита от разбиения по точке внутри чисел и сокращений
_ABBREVIATIONS = re.compile(
    r"(?:"
    r"\b(?:т\.д|т\.е|т\.п|т\.к|т\.н|и\.т\.д|и\.т\.п|др|пр|ст|гг|г|руб|коп|тыс|млн|млрд|ул|пр-т)"
    r"|"
    r"\d"  # Цифра перед точкой (напр. 3.14)
    r")$"
)


def _looks_like_abbreviation(text_before_dot: str) -> bool:
    """Проверить, заканчивается ли текст перед точкой на сокращение или цифру."""
    return bool(_ABBREVIATIONS.search(text_before_dot.rstrip()))


# ── Markdown cleanup (лёгкая версия для потока) ──────────────────────────

def _clean_for_speech(text: str) -> str:
    """Убрать markdown-разметку, которая плохо звучит в TTS."""
    cleaned = text

    # Кодовые блоки
    cleaned = re.sub(r"```.*?```", "", cleaned, flags=re.DOTALL)

    # Inline code
    cleaned = re.sub(r"`([^`]+)`", r"\1", cleaned)

    # Bold / italic
    cleaned = re.sub(r"\*\*(.+?)\*\*", r"\1", cleaned)
    cleaned = re.sub(r"__(.+?)__", r"\1", cleaned)
    cleaned = re.sub(r"\*(.+?)\*", r"\1", cleaned)
    cleaned = re.sub(r"_(.+?)_", r"\1", cleaned)

    # Markdown links
    cleaned = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", cleaned)

    # Заголовки
    cleaned = re.sub(r"^#{1,6}\s*", "", cleaned, flags=re.MULTILINE)

    # Strikethrough
    cleaned = re.sub(r"~~(.+?)~~", r"\1", cleaned)

    # Лишние пробелы
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    return cleaned


class SentenceChunker:
    """
    Потоковый разбиватель текста на предложения для TTS.

    Использование:
        chunker = SentenceChunker()
        for token in llm_stream:
            for sentence in chunker.feed(token):
                tts.enqueue(sentence)
        remainder = chunker.flush()
        if remainder:
            tts.enqueue(remainder)
    """

    def __init__(self, min_chars: int | None = None) -> None:
        self._buffer: str = ""
        self._min_chars: int = min_chars if min_chars is not None else config.TTS_MIN_SENTENCE_CHARS
        self._inside_think: bool = False

    def feed(self, token: str) -> list[str]:
        """
        Добавить токен в буфер, вернуть список готовых предложений.

        Может вернуть пустой список (токен накоплен, но предложение ещё не завершено),
        один элемент (одно завершённое предложение) или несколько
        (если токен содержал несколько знаков препинания).
        """
        if not token:
            return []

        # ── Фильтрация <think>...</think> ────────────────────────────────
        filtered = self._filter_think_tags(token)
        if not filtered:
            return []

        self._buffer += filtered
        return self._extract_sentences()

    def flush(self) -> Optional[str]:
        """
        Вернуть остаток буфера после окончания потока.

        Должен вызываться, когда LLM-стрим завершён, чтобы
        не потерять последнюю неоконченную фразу.
        """
        if not self._buffer.strip():
            self._buffer = ""
            return None

        sentence = _clean_for_speech(self._buffer.strip())
        self._buffer = ""
        self._inside_think = False

        if not sentence:
            return None

        logger.info("[CHUNKER] flush: '%s'", sentence[:80])
        return sentence

    def reset(self) -> None:
        """Сбросить состояние (новый запрос)."""
        self._buffer = ""
        self._inside_think = False

    def _filter_think_tags(self, token: str) -> str:
        """Отфильтровать содержимое тегов <think>...</think> на лету."""
        result: list[str] = []
        i = 0

        while i < len(token):
            if self._inside_think:
                # Ищем закрывающий тег
                close_pos = token.find("</think>", i)
                if close_pos == -1:
                    # Весь остаток токена внутри think — пропускаем
                    break
                else:
                    self._inside_think = False
                    i = close_pos + len("</think>")
            else:
                # Ищем открывающий тег
                open_pos = token.find("<think>", i)
                if open_pos == -1:
                    # Нет тегов — берём весь остаток
                    result.append(token[i:])
                    break
                else:
                    # Берём текст до <think>
                    result.append(token[i:open_pos])
                    self._inside_think = True
                    i = open_pos + len("<think>")

        return "".join(result)

    def _extract_sentences(self) -> list[str]:
        """Извлечь завершённые предложения из буфера."""
        sentences: list[str] = []

        while True:
            split_pos = self._find_split_point()
            if split_pos is None:
                break

            # Разбиваем по найденной позиции (включая знак препинания)
            raw_sentence = self._buffer[:split_pos + 1].strip()
            self._buffer = self._buffer[split_pos + 1:]

            if not raw_sentence:
                continue

            cleaned = _clean_for_speech(raw_sentence)
            if cleaned:
                logger.info("[CHUNKER] sentence: '%s'", cleaned[:80])
                sentences.append(cleaned)

        return sentences

    def _find_split_point(self) -> Optional[int]:
        """
        Найти позицию для разбиения буфера.

        Ищет знак препинания (.!?;:\\n), после которого буфер
        достаточно длинный (>= min_chars).  Защищает от разбиения
        по точке в числах (3.14) и сокращениях (т.д.).
        """
        for i, char in enumerate(self._buffer):
            if char in _SENTENCE_TERMINATORS:
                candidate_text = self._buffer[:i]

                # Защита: не разбивать по точке внутри чисел/сокращений
                if char == "." and _looks_like_abbreviation(candidate_text):
                    continue

                # Защита: не разбивать по точке, если за ней идёт цифра (3.14)
                if char == "." and i + 1 < len(self._buffer) and self._buffer[i + 1].isdigit():
                    continue

                # Проверяем минимальную длину
                if len(candidate_text.strip()) + 1 >= self._min_chars:
                    return i

            elif char in _CLAUSE_SEPARATORS:
                candidate_text = self._buffer[:i]
                if len(candidate_text.strip()) + 1 >= self._min_chars:
                    return i

        return None
