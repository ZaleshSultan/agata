"""
Тесты для модуля sentence_chunker (разбиватель потокового текста на предложения).
"""

import unittest

from sentence_chunker import SentenceChunker, _clean_for_speech


class TestSentenceChunker(unittest.TestCase):

    def test_single_sentence_with_period(self) -> None:
        """Одно предложение со знаком точки превышающее min_chars отдаётся полностью."""
        chunker = SentenceChunker(min_chars=10)
        tokens = ["Привет", ", ", "как ", "твои ", "дела", "? "]
        results = []
        for token in tokens:
            results.extend(chunker.feed(token))
        self.assertEqual(results, ["Привет, как твои дела?"])
        self.assertIsNone(chunker.flush())

    def test_combining_short_tokens(self) -> None:
        """Токены 'Хорош', 'о, ', 'сейчас ', 'открою ', 'Chrome.' образуют одно предложение."""
        chunker = SentenceChunker(min_chars=15)
        tokens = ["Хорош", "о, ", "сейчас ", "открою ", "Chrome. ", "После ", "этого ", "выполню ", "задачу."]
        results = []
        for token in tokens:
            results.extend(chunker.feed(token))

        remainder = chunker.flush()
        if remainder:
            results.append(remainder)

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0], "Хорошо, сейчас открою Chrome.")
        self.assertEqual(results[1], "После этого выполню задачу.")

    def test_abbreviation_and_number_protection(self) -> None:
        """Числа вида 3.14 и сокращения т.д. не должны разбивать предложение."""
        chunker = SentenceChunker(min_chars=15)
        text = "Значение пи равен 3.14 примерно и т.д. в математике."
        results = []
        for word in text.split():
            results.extend(chunker.feed(word + " "))

        remainder = chunker.flush()
        if remainder:
            results.append(remainder)

        # Не должно быть разбиения по '3.' или 'т.'
        full_result = " ".join(results)
        self.assertIn("3.14", full_result)

    def test_think_tag_stripping(self) -> None:
        """Теги <think>...</think> должны фильтроваться на лету."""
        chunker = SentenceChunker(min_chars=5)
        tokens = [
            "<think>", "Думаю над ", "ответом...", "</think>",
            "Привет", "! ", "Чем ", "могу ", "помочь", "?",
        ]
        results = []
        for token in tokens:
            results.extend(chunker.feed(token))

        remainder = chunker.flush()
        if remainder:
            results.append(remainder)

        self.assertNotIn("Думаю", " ".join(results))
        self.assertIn("Привет!", results)

    def test_empty_stream_and_flush(self) -> None:
        """Пустой стрим возвращает пустой список и None на flush()."""
        chunker = SentenceChunker()
        self.assertEqual(chunker.feed(""), [])
        self.assertIsNone(chunker.flush())

    def test_flush_remaining_buffer(self) -> None:
        """Остаток буфера без финального знака препинания отдаётся по flush()."""
        chunker = SentenceChunker(min_chars=10)
        chunker.feed("Это предложение не имеет точки в конце")
        remainder = chunker.flush()
        self.assertEqual(remainder, "Это предложение не имеет точки в конце")

    def test_markdown_cleaning(self) -> None:
        """Проверка очистки markdown элементов."""
        raw = "## Заголовок\nВот **жирный** и *курсивный* текст с `кодом`."
        cleaned = _clean_for_speech(raw)
        self.assertNotIn("#", cleaned)
        self.assertNotIn("**", cleaned)
        self.assertNotIn("`", cleaned)
        self.assertIn("жирный", cleaned)


if __name__ == "__main__":
    unittest.main()
