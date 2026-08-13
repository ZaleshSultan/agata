"""
Unit tests for tts_text_normalizer.py
"""

import unittest
from tts_text_normalizer import normalize_text_for_tts


class TestTTSTextNormalizer(unittest.TestCase):

    def test_markdown_removal(self) -> None:
        raw = "### Заголовок\n**Жирный текст** и *курсив* с `кодом` и [ссылкой](https://example.com)."
        cleaned = normalize_text_for_tts(raw)
        self.assertNotIn("###", cleaned)
        self.assertNotIn("**", cleaned)
        self.assertNotIn("`", cleaned)
        self.assertNotIn("[ссылкой]", cleaned)
        self.assertIn("Жирный текст", cleaned)
        self.assertIn("ссылкой", cleaned)

    def test_latex_formulas(self) -> None:
        raw = "Формула: $ D = b^2 - 4ac $"
        cleaned = normalize_text_for_tts(raw)
        self.assertNotIn("$", cleaned)
        self.assertIn("в квадрате", cleaned)
        self.assertIn("минус", cleaned)
        self.assertIn("равно", cleaned)

    def test_fraction_and_sqrt(self) -> None:
        raw = "Результат равен $\\sqrt{16}$ и $\\frac{a}{b}$."
        cleaned = normalize_text_for_tts(raw)
        self.assertIn("корень из 16", cleaned)
        self.assertIn("a разделить на b", cleaned)

    def test_url_replacement(self) -> None:
        raw = "Смотрите на https://www.youtube.com/watch?v=12345 подробнее."
        cleaned = normalize_text_for_tts(raw)
        self.assertNotIn("https://", cleaned)
        self.assertIn("ссылка на Ютуб", cleaned)

    def test_math_operators(self) -> None:
        raw = "12 + 17 = 29"
        cleaned = normalize_text_for_tts(raw)
        self.assertIn("плюс", cleaned)
        self.assertIn("равно", cleaned)


if __name__ == "__main__":
    unittest.main()
