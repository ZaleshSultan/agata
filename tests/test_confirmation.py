"""
Unit tests for confirmation.py
"""

import unittest
from confirmation import ConfirmationResult, parse_confirmation


class TestConfirmation(unittest.TestCase):

    def test_exact_positive(self) -> None:
        positives = [
            "да", "ага", "угу", "конечно", "подтверждаю",
            "выполняй", "давай", "точно", "верно", "ок", "окей", "yes", "валяй",
        ]
        for word in positives:
            with self.subTest(word=word):
                self.assertEqual(parse_confirmation(word), ConfirmationResult.TRUE)
                self.assertEqual(parse_confirmation(f"  {word.upper()}! "), ConfirmationResult.TRUE)

    def test_exact_negative(self) -> None:
        negatives = [
            "нет", "не надо", "отмена", "отмени", "стоп",
            "не выполняй", "не нужно", "не стоит", "отказ", "no",
        ]
        for word in negatives:
            with self.subTest(word=word):
                self.assertEqual(parse_confirmation(word), ConfirmationResult.FALSE)
                self.assertEqual(parse_confirmation(f"  {word.upper()}! "), ConfirmationResult.FALSE)

    def test_phrases(self) -> None:
        self.assertEqual(parse_confirmation("ну давай конечно"), ConfirmationResult.TRUE)
        self.assertEqual(parse_confirmation("да давай"), ConfirmationResult.TRUE)
        self.assertEqual(parse_confirmation("нет не надо"), ConfirmationResult.FALSE)
        self.assertEqual(parse_confirmation("лучше не надо"), ConfirmationResult.FALSE)

    def test_stt_acoustic_recovery(self) -> None:
        # Acoustic variations and phonetic errors from Whisper
        self.assertEqual(parse_confirmation("отверждаю"), ConfirmationResult.TRUE)
        self.assertEqual(parse_confirmation("подверждаю"), ConfirmationResult.TRUE)
        self.assertEqual(parse_confirmation("конешно"), ConfirmationResult.TRUE)
        self.assertEqual(parse_confirmation("валяи"), ConfirmationResult.TRUE)
        self.assertEqual(parse_confirmation("агата да"), ConfirmationResult.TRUE)
        self.assertEqual(parse_confirmation("агата нет"), ConfirmationResult.FALSE)
        self.assertEqual(parse_confirmation("отменит"), ConfirmationResult.FALSE)

    def test_unknown_phrases(self) -> None:
        unknowns = [
            "что такое квантовая механика",
            "почему небо синее",
            "какая завтра погода",
            "расскажи анекдот",
        ]
        for phrase in unknowns:
            with self.subTest(phrase=phrase):
                self.assertEqual(parse_confirmation(phrase), ConfirmationResult.UNKNOWN)

    def test_empty_or_whitespace(self) -> None:
        self.assertEqual(parse_confirmation(""), ConfirmationResult.UNKNOWN)
        self.assertEqual(parse_confirmation("   "), ConfirmationResult.UNKNOWN)


if __name__ == "__main__":
    unittest.main()
