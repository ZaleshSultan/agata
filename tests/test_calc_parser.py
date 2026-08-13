"""
Unit tests for calc_parser.py
"""

import unittest
from calc_parser import (
    format_number_for_speech,
    number_to_russian_words,
    parse_natural_math_to_expression,
    replace_words_with_numbers,
    safe_evaluate_math,
)


class TestCalcParser(unittest.TestCase):

    def test_numeral_word_replacement(self) -> None:
        self.assertEqual(replace_words_with_numbers("ноль"), "0")
        self.assertEqual(replace_words_with_numbers("двенадцать плюс семнадцать"), "12 плюс 17")
        self.assertEqual(replace_words_with_numbers("двадцать пять умножить на восемь"), "25 умножить на 8")
        self.assertEqual(replace_words_with_numbers("сто двадцать четыре"), "124")
        self.assertEqual(replace_words_with_numbers("две тысячи пятьсот"), "2500")
        self.assertEqual(replace_words_with_numbers("один миллион двести тысяч"), "1200000")

    def test_natural_math_expressions(self) -> None:
        # Addition
        self.assertEqual(parse_natural_math_to_expression("двенадцать плюс семнадцать"), "12 + 17")
        self.assertEqual(parse_natural_math_to_expression("посчитай 12 + 17"), "12 + 17")

        # Multiplication
        self.assertEqual(parse_natural_math_to_expression("двадцать пять умножить на восемь"), "25 * 8")
        self.assertEqual(parse_natural_math_to_expression("умножь пять на семь"), "5 * 7")

        # Division
        self.assertEqual(parse_natural_math_to_expression("сто двадцать разделить на четыре"), "120 / 4")
        self.assertEqual(parse_natural_math_to_expression("сто разделить на четыре"), "100 / 4")

        # Subtraction
        self.assertEqual(parse_natural_math_to_expression("двадцать пять минус семь"), "25 - 7")

        # Powers
        self.assertEqual(parse_natural_math_to_expression("три в квадрате"), "3 ** 2")
        self.assertEqual(parse_natural_math_to_expression("два в кубе"), "2 ** 3")
        self.assertEqual(parse_natural_math_to_expression("пять в степени три"), "5 ** 3")

        # Roots
        self.assertEqual(parse_natural_math_to_expression("корень из девяти"), "sqrt(9)")
        self.assertEqual(parse_natural_math_to_expression("квадратный корень из шестнадцати"), "sqrt(16)")

    def test_safe_evaluation(self) -> None:
        # Addition
        success, val, answer = safe_evaluate_math("12 + 17")
        self.assertTrue(success)
        self.assertEqual(val, 29)
        self.assertEqual(answer, "29")
        self.assertEqual(format_number_for_speech(val), "Двадцать девять.")

        # Multiplication
        success, val, _ = safe_evaluate_math("25 * 8")
        self.assertTrue(success)
        self.assertEqual(val, 200)

        # Division
        success, val, _ = safe_evaluate_math("120 / 4")
        self.assertTrue(success)
        self.assertEqual(val, 30)

        # Subtraction
        success, val, _ = safe_evaluate_math("25 - 7")
        self.assertTrue(success)
        self.assertEqual(val, 18)

        # Power
        success, val, _ = safe_evaluate_math("3 ** 2")
        self.assertTrue(success)
        self.assertEqual(val, 9)

        # Sqrt
        success, val, _ = safe_evaluate_math("sqrt(9)")
        self.assertTrue(success)
        self.assertEqual(val, 3)

    def test_whitelist_rejection(self) -> None:
        # Arbitrary code execution prevention
        expr = "__import__('os').system('dir')"
        success, val, err = safe_evaluate_math(expr)
        self.assertFalse(success)

        parsed = parse_natural_math_to_expression("выполни rm -rf")
        self.assertIsNone(parsed)

    def test_number_to_russian_words(self) -> None:
        self.assertEqual(number_to_russian_words(0), "ноль")
        self.assertEqual(number_to_russian_words(29), "двадцать девять")
        self.assertEqual(number_to_russian_words(100), "сто")
        self.assertEqual(number_to_russian_words(200), "двести")
        self.assertEqual(number_to_russian_words(1234), "одна тысяча двести тридцать четыре")
        self.assertEqual(number_to_russian_words(-5), "минус пять")

    def test_format_number_for_speech(self) -> None:
        self.assertEqual(format_number_for_speech(29), "Двадцать девять.")
        self.assertEqual(format_number_for_speech(200), "Двести.")


if __name__ == "__main__":
    unittest.main()
