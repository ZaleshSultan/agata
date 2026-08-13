"""
Tests for Domain-Aware STT Corrections.
"""

import pytest
from stt_corrections import (
    correct_command_text,
    correct_math_text,
    correct_confirmation_text,
)


def test_command_corrections():
    assert correct_command_text("открой телеграмм") == "открой телеграм"
    assert correct_command_text("закрой телеграмму") == "закрой телеграм"
    assert correct_command_text("открой спотик") == "открой spotify"
    assert correct_command_text("открой дискордик") == "открой дискорд"
    assert correct_command_text("открой хромиум") == "открой хром"
    assert correct_command_text("открой ютубчик") == "открой youtube"


def test_math_corrections():
    assert correct_math_text("25 тус 17") == "25 плюс 17"
    assert correct_math_text("25 тус 17 умножит на 2") == "25 плюс 17 умножь на 2"
    assert correct_math_text("100 делить на 4") == "100 разделить на 4"
    assert correct_math_text("5 с четвертой степени") == "5 в четвертой степени"
    assert correct_math_text("5 с 4 степени") == "5 в 4 степени"
    assert correct_math_text("корень с 16") == "корень из 16"


def test_confirmation_corrections():
    assert correct_confirmation_text("отверждаю") == "подтверждаю"
    assert correct_confirmation_text("подверждаю") == "подтверждаю"
    assert correct_confirmation_text("отверди") == "подтверждаю"
    assert correct_confirmation_text("конешно") == "конечно"
    assert correct_confirmation_text("валяи") == "валяй"
    assert correct_confirmation_text("агата да") == "да"
    assert correct_confirmation_text("агата нет") == "нет"
    assert correct_confirmation_text("отменит") == "отменить"
