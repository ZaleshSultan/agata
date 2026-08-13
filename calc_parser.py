"""
Natural-language Russian Numeral & Mathematical Expression Parser.

Converts Russian spoken math (e.g. "двенадцать плюс семнадцать", "двадцать пять умножить на восемь",
"корень из девяти", "три в квадрате") into safe mathematical expressions and computes results
without calling any LLM.
"""

import logging
import math
import re
from typing import Any

logger = logging.getLogger(__name__)

# ── СЛОВАРЬ РУССКИХ ЧИСЛИТЕЛЬНЫХ ──────────────────────────────────────────

_NUMERAL_UNITS: dict[str, int] = {
    "ноль": 0,
    "нуль": 0,
    "один": 1,
    "одна": 1,
    "одно": 1,
    "одну": 1,
    "одни": 1,
    "первый": 1,
    "первая": 1,
    "первое": 1,
    "два": 2,
    "две": 2,
    "двух": 2,
    "второй": 2,
    "вторая": 2,
    "второе": 2,
    "три": 3,
    "трех": 3,
    "трёх": 3,
    "третий": 3,
    "третья": 3,
    "третье": 3,
    "четыре": 4,
    "четырех": 4,
    "четырёх": 4,
    "четвертый": 4,
    "четвёртая": 4,
    "пять": 5,
    "пяти": 5,
    "пятый": 5,
    "пятая": 5,
    "шесть": 6,
    "шести": 6,
    "шестой": 6,
    "шестая": 6,
    "семь": 7,
    "семи": 7,
    "седьмой": 7,
    "седьмая": 7,
    "восемь": 8,
    "восьми": 8,
    "восьмой": 8,
    "восьмая": 8,
    "девять": 9,
    "девяти": 9,
    "девятый": 9,
    "девятая": 9,
}

_NUMERAL_TEENS: dict[str, int] = {
    "десять": 10,
    "десяти": 10,
    "десятый": 10,
    "одиннадцать": 11,
    "одиннадцати": 11,
    "двенадцать": 12,
    "двенадцати": 12,
    "тринадцать": 13,
    "тринадцати": 13,
    "четырнадцать": 14,
    "четырнадцати": 14,
    "пятнадцать": 15,
    "пятнадцати": 15,
    "шестнадцать": 16,
    "шестнадцати": 16,
    "семнадцать": 17,
    "семнадцати": 17,
    "восемнадцать": 18,
    "восемнадцати": 18,
    "девятнадцать": 19,
    "девятнадцати": 19,
}

_NUMERAL_TENS: dict[str, int] = {
    "двадцать": 20,
    "двадцати": 20,
    "тридцать": 30,
    "тридцати": 30,
    "сорок": 40,
    "сорока": 40,
    "пятьдесят": 50,
    "пятидесяти": 50,
    "шестьдесят": 60,
    "шестидесяти": 60,
    "семьдесят": 70,
    "семидесяти": 70,
    "восемьдесят": 80,
    "восьмидесяти": 80,
    "девяносто": 90,
    "девяноста": 90,
}

_NUMERAL_HUNDREDS: dict[str, int] = {
    "сто": 100,
    "ста": 100,
    "двести": 200,
    "двухсот": 200,
    "триста": 300,
    "трехсот": 300,
    "трёхсот": 300,
    "четыреста": 400,
    "четырехсот": 400,
    "пятьсот": 500,
    "пятисот": 500,
    "шестьсот": 600,
    "шестисот": 600,
    "семьсот": 700,
    "семисот": 700,
    "восемьсот": 800,
    "восьмисот": 800,
    "девятьсот": 900,
    "девятисот": 900,
}

_NUMERAL_MULTIPLIERS: dict[str, int] = {
    "тысяча": 1000,
    "тысячи": 1000,
    "тысяч": 1000,
    "тысячу": 1000,
    "тысячам": 1000,
    "тысячами": 1000,
    "миллион": 1000000,
    "миллиона": 1000000,
    "миллионов": 1000000,
    "миллиону": 1000000,
    "миллионами": 1000000,
    "миллиард": 1000000000,
    "миллиарда": 1000000000,
    "миллиардов": 1000000000,
}

# Все числительные слова
_ALL_NUMERAL_WORDS: dict[str, int] = {
    **_NUMERAL_UNITS,
    **_NUMERAL_TEENS,
    **_NUMERAL_TENS,
    **_NUMERAL_HUNDREDS,
}

# ── МАТЕМАТИЧЕСКИЕ ОПЕРАЦИИ В РЕЧИ ──────────────────────────────────────────

_CALC_COMMAND_PREFIXES = (
    "посчитай", "вычисли", "сколько будет", "реши", "найди значение",
    "подсчитай", "рассчитай", "скажи сколько будет",
)

_ALLOWED_EXPR_CHARS = re.compile(r"^[0-9\s+\-*/().,%*sqrt]+$", re.IGNORECASE)


def parse_russian_number_sequence(words: list[str]) -> tuple[int | float | None, int]:
    """
    Parse consecutive Russian numeral words starting at words[0].

    Returns:
        (parsed_number, consumed_word_count) or (None, 0)
    """
    if not words:
        return None, 0

    total = 0
    current = 0
    consumed = 0

    for word in words:
        w = word.lower().strip(".,!?:;")
        if w in _ALL_NUMERAL_WORDS:
            val = _ALL_NUMERAL_WORDS[w]
            current += val
            consumed += 1
        elif w in _NUMERAL_MULTIPLIERS:
            mult = _NUMERAL_MULTIPLIERS[w]
            if current == 0:
                current = 1
            total += current * mult
            current = 0
            consumed += 1
        else:
            break

    if consumed > 0:
        return total + current, consumed

    return None, 0


def replace_words_with_numbers(text: str) -> str:
    """
    Replace Russian numeral words in text with numeric digits.
    e.g. "сто двадцать четыре плюс двенадцать" -> "124 плюс 12"
    """
    words = text.split()
    output_tokens: list[str] = []
    i = 0

    while i < len(words):
        num, count = parse_russian_number_sequence(words[i:])
        if count > 0 and num is not None:
            output_tokens.append(str(num))
            i += count
        else:
            output_tokens.append(words[i])
            i += 1

    return " ".join(output_tokens)


_ORDINAL_EXPONENTS: dict[str, int] = {
    "первой": 1, "первую": 1, "первая": 1,
    "второй": 2, "вторую": 2, "вторая": 2,
    "третьей": 3, "третью": 3, "третья": 3,
    "четвертой": 4, "четвёртой": 4, "четвертую": 4, "четвёртую": 4, "четвертая": 4, "четвёртая": 4,
    "пятой": 5, "пятую": 5, "пятая": 5,
    "шестой": 6, "шестую": 6, "шестая": 6,
    "седьмой": 7, "седьмую": 7, "седьмая": 7,
    "восьмой": 8, "восьмую": 8, "восьмая": 8,
    "девятой": 9, "девятую": 9, "девятая": 9,
    "десятой": 10, "десятую": 10, "десятая": 10,
}


def parse_natural_math_to_expression(text: str) -> str | None:
    """
    Convert a natural language Russian math phrase into an executable mathematical expression.

    Examples:
        "двенадцать плюс семнадцать" -> "12 + 17"
        "двадцать пять умножить на восемь" -> "25 * 8"
        "25 плюс 17 умножить на 2" -> "25 + 17 * 2"
        "пять в четвертой степени" -> "5 ** 4"
        "5 в четвёртой" -> "5 ** 4"
        "сто двадцать разделить на четыре" -> "120 / 4"
        "двадцать пять минус семь" -> "25 - 7"
        "три в квадрате" -> "3 ** 2"
        "два в кубе" -> "2 ** 3"
        "корень из девяти" -> "sqrt(9)"
        "пять в степени три" -> "5 ** 3"
        "двадцать процентов от ста" -> "100 * (20 / 100)"
        "12 + 17" -> "12 + 17"
    """
    if not text or not text.strip():
        return None

    # Apply math-specific domain corrections first (e.g. "тус" -> "плюс", "умножит" -> "умножь", "5 с четвертой степени")
    from stt_corrections import correct_math_text
    corrected_input = correct_math_text(text)

    cleaned = corrected_input.strip().lower().rstrip("?").strip()

    # Remove assistant address and command prefixes
    cleaned = re.sub(r"^\s*агата\s*[,!]?\s*", "", cleaned, flags=re.IGNORECASE)
    for prefix in _CALC_COMMAND_PREFIXES:
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix):].strip(" ,:")
            break

    # First convert numeral words to digits
    text_with_digits = replace_words_with_numbers(cleaned)

    # ── Handle special constructs ──────────────────────────────────────────

    # "корень из X" / "квадратный корень из X"
    sqrt_match = re.search(r"(?:квадратный\s+)?корень\s+из\s+([0-9.]+)", text_with_digits)
    if sqrt_match:
        val = sqrt_match.group(1)
        text_with_digits = text_with_digits[:sqrt_match.start()] + f"sqrt({val})" + text_with_digits[sqrt_match.end():]

    # "X в квадрате" -> "X ** 2"
    text_with_digits = re.sub(r"([0-9.]+)\s+в\s+квадрате", r"\1 ** 2", text_with_digits)

    # "X в кубе" -> "X ** 3"
    text_with_digits = re.sub(r"([0-9.]+)\s+в\s+кубе", r"\1 ** 3", text_with_digits)

    # Ordinal powers: "X в четвёртой степени", "X в четвертой", "X в пятой", etc.
    for ord_word, pwr in _ORDINAL_EXPONENTS.items():
        # "X в {ord_word} степени" / "X в {ord_word}"
        pattern_with_deg = re.compile(rf"([0-9.]+)\s+в\s+{ord_word}(?:\s+степен[ьи])?", re.IGNORECASE)
        text_with_digits = pattern_with_deg.sub(rf"\1 ** {pwr}", text_with_digits)

    # "X в степени Y" / "X в Y степень" / "X в Y степени" -> "X ** Y"
    text_with_digits = re.sub(r"([0-9.]+)\s+в\s+степени\s+([0-9.]+)", r"\1 ** \2", text_with_digits)
    text_with_digits = re.sub(r"([0-9.]+)\s+в\s+([0-9.]+)\s+степен[ьи]", r"\1 ** \2", text_with_digits)
    text_with_digits = re.sub(r"([0-9.]+)\s+в\s+([0-9.]+)\s+степень", r"\1 ** \2", text_with_digits)

    # "X процент(ов) от Y" -> "Y * (X / 100)"
    pct_match = re.search(r"([0-9.]+)\s*%\s+от\s+([0-9.]+)|([0-9.]+)\s+процент(?:а|ов)?\s+от\s+([0-9.]+)", text_with_digits)
    if pct_match:
        pct = pct_match.group(1) or pct_match.group(3)
        base = pct_match.group(2) or pct_match.group(4)
        text_with_digits = text_with_digits[:pct_match.start()] + f"{base} * ({pct} / 100)" + text_with_digits[pct_match.end():]

    # "умножь X на Y" -> "X * Y"
    text_with_digits = re.sub(r"(?:умножь|умножить|перемножь|помножь)\s+([0-9.]+)\s+на\s+([0-9.]+)", r"\1 * \2", text_with_digits)
    # "раздели X на Y" -> "X / Y"
    text_with_digits = re.sub(r"(?:раздели|разделить|подели|поделить)\s+([0-9.]+)\s+на\s+([0-9.]+)", r"\1 / \2", text_with_digits)
    # "прибавь X к Y" -> "Y + X"
    text_with_digits = re.sub(r"(?:прибавь|прибавить)\s+([0-9.]+)\s+к\s+([0-9.]+)", r"\2 + \1", text_with_digits)
    # "вычти X из Y" -> "Y - X"
    text_with_digits = re.sub(r"(?:вычти|вычесть|отними|отнять)\s+([0-9.]+)\s+из\s+([0-9.]+)", r"\2 - \1", text_with_digits)

    # ── Standard operators ────────────────────────────────────────────────

    # Multiplications
    text_with_digits = re.sub(r"\s*(?:умножить\s+на|умножь\s+на|умноженное\s+на|помножить\s+на|умножить|умножь)\s*", " * ", text_with_digits)
    # Divisions
    text_with_digits = re.sub(r"\s*(?:разделить\s+на|раздели\s+на|поделить\s+на|подели\s+на|делённое\s+на|деленное\s+на|разделить|поделить)\s*", " / ", text_with_digits)
    # Additions
    text_with_digits = re.sub(r"\s*(?:плюс|прибавить|сложить\s+с|прибавь)\s*", " + ", text_with_digits)
    # Subtractions
    text_with_digits = re.sub(r"\s*(?:минус|вычесть|отнять|отними|вычти)\s*", " - ", text_with_digits)

    # Clean characters
    expr = text_with_digits.replace("×", "*").replace("÷", "/").replace(":", "/").strip()

    # Validate that the resulting string is an actual math expression containing digits and operators
    if not re.search(r"\d", expr):
        return None

    # Check that it contains at least one operator, sqrt, power, or is a pure arithmetic query
    has_operator = any(op in expr for op in ("+", "-", "*", "/", "**", "sqrt", "%"))
    if not has_operator:
        return None

    # Whitelist verification: only digits, spaces, and whitelisted symbols
    cleaned_check = re.sub(r"sqrt", "", expr, flags=re.IGNORECASE)
    if not _ALLOWED_EXPR_CHARS.fullmatch(cleaned_check):
        return None

    return expr.strip()


def safe_evaluate_math(expression: str) -> tuple[bool, Any, str]:
    """
    Safely evaluate a mathematical expression using numexpr or safe math.

    Returns:
        (success, numeric_value, formatted_string_answer)
    """
    expr = expression.strip().strip("\"'")

    # Whitelist check
    cleaned_check = re.sub(r"sqrt", "", expr, flags=re.IGNORECASE)
    if not _ALLOWED_EXPR_CHARS.fullmatch(cleaned_check):
        return False, None, f"Недопустимое выражение: '{expr}'"

    try:
        import numexpr  # type: ignore[import-untyped]
        # Replace sqrt if needed for numexpr
        numexpr_clean = expr.replace("sqrt", "sqrt")
        res = numexpr.evaluate(numexpr_clean)
        value = float(res)
    except Exception:
        # Fallback to safe math eval
        try:
            import math
            allowed_names = {"sqrt": math.sqrt, "abs": abs, "pow": pow, "sin": math.sin, "cos": math.cos}
            compiled = compile(expr, "<math>", "eval")
            for name in compiled.co_names:
                if name not in allowed_names:
                    raise ValueError(f"Disallowed name: {name}")
            value = eval(compiled, {"__builtins__": {}}, allowed_names)
        except Exception as exc:
            return False, None, f"Ошибка вычисления '{expr}': {exc}"

    # Format result nicely
    if isinstance(value, float):
        if value.is_integer():
            value = int(value)
        else:
            value = round(value, 4)

    return True, value, str(value)


# ── СЛОВАРЬ ДЛЯ ОЗВУЧИВАНИЯ ЧИСЕЛ ──────────────────────────────────────────

_ONES_WORDS = ["ноль", "один", "два", "три", "четыре", "пять", "шесть", "семь", "восемь", "девять"]
_TEENS_WORDS = ["десять", "одиннадцать", "двенадцать", "тринадцать", "четырнадцать", "пятнадцать",
                "шестнадцать", "семнадцать", "восемнадцать", "девятнадцать"]
_TENS_WORDS = ["", "", "двадцать", "тридцать", "сорок", "пятьдесят", "шестьдесят", "семьдесят", "восемьдесят", "девяносто"]
_HUNDREDS_WORDS = ["", "сто", "двести", "триста", "четыреста", "пятьсот", "шестьсот", "семьсот", "восемьсот", "девятьсот"]


def number_to_russian_words(n: int) -> str:
    """Convert an integer between -999,999,999 and 999,999,999 to spoken Russian words."""
    if n == 0:
        return "ноль"

    if n < 0:
        return "минус " + number_to_russian_words(-n)

    parts: list[str] = []

    # Millions
    millions = n // 1000000
    if millions > 0:
        parts.append(_format_triad(millions, is_female=False))
        if millions % 10 == 1 and millions % 100 != 11:
            parts.append("миллион")
        elif 2 <= millions % 10 <= 4 and not (12 <= millions % 100 <= 14):
            parts.append("миллиона")
        else:
            parts.append("миллионов")
        n %= 1000000

    # Thousands
    thousands = n // 1000
    if thousands > 0:
        parts.append(_format_triad(thousands, is_female=True))
        if thousands % 10 == 1 and thousands % 100 != 11:
            parts.append("тысяча")
        elif 2 <= thousands % 10 <= 4 and not (12 <= thousands % 100 <= 14):
            parts.append("тысячи")
        else:
            parts.append("тысяч")
        n %= 1000

    # Remainder (0..999)
    if n > 0:
        parts.append(_format_triad(n, is_female=False))

    return " ".join(p for p in parts if p).strip()


def _format_triad(val: int, is_female: bool = False) -> str:
    """Format a 3-digit number 1..999 in Russian."""
    res = []
    h = val // 100
    t = (val % 100) // 10
    u = val % 10

    if h > 0:
        res.append(_HUNDREDS_WORDS[h])

    if t == 1:
        res.append(_TEENS_WORDS[u])
    else:
        if t > 1:
            res.append(_TENS_WORDS[t])
        if u > 0:
            if u == 1:
                res.append("одна" if is_female else "один")
            elif u == 2:
                res.append("две" if is_female else "два")
            else:
                res.append(_ONES_WORDS[u])

    return " ".join(res)


def format_number_for_speech(val: int | float) -> str:
    """
    Format numeric result as concise speech text.
    e.g. 29 -> "Двадцать девять."
    """
    if isinstance(val, int) and -1000000 <= val <= 1000000:
        words = number_to_russian_words(val).capitalize()
        return f"{words}."
    return f"{val}."
