"""
Калькулятор — безопасное вычисление математических выражений.

Использует numexpr вместо eval() для безопасности.
Побочные эффекты: нет.
"""

import logging

import numexpr  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)


def calculate(expression: str) -> str:
    """
    Вычислить математическое выражение.

    Args:
        expression: Строка с математическим выражением, например "847 * 23".

    Returns:
        Результат вычисления в виде строки.
    """
    # Убираем лишние пробелы и возможные кавычки
    expr = expression.strip().strip("\"'")
    try:
        result = numexpr.evaluate(expr)
        # numexpr возвращает numpy-массив, берём скалярное значение
        value = result.item() if hasattr(result, "item") else result
        logger.info("Вычислено: %s = %s", expr, value)
        return str(value)
    except Exception as exc:
        error_msg = f"Ошибка вычисления '{expr}': {exc}"
        logger.warning(error_msg)
        return error_msg


# ── JSON Schema для tool calling ──────────────────────────────────────────

TOOL_DEFINITION: dict = {
    "function": {
        "name": "calculate",
        "description": "Вычислить математическое выражение. Поддерживает +, -, *, /, **, sqrt и т.д.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Математическое выражение для вычисления, например '847 * 23' или 'sqrt(144)'",
                },
            },
            "required": ["expression"],
        },
    },
    "requires_confirmation": False,
}
