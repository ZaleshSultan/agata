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
        expression: Строка с математическим выражением, например "847 * 23" или "двенадцать плюс семнадцать".

    Returns:
        Результат вычисления в виде строки (краткий для речи).
    """
    from calc_parser import parse_natural_math_to_expression, safe_evaluate_math

    expr = expression.strip().strip("\"'")
    # Check if expression is in words e.g. "двенадцать плюс семнадцать"
    parsed = parse_natural_math_to_expression(expr)
    target_expr = parsed if parsed else expr

    success, value, answer = safe_evaluate_math(target_expr)
    if success:
        logger.info("[CALC] Evaluated: '%s' -> '%s' (value=%s, answer='%s')", expression, target_expr, value, answer)
        return answer
    else:
        logger.warning("[CALC] Evaluation failed for '%s': %s", expression, answer)
        return answer


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
