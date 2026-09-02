"""Transparent weighted scores, never an automatic identity claim."""

from __future__ import annotations

CORRELATION_WEIGHTS = {"username": 0.20, "display_name": 0.20, "email": 0.35, "website": 0.20, "bio": 0.10, "avatar": 0.15}

def confidence(signals: set[str]) -> float:
    return min(0.95, round(sum(CORRELATION_WEIGHTS.get(signal, 0.0) for signal in signals), 2))
