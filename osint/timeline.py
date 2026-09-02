"""Chronological projection over sourced observations."""

from __future__ import annotations

from osint.context import InvestigationContext


def build_timeline(context: InvestigationContext) -> list[dict]:
    items = [observation.to_dict() for observation in context.observations.values()]
    return sorted(items, key=lambda item: item["timestamp"])
