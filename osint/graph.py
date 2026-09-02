"""In-memory evidence graph suitable for JSON export and later visualization."""

from __future__ import annotations

from osint.context import InvestigationContext


def export_graph(context: InvestigationContext) -> dict[str, list[dict]]:
    return {
        "nodes": [entity.to_dict() for entity in context.entities.values()],
        "edges": [relation.to_dict() for relation in context.relations.values()],
    }
