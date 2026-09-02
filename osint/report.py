"""Markdown report which keeps observations, inferences and unknowns separate."""

from __future__ import annotations

from osint.context import InvestigationContext
from osint.models import AssertionStatus


def render_report(context: InvestigationContext) -> str:
    def section(title: str, status: AssertionStatus) -> list[str]:
        lines = [f"## {title}"]
        items = [item for item in context.observations.values() if item.status is status]
        lines.extend([f"- {item.claim} (confidence {item.confidence:.2f})" for item in items] or ["- None."])
        return lines
    lines = [f"# Investigation {context.investigation_id}", f"Target: `{context.target}` ({context.target_type})", "", "## Entities"]
    labels = {"username": "Usernames", "email": "Emails", "social": "Social profiles", "website": "Websites", "organization": "Organizations", "location": "Locations", "image": "Image findings", "event": "Timeline events"}
    for entity_type, heading in labels.items():
        matches = [item for item in context.entities.values() if item.type.value == entity_type]
        if matches:
            lines.extend([f"### {heading}", *[f"- {item.label or item.value}: {item.value}" for item in matches]])
    lines.extend(["", *section("Confirmed observations", AssertionStatus.OBSERVED), "", *section("Probable associations", AssertionStatus.INFERRED), "", *section("Unverified findings", AssertionStatus.POSSIBLE), "", "## Relationships"])
    lines.extend([f"- {item.type}: {item.reason} (confidence {item.confidence:.2f})" for item in context.relations.values()] or ["- None."])
    lines.append(""); lines.append("## Sources")
    lines.extend(f"- {proof.source_url or proof.source}: {proof.claim}" for proof in context.evidence.values())
    lines.extend(["", "## Metrics", f"- Runtime: {context.budget.elapsed_seconds:.2f}s", f"- Requests: {context.budget.requests}", f"- Cache hits: {context.cache.hits}", f"- Entities: {len(context.entities)}"])
    return "\n".join(lines) + "\n"
