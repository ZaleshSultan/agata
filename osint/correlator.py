"""Create explicitly tentative relation candidates from multiple public signals."""

from __future__ import annotations

from osint.context import InvestigationContext
from osint.models import AssertionStatus, Relation
from osint.scoring import confidence


class Correlator:
    def correlate(self, context: InvestigationContext) -> list[Relation]:
        profiles = [item for item in context.entities.values() if item.type.value == "social"]
        created: list[Relation] = []
        for index, first in enumerate(profiles):
            for second in profiles[index + 1:]:
                signals: set[str] = set()
                if first.attributes.get("username") and first.attributes.get("username") == second.attributes.get("username"): signals.add("username")
                if first.attributes.get("display_name") and first.attributes.get("display_name") == second.attributes.get("display_name"): signals.add("display_name")
                if first.attributes.get("website") and first.attributes.get("website") == second.attributes.get("website"): signals.add("website")
                score = confidence(signals)
                if len(signals) >= 2 and score >= 0.35:
                    created.append(Relation(first.id, second.id, "same_as_candidate", score, status=AssertionStatus.POSSIBLE, reason="Matching public signals: " + ", ".join(sorted(signals))))
        return created
