"""Location hypotheses are intentionally non-conclusive."""

from __future__ import annotations

from osint.models import AssertionStatus, Observation


def hypothesis(subject_id: str, candidate: str, reasons: list[str], confidence: float) -> Observation:
    return Observation(subject_id, f"Possible location: {candidate}. Reasons: {'; '.join(reasons)}", status=AssertionStatus.POSSIBLE, confidence=min(0.8, confidence), source="image_geolocation")
