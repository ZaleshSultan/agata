"""Evidence-driven assertion helpers."""

from __future__ import annotations

from osint.models import AssertionStatus, Evidence


def supports_inference(evidence: list[Evidence], minimum: int = 2) -> AssertionStatus:
    """One source is observed; independent corroboration is needed for inference."""
    sources = {item.source_url or item.source for item in evidence}
    return AssertionStatus.INFERRED if len(sources) >= minimum else AssertionStatus.POSSIBLE
