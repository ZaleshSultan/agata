"""Reverse-image lookup intentionally remains opt-in and provider-gated."""
from __future__ import annotations
from osint.models import Observation, AssertionStatus
def unavailable(subject_id: str) -> Observation: return Observation(subject_id, "Reverse-image search is not configured; no third-party upload was performed.", status=AssertionStatus.UNVERIFIED, source="reverse_image")
