"""OCR is optional; no bundled heavyweight model or cloud upload is required."""
from __future__ import annotations
from osint.models import AssertionStatus, Observation
def unavailable(subject_id: str) -> Observation: return Observation(subject_id, "OCR is not configured; no text was inferred.", status=AssertionStatus.UNVERIFIED, source="image_ocr")
