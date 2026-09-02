"""Reserved deterministic landmark evidence boundary."""
from __future__ import annotations
from osint.models import Observation
def no_match(subject_id: str) -> Observation: return Observation(subject_id, "No verified landmark match was produced.", source="landmarks")
