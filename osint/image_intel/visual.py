"""Optional AI visual-clue adapter; it emits observations, not identity decisions."""

from __future__ import annotations

from osint.models import AssertionStatus, Evidence, Observation
from osint.providers.base import AIProvider


async def analyze_visual_clues(image: bytes, subject_id: str, provider: AIProvider | None) -> tuple[list[Evidence], list[Observation]]:
    if provider is None:
        return [], [Observation(subject_id, "Visual analysis was skipped because no OSINT AI provider is configured.", status=AssertionStatus.UNVERIFIED, source="image_visual")]
    analysis = await provider.analyze_image(image, "List only visible geographic, textual, architectural, road, terrain, landmark, and weather clues. Do not identify people or make certainty claims. Return concise structured facts.")
    proof = Evidence("Optional AI visual-clue analysis.", source=provider.name, excerpt=analysis.text, status=AssertionStatus.UNVERIFIED)
    return [proof], [Observation(subject_id, "AI returned visual clues for analyst verification.", status=AssertionStatus.UNVERIFIED, confidence=0.3, evidence_ids=[proof.id], source=provider.name)]
