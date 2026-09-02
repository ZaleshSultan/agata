"""Deterministic image metadata extraction; metadata is never treated as proof."""

from __future__ import annotations

from io import BytesIO
from typing import Any

from PIL import Image, ExifTags

from osint.models import AssertionStatus, Evidence, Observation


def extract_metadata(image: bytes, subject_id: str) -> tuple[dict[str, Any], list[Evidence], list[Observation]]:
    with Image.open(BytesIO(image)) as opened:
        raw = opened.getexif()
        tags = {ExifTags.TAGS.get(key, str(key)): str(value) for key, value in raw.items()}
        metadata = {"format": opened.format, "width": opened.width, "height": opened.height, "exif": tags}
    proof = Evidence("Image metadata extracted locally; EXIF can be absent or altered.", source="image_metadata", status=AssertionStatus.OBSERVED)
    observation = Observation(subject_id, f"Image is {metadata['width']}x{metadata['height']} ({metadata['format']}).", evidence_ids=[proof.id], source="image_metadata")
    return metadata, [proof], [observation]
