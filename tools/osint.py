"""Thin synchronous bridge: Agata sees an investigation as one tool call."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from osint import investigate


def investigate_person(target: str, target_type: str = "auto", depth: str = "standard", image_path: str | None = None) -> str:
    """Run one bounded public-source investigation without exposing internal steps."""
    image = None
    if image_path:
        try: image = Path(image_path).read_bytes()
        except OSError as exc: return json.dumps({"success": False, "error": f"Unable to read image: {exc}"}, ensure_ascii=False)
    try:
        result = asyncio.run(investigate(target, target_type=target_type, image=image, depth=depth))
        return json.dumps(result, ensure_ascii=False)
    except Exception as exc:
        return json.dumps({"success": False, "error": f"Investigation failed: {exc}"}, ensure_ascii=False)


TOOL_DEFINITION: dict = {
    "function": {
        "name": "investigate_person",
        "description": "Запускает одну ограниченную OSINT-проверку по публичным источникам и возвращает структурированный отчёт. Не используй для приватных аккаунтов, трекинга или несанкционированного доступа.",
        "parameters": {
            "type": "object",
            "properties": {
                "target": {"type": "string", "description": "Предоставленный пользователем username, email, domain, URL, IP или имя для поиска по публичным источникам."},
                "target_type": {"type": "string", "enum": ["auto", "username", "email", "phone", "domain", "website", "ip", "person"], "default": "auto"},
                "depth": {"type": "string", "enum": ["quick", "standard", "deep"], "default": "standard"},
                "image_path": {"type": "string", "description": "Optional local image path supplied by the user; only metadata and non-identifying visual clues are analyzed."},
            },
            "required": ["target"],
        },
    },
    "requires_confirmation": False,
}
