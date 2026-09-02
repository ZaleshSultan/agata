"""Canonicalisation rules shared by every collector."""

from __future__ import annotations

import re
from urllib.parse import urlsplit, urlunsplit

from osint.models import EntityType


def normalize(value: str, entity_type: EntityType) -> str:
    text = (value or "").strip()
    if entity_type is EntityType.EMAIL:
        return text.casefold()
    if entity_type is EntityType.USERNAME:
        return text.lstrip("@").casefold()
    if entity_type in {EntityType.DOMAIN, EntityType.IP}:
        return text.casefold().rstrip(".")
    if entity_type in {EntityType.WEBSITE, EntityType.SOCIAL}:
        parts = urlsplit(text if "://" in text else f"https://{text}")
        return urlunsplit((parts.scheme.casefold(), parts.netloc.casefold(), parts.path.rstrip("/"), "", ""))
    if entity_type is EntityType.PHONE:
        return re.sub(r"[^0-9+]", "", text)
    return " ".join(text.split())


def infer_target_type(value: str) -> EntityType:
    value = (value or "").strip()
    if re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", value): return EntityType.EMAIL
    if re.fullmatch(r"(?:\+?\d[\d\s().-]{6,}\d)", value): return EntityType.PHONE
    if re.fullmatch(r"(?:\d{1,3}\.){3}\d{1,3}", value): return EntityType.IP
    if re.fullmatch(r"(?:https?://)?[^\s/]+\.[a-z]{2,}(?:/[^\s]*)?", value, re.I): return EntityType.WEBSITE
    return EntityType.USERNAME
