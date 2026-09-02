"""Small TTL cache for requests and collector results; no secrets are persisted."""

from __future__ import annotations

from dataclasses import dataclass
from time import monotonic
from typing import Any


@dataclass(slots=True)
class _Cached:
    value: Any
    expires_at: float


class InvestigationCache:
    def __init__(self, ttl_seconds: float = 900) -> None:
        self._ttl, self._items, self.hits = ttl_seconds, {}, 0

    def get(self, key: str) -> Any | None:
        item = self._items.get(key)
        if item is None or item.expires_at < monotonic():
            self._items.pop(key, None); return None
        self.hits += 1; return item.value

    def set(self, key: str, value: Any) -> None:
        self._items[key] = _Cached(value, monotonic() + self._ttl)
