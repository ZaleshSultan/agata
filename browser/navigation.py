"""Navigation helpers built on the shared browser manager."""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import quote_plus, urlsplit
from typing import Any

import config
from browser.manager import BrowserManager


@dataclass(slots=True)
class NavigationResult:
    success: bool
    url: str
    page: Any | None = None
    title: str = ""
    error: str | None = None

    def as_dict(self) -> dict[str, str | bool]:
        result: dict[str, str | bool] = {"success": self.success, "url": self.url}
        if self.title:
            result["title"] = self.title
        if self.error:
            result["error"] = self.error
        return result


class Navigation:
    def __init__(self, manager: BrowserManager) -> None:
        self._manager = manager

    @staticmethod
    def _normalise_url(url: str) -> str:
        candidate = (url or "").strip()
        parts = urlsplit(candidate)
        if parts.scheme not in {"http", "https"} or not parts.netloc:
            raise ValueError("Only absolute http(s) URLs are allowed.")
        return candidate

    async def open_url(self, url: str) -> NavigationResult:
        """Navigate a managed page and expose errors as structured data."""
        try:
            target = self._normalise_url(url)
        except ValueError as exc:
            return NavigationResult(False, url, error=str(exc))
        page = None
        try:
            page = await self._manager.get_page()
            await page.goto(target, wait_until="domcontentloaded", timeout=config.BROWSER_TIMEOUT_MS)
            return NavigationResult(True, page.url, page=page, title=await page.title())
        except Exception as exc:
            if page is not None:
                await page.close()
            return NavigationResult(False, target, error=f"Browser navigation failed: {exc}")

    async def search(self, query: str) -> NavigationResult:
        """Open a DuckDuckGo results page without coupling search to an LLM."""
        cleaned = (query or "").strip()
        if not cleaned:
            return NavigationResult(False, "", error="Search query is empty.")
        return await self.open_url(f"https://duckduckgo.com/?q={quote_plus(cleaned)}")
