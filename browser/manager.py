"""Lifecycle management for a shared, reusable Playwright browser context."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import config

logger = logging.getLogger(__name__)


class BrowserUnavailableError(RuntimeError):
    """Raised when a browser operation cannot safely be started."""


class BrowserManager:
    """Own one Playwright instance, browser and context for an application run."""

    def __init__(self) -> None:
        self._playwright: Any | None = None
        self._browser: Any | None = None
        self._context: Any | None = None
        # A list preserves creation order so the page cap closes the oldest
        # page deterministically rather than an arbitrary set member.
        self._pages: list[Any] = []
        self._lock: asyncio.Lock | None = None

    async def start(self) -> None:
        """Start Chromium once; subsequent calls safely reuse its context."""
        if self._context is not None:
            return
        if not config.BROWSER_ENABLED:
            raise BrowserUnavailableError("Browser is disabled by BROWSER_ENABLED.")
        if self._lock is None:
            self._lock = asyncio.Lock()
        async with self._lock:
            if self._context is not None:
                return
            try:
                from playwright.async_api import async_playwright

                self._playwright = await async_playwright().start()
                self._browser = await self._playwright.chromium.launch(headless=config.BROWSER_HEADLESS)
                self._context = await self._browser.new_context(
                    user_agent=(
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
                    )
                )
                self._context.set_default_timeout(config.BROWSER_TIMEOUT_MS)
                self._context.set_default_navigation_timeout(config.BROWSER_TIMEOUT_MS)
                logger.info("[BROWSER] shared Chromium context started (headless=%s)", config.BROWSER_HEADLESS)
            except ImportError as exc:
                await self.close()
                raise BrowserUnavailableError("Playwright is not installed. Run: pip install -r requirements.txt") from exc
            except Exception as exc:
                await self.close()
                raise BrowserUnavailableError(f"Browser failed to start: {exc}") from exc

    async def get_page(self) -> Any:
        """Return a tracked page, closing the oldest pages before exceeding the cap."""
        await self.start()
        assert self._context is not None
        while len(self._pages) >= max(1, config.BROWSER_MAX_PAGES):
            oldest = self._pages.pop(0)
            await oldest.close()
        try:
            page = await self._context.new_page()
        except Exception as exc:
            raise BrowserUnavailableError(f"Unable to create a browser page: {exc}") from exc
        self._pages.append(page)

        def forget_page() -> None:
            if page in self._pages:
                self._pages.remove(page)

        page.on("close", forget_page)
        return page

    async def close(self) -> None:
        """Close pages, browser and Playwright runner; safe after partial startup."""
        pages, self._pages = self._pages, []
        for page in pages:
            try:
                await page.close()
            except Exception:
                pass
        browser, self._browser = self._browser, None
        context, self._context = self._context, None
        if browser is not None:
            try:
                await browser.close()
            except Exception as exc:
                logger.debug("[BROWSER] browser close failed: %s", exc)
        elif context is not None:
            try:
                await context.close()
            except Exception:
                pass
        playwright, self._playwright = self._playwright, None
        if playwright is not None:
            try:
                await playwright.stop()
            except Exception as exc:
                logger.debug("[BROWSER] Playwright stop failed: %s", exc)
