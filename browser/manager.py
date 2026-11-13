"""Lifecycle management for a shared, reusable Playwright browser context."""

from __future__ import annotations

import asyncio
import logging
import threading
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
        self._media_page: Any | None = None
        self._general_page: Any | None = None
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
                profile_path = config.DATA_DIR / "browser_profile"
                profile_path.mkdir(parents=True, exist_ok=True)

                chromium_args = [
                    "--autoplay-policy=no-user-gesture-required",
                    "--disable-blink-features=AutomationControlled",
                ]

                try:
                    self._context = await self._playwright.chromium.launch_persistent_context(
                        user_data_dir=str(profile_path),
                        headless=config.BROWSER_HEADLESS,
                        args=chromium_args,
                        user_agent=(
                            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
                        ),
                    )
                    self._browser = getattr(self._context, "browser", None)
                    logger.info("[BROWSER] persistent Chromium context started at %s (headless=%s)", profile_path, config.BROWSER_HEADLESS)
                except Exception as p_exc:
                    logger.warning("[BROWSER] persistent context fallback: %s", p_exc)
                    self._browser = await self._playwright.chromium.launch(
                        headless=config.BROWSER_HEADLESS,
                        args=chromium_args,
                    )
                    self._context = await self._browser.new_context(
                        user_agent=(
                            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
                        )
                    )
                    logger.info("[BROWSER] standard Chromium context started (headless=%s)", config.BROWSER_HEADLESS)

                self._context.set_default_timeout(config.BROWSER_TIMEOUT_MS)
                self._context.set_default_navigation_timeout(config.BROWSER_TIMEOUT_MS)
            except ImportError as exc:
                await self.close()
                raise BrowserUnavailableError("Playwright is not installed. Run: pip install -r requirements.txt") from exc
            except Exception as exc:
                await self.close()
                raise BrowserUnavailableError(f"Browser failed to start: {exc}") from exc

    async def get_media_page(self) -> Any:
        """Return the dedicated persistent page for media playback (YouTube/SoundCloud)."""
        await self.start()
        assert self._context is not None
        if self._media_page is not None and not self._media_page.is_closed():
            return self._media_page

        # Reuse existing page in context if already on youtube/soundcloud
        for p in self._context.pages:
            if not p.is_closed():
                url = p.url or ""
                if "youtube.com" in url or "soundcloud.com" in url:
                    self._media_page = p
                    return self._media_page

        try:
            self._media_page = await self._context.new_page()
        except Exception as exc:
            raise BrowserUnavailableError(f"Unable to create media page: {exc}") from exc

        def on_media_close() -> None:
            self._media_page = None

        self._media_page.on("close", on_media_close)
        return self._media_page

    async def get_general_page(self) -> Any:
        """Return the dedicated persistent page for general web browsing."""
        await self.start()
        assert self._context is not None
        if self._general_page is not None and not self._general_page.is_closed():
            return self._general_page

        try:
            self._general_page = await self._context.new_page()
        except Exception as exc:
            raise BrowserUnavailableError(f"Unable to create general page: {exc}") from exc

        def on_gen_close() -> None:
            self._general_page = None

        self._general_page.on("close", on_gen_close)
        return self._general_page

    async def get_page(self) -> Any:
        """Return a tracked page, closing the oldest pages before exceeding the cap."""
        await self.start()
        assert self._context is not None
        while len(self._pages) >= max(1, config.BROWSER_MAX_PAGES):
            oldest = self._pages.pop(0)
            if oldest != self._media_page:  # Never auto-evict the active media page
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
        if self._media_page is not None and not self._media_page.is_closed():
            try:
                await self._media_page.close()
            except Exception:
                pass
        self._media_page = None

        if self._general_page is not None and not self._general_page.is_closed():
            try:
                await self._general_page.close()
            except Exception:
                pass
        self._general_page = None

        browser, self._browser = self._browser, None
        context, self._context = self._context, None
        if context is not None:
            try:
                await context.close()
            except Exception as exc:
                logger.debug("[BROWSER] context close failed: %s", exc)
        if browser is not None:
            try:
                await browser.close()
            except Exception as exc:
                logger.debug("[BROWSER] browser close failed: %s", exc)
        playwright, self._playwright = self._playwright, None
        if playwright is not None:
            try:
                await playwright.stop()
            except Exception as exc:
                logger.debug("[BROWSER] Playwright stop failed: %s", exc)


_shared_browser_manager: BrowserManager | None = None
_browser_loop: asyncio.AbstractEventLoop | None = None
_browser_thread: threading.Thread | None = None
_loop_lock = threading.Lock()


def get_browser_manager() -> BrowserManager:
    """Return the application-level shared BrowserManager instance."""
    global _shared_browser_manager
    if _shared_browser_manager is None:
        _shared_browser_manager = BrowserManager()
    return _shared_browser_manager


def get_browser_loop() -> asyncio.AbstractEventLoop:
    """Return a single persistent asyncio event loop running on a dedicated thread for Playwright."""
    global _browser_loop, _browser_thread
    with _loop_lock:
        if _browser_loop is None or not _browser_loop.is_running():
            ready_event = threading.Event()

            def _run_loop():
                nonlocal ready_event
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                global _browser_loop
                _browser_loop = loop
                ready_event.set()
                loop.run_forever()

            _browser_thread = threading.Thread(target=_run_loop, name="playwright-browser-loop", daemon=True)
            _browser_thread.start()
            ready_event.wait()
        return _browser_loop


def run_browser_coro(coro: Any, timeout: float = 45.0) -> Any:
    """Execute an async coroutine on the dedicated persistent Playwright event loop."""
    loop = get_browser_loop()
    try:
        current_loop = asyncio.get_running_loop()
    except RuntimeError:
        current_loop = None

    if current_loop is loop:
        return coro

    future = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result(timeout=timeout)

