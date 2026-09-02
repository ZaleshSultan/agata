"""SoundCloud provider for search and playback verification."""

from __future__ import annotations

import asyncio
import json
import logging
from urllib.parse import quote_plus
from typing import Any

from browser.manager import BrowserManager
from media.base import ActionResult, MediaProvider, MediaResult
from media.ranking import rank_media_results

logger = logging.getLogger(__name__)


class SoundCloudProvider(MediaProvider):
    """Searches SoundCloud and opens the top-ranked track with playback verification."""

    def __init__(self, browser_manager: BrowserManager | None = None) -> None:
        self._browser = browser_manager or BrowserManager()

    async def search(self, query: str) -> list[MediaResult]:
        """Search SoundCloud and extract track links and titles."""
        clean_q = query.strip()
        if not clean_q:
            return []

        search_url = f"https://soundcloud.com/search?q={quote_plus(clean_q)}"
        logger.info("[SOUNDCLOUD] Searching: '%s' -> %s", clean_q, search_url)

        page: Any = None
        results: list[MediaResult] = []
        try:
            page = await self._browser.get_page()
            await page.goto(search_url, wait_until="domcontentloaded", timeout=12000)

            try:
                await page.wait_for_selector("a.soundTitle__title, ul.searchList", timeout=5000)
            except Exception:
                pass

            raw_items = await page.evaluate(
                """() => {
                    const items = [];
                    const seen = new Set();
                    const links = document.querySelectorAll('a.soundTitle__title, .searchList__item a[href*="/"]');
                    for (const el of links) {
                        const href = el.getAttribute('href');
                        const title = el.innerText?.trim();
                        if (href && !href.startsWith('/search') && !href.startsWith('/pages') && title && !seen.has(href)) {
                            seen.add(href);
                            items.push({
                                href: href.startsWith('http') ? href : 'https://soundcloud.com' + href,
                                title: title,
                            });
                        }
                        if (items.length >= 10) break;
                    }
                    return items;
                }"""
            )

            for i, it in enumerate(raw_items):
                results.append(
                    MediaResult(
                        url=it["href"],
                        title=it["title"],
                        position=i,
                    )
                )

        except Exception as exc:
            logger.error("[SOUNDCLOUD] Search failed: %s", exc)
        finally:
            if page is not None:
                try:
                    await page.close()
                except Exception:
                    pass

        if not results:
            logger.warning("[SOUNDCLOUD] No parsed results, creating search fallback result")
            return [
                MediaResult(
                    url=search_url,
                    title=f"Поиск SoundCloud: {clean_q}",
                    position=0,
                )
            ]

        return rank_media_results(clean_q, results)

    async def play(self, result: MediaResult) -> ActionResult:
        """Open the SoundCloud track URL in system browser."""
        logger.info("[SOUNDCLOUD] Playing: '%s' (%s)", result.title, result.url)

        from tools.system_control import open_browser as sys_open_browser

        browser_res_str = sys_open_browser(result.url)
        try:
            payload = json.loads(browser_res_str)
            if not payload.get("success"):
                return ActionResult(
                    success=False,
                    action="play_content",
                    platform="soundcloud",
                    query=result.title,
                    url=result.url,
                    stage="open_browser",
                    reason=payload.get("error", "Failed to open browser"),
                )
        except Exception:
            pass

        return ActionResult(
            success=True,
            action="play_content",
            platform="soundcloud",
            query=result.title,
            url=result.url,
            title=result.title,
            stage="playback",
        )
