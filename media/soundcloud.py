"""SoundCloud provider for search and verified DOM playback."""

from __future__ import annotations

import logging
from urllib.parse import quote_plus
from typing import Any

from browser.manager import BrowserManager, get_browser_manager
from media.base import ActionResult, MediaProvider, MediaResult
from media.ranking import rank_media_results
from media.session import get_media_session

logger = logging.getLogger(__name__)


class SoundCloudProvider(MediaProvider):
    """Searches SoundCloud and plays tracks inside persistent MediaSession."""

    def __init__(self, browser_manager: BrowserManager | None = None) -> None:
        self._browser = browser_manager or get_browser_manager()
        self._session = get_media_session()

    async def search(
        self,
        query: str,
        content_type: str = "music",
        query_title: str = "",
        query_artist: str = "",
    ) -> list[MediaResult]:
        """Search SoundCloud and extract track links, titles, and artists."""
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
                            // Look for artist name
                            const parent = el.closest('.soundTitle__titleContainer, .searchList__item');
                            const userEl = parent ? parent.querySelector('.soundTitle__username, .sound__header .soundTitle__usernameText') : null;
                            const channel = userEl ? userEl.innerText.trim() : '';

                            items.push({
                                href: href.startsWith('http') ? href : 'https://soundcloud.com' + href,
                                title: title,
                                channel: channel,
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
                        channel=it.get("channel", ""),
                        position=i,
                        content_type="music",
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
            logger.warning("[SOUNDCLOUD] No parsed results for query '%s'", clean_q)
            return []

        return rank_media_results(
            query=clean_q,
            items=results,
            content_type="music",
            query_title=query_title,
            query_artist=query_artist,
        )

    async def play(self, result: MediaResult) -> ActionResult:
        """Execute playback of the SoundCloud track inside persistent MediaSession and verify."""
        logger.info("[SOUNDCLOUD] Playing in MediaSession: '%s' (%s)", result.title, result.url)

        status = await self._session.play(url=result.url, title=result.title, platform="soundcloud")

        if status.verification in {"verified", "playback_verified_audio_unknown"}:
            return ActionResult(
                success=True,
                action="play_content",
                platform="soundcloud",
                query=result.title,
                url=result.url,
                title=result.title,
                stage="playback",
                verification=status.verification,
                playback_status=status,
                extra={"verification": status.verification},
            )

        if status.verification == "started":
            return ActionResult(
                success=True,
                action="play_content",
                platform="soundcloud",
                query=result.title,
                url=result.url,
                title=result.title,
                stage="started",
                verification="not_available",
                playback_status=status,
                extra={"verification": "not_available"},
            )

        return ActionResult(
            success=False,
            action="play_content",
            platform="soundcloud",
            query=result.title,
            url=result.url,
            title=result.title,
            stage="failed",
            reason=status.error or "SoundCloud playback verification failed",
            verification="failed",
            playback_status=status,
        )
