"""YouTube provider for rich search and verified DOM playback."""

from __future__ import annotations

import logging
from urllib.parse import quote_plus
from typing import Any

from browser.manager import BrowserManager, get_browser_manager
from media.base import ActionResult, MediaProvider, MediaResult
from media.ranking import rank_media_results
from media.session import get_media_session

logger = logging.getLogger(__name__)


class YouTubeProvider(MediaProvider):
    """Searches YouTube with rich metadata extraction and plays via persistent MediaSession."""

    def __init__(self, browser_manager: BrowserManager | None = None) -> None:
        self._browser = browser_manager or get_browser_manager()
        self._session = get_media_session()

    async def search(
        self,
        query: str,
        content_type: str = "general",
        query_title: str = "",
        query_artist: str = "",
    ) -> list[MediaResult]:
        """Search YouTube and extract video titles, channels, duration, and descriptions."""
        clean_q = query.strip()
        if not clean_q:
            return []

        search_url = f"https://www.youtube.com/results?search_query={quote_plus(clean_q)}"
        logger.info("[YOUTUBE] Searching: '%s' (type=%s) -> %s", clean_q, content_type, search_url)

        page: Any = None
        results: list[MediaResult] = []
        try:
            page = await self._browser.get_page()
            await page.goto(search_url, wait_until="domcontentloaded", timeout=12000)

            # Wait briefly for video items to appear
            try:
                await page.wait_for_selector("a#video-title, ytd-video-renderer", timeout=5000)
            except Exception:
                pass

            # Extract rich video metadata
            raw_items = await page.evaluate(
                """() => {
                    const items = [];
                    const seen = new Set();
                    const renderers = document.querySelectorAll('ytd-video-renderer');

                    if (renderers.length > 0) {
                        for (const r of renderers) {
                            const titleEl = r.querySelector('a#video-title');
                            if (!titleEl) continue;
                            const href = titleEl.getAttribute('href');
                            const title = titleEl.getAttribute('title') || titleEl.innerText?.trim();
                            if (!href || !href.includes('/watch?v=') || !title || seen.has(href)) continue;
                            seen.add(href);

                            const channelEl = r.querySelector('#channel-info #channel-name, #channel-name a, ytd-channel-name');
                            const channel = channelEl ? channelEl.innerText.trim() : '';

                            const verified = Boolean(
                                r.querySelector('.badge-style-type-verified, [aria-label*="Подтверждено"], [aria-label*="Verified"]')
                            );

                            const durEl = r.querySelector('span#text.ytd-thumbnail-overlay-time-status-renderer, #time-status');
                            const duration = durEl ? durEl.innerText.trim() : '';

                            const descEl = r.querySelector('#description-text, .metadata-snippet-text');
                            const desc = descEl ? descEl.innerText.trim() : '';

                            items.push({
                                href: href.startsWith('http') ? href : 'https://www.youtube.com' + href,
                                title: title.trim(),
                                channel: channel,
                                verified: verified,
                                duration: duration,
                                description: desc,
                            });
                            if (items.length >= 10) break;
                        }
                    }

                    // Fallback to basic video title tags if ytd-video-renderer is not present
                    if (items.length === 0) {
                        const titleElements = document.querySelectorAll('a#video-title');
                        for (const el of titleElements) {
                            const href = el.getAttribute('href');
                            const title = el.getAttribute('title') || el.innerText?.trim();
                            if (href && href.includes('/watch?v=') && title && !seen.has(href)) {
                                seen.add(href);
                                items.push({
                                    href: href.startsWith('http') ? href : 'https://www.youtube.com' + href,
                                    title: title.trim(),
                                    channel: '',
                                    verified: false,
                                    duration: '',
                                    description: '',
                                });
                            }
                            if (items.length >= 10) break;
                        }
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
                        verified_channel=it.get("verified", False),
                        duration=it.get("duration", ""),
                        description=it.get("description", ""),
                        position=i,
                        content_type=content_type,
                    )
                )

        except Exception as exc:
            logger.error("[YOUTUBE] Search failed: %s", exc)
        finally:
            if page is not None:
                try:
                    await page.close()
                except Exception:
                    pass

        if not results:
            logger.warning("[YOUTUBE] No parsed results, creating search fallback result")
            return [
                MediaResult(
                    url=search_url,
                    title=f"Поиск YouTube: {clean_q}",
                    position=0,
                    content_type=content_type,
                )
            ]

        return rank_media_results(
            query=clean_q,
            items=results,
            content_type=content_type,
            query_title=query_title,
            query_artist=query_artist,
        )

    async def play(self, result: MediaResult) -> ActionResult:
        """Execute playback of the YouTube video inside persistent MediaSession and verify."""
        logger.info("[YOUTUBE] Playing in MediaSession: '%s' (%s)", result.title, result.url)

        status = await self._session.play(url=result.url, title=result.title, platform="youtube")

        if status.verification in {"verified", "playback_verified_audio_unknown"}:
            return ActionResult(
                success=True,
                action="play_content",
                platform="youtube",
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
                platform="youtube",
                query=result.title,
                url=result.url,
                title=result.title,
                stage="started",
                verification="not_available",
                playback_status=status,
                extra={"verification": "not_available"},
            )

        # Failed
        return ActionResult(
            success=False,
            action="play_content",
            platform="youtube",
            query=result.title,
            url=result.url,
            title=result.title,
            stage="failed",
            reason=status.error or "Playback verification failed (element missing or blocked)",
            verification="failed",
            playback_status=status,
        )
