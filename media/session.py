"""Persistent Media Session managing playback state and browser media context."""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
from dataclasses import dataclass, field
from typing import Any

from browser.manager import BrowserManager, get_browser_manager
from media.base import PlaybackStatus

logger = logging.getLogger(__name__)


@dataclass
class MediaSession:
    """Manages active media playback, tab reuse, and player state verification."""

    current_platform: str = ""
    current_url: str = ""
    current_title: str = ""
    current_query: str = ""
    page: Any | None = None
    status: PlaybackStatus = field(default_factory=PlaybackStatus)
    browser_manager: BrowserManager | None = None

    @property
    def playing(self) -> bool:
        return self.status.playing

    @playing.setter
    def playing(self, value: bool) -> None:
        self.status.playing = bool(value)

    def _get_browser(self) -> BrowserManager:
        if self.browser_manager is None:
            self.browser_manager = get_browser_manager()
        return self.browser_manager

    async def get_or_create_page(self, platform: str = "") -> Any:
        """Return the persistent media page, reusing existing instance."""
        if self.page is not None and not self.page.is_closed():
            return self.page

        bm = self._get_browser()
        page = await bm.get_media_page()
        self.page = page

        def _on_close() -> None:
            logger.info("[MEDIA SESSION] Media page was closed")
            self.page = None
            self.status.playing = False
            self.status.verification = "failed"

        self.page.on("close", _on_close)
        if platform:
            self.current_platform = platform
        return self.page

    async def play(self, url: str, title: str = "", platform: str = "youtube") -> PlaybackStatus:
        """Navigate to media URL and initiate and verify real DOM playback."""
        clean_platform = platform.lower().strip() or "youtube"
        self.current_platform = clean_platform
        self.current_url = url
        self.current_title = title or url
        self.status = PlaybackStatus(verification="pending")

        logger.info("[MEDIA SESSION] Playing '%s' on %s (%s)", self.current_title, self.current_platform, url)

        page = await self.get_or_create_page(clean_platform)

        # 1. Navigation
        try:
            current_page_url = page.url or ""
            if current_page_url != url:
                await page.goto(url, wait_until="domcontentloaded", timeout=15000)
            self.status.navigation = True
        except Exception as exc:
            logger.error("[MEDIA SESSION] Navigation failed: %s", exc)
            self.status.verification = "failed"
            self.status.error = f"Navigation error: {exc}"
            return self.status

        # 2. Wait for player & execute playback based on platform
        if clean_platform == "youtube":
            await self._execute_youtube_playback(page)
        elif clean_platform == "soundcloud":
            await self._execute_soundcloud_playback(page)
        else:
            await self._execute_generic_playback(page)

        return self.status

    async def _execute_youtube_playback(self, page: Any) -> None:
        """Handle YouTube consent banners, wait for video element, play, and verify."""
        # Handle cookie / consent popups if present
        try:
            await page.evaluate(
                """() => {
                    const buttons = Array.from(document.querySelectorAll('button, .yt-spec-button-shape-next'));
                    const acceptBtn = buttons.find(b => {
                        const txt = (b.innerText || b.getAttribute('aria-label') || '').toLowerCase();
                        return txt.includes('принять') || txt.includes('agree') || txt.includes('accept') || txt.includes('i agree');
                    });
                    if (acceptBtn) acceptBtn.click();
                }"""
            )
        except Exception:
            pass

        # Wait for <video> element
        try:
            await page.wait_for_selector("video", timeout=10000)
            self.status.element_found = True
        except Exception as exc:
            logger.warning("[MEDIA SESSION] YouTube video element not found: %s", exc)
            self.status.element_found = False
            self.status.verification = "failed"
            self.status.error = "Video player element not found"
            return

        # Execute playback via DOM API and click fallback
        self.status.play_requested = True
        try:
            await page.evaluate(
                """() => {
                    const video = document.querySelector('video');
                    if (video) {
                        video.muted = false;
                        const promise = video.play();
                        if (promise !== undefined) {
                            promise.catch(e => console.log('Autoplay blocked:', e));
                        }
                    }
                    const playBtn = document.querySelector('.ytp-play-button, .ytp-large-play-button');
                    if (playBtn && video && video.paused) {
                        playBtn.click();
                    }
                }"""
            )
        except Exception as exc:
            logger.warning("[MEDIA SESSION] Play execution exception: %s", exc)

        # Allow player a moment to start buffering and streaming
        await asyncio.sleep(1.2)

        # Sample playback properties from DOM
        dom_state = await page.evaluate(
            """() => {
                const video = document.querySelector('video');
                if (!video) return null;
                return {
                    paused: video.paused,
                    currentTime: video.currentTime,
                    duration: video.duration || 0,
                    muted: video.muted,
                    readyState: video.readyState,
                    ended: video.ended,
                };
            }"""
        )

        if not dom_state:
            self.status.verification = "failed"
            self.status.error = "Failed to query video state"
            return

        is_paused = dom_state.get("paused", True)
        current_time = float(dom_state.get("currentTime", 0.0))
        duration = float(dom_state.get("duration", 0.0))
        muted = bool(dom_state.get("muted", False))

        self.status.playing = not is_paused
        self.status.current_time = current_time
        self.status.duration = duration
        self.status.muted = muted

        if not is_paused or current_time > 0:
            if muted:
                self.status.verification = "playback_verified_audio_unknown"
            else:
                self.status.verification = "verified"
        else:
            self.status.verification = "started"

    async def _execute_soundcloud_playback(self, page: Any) -> None:
        """Handle SoundCloud play button, wait for audio, and verify."""
        try:
            # Handle cookie consent banner if present
            await page.evaluate(
                """() => {
                    const acceptBtn = document.querySelector('#onetrust-accept-btn-handler');
                    if (acceptBtn) acceptBtn.click();
                }"""
            )
        except Exception:
            pass

        try:
            await page.wait_for_selector(".playControl, .playButton, button.sc-button-play, audio", timeout=8000)
            self.status.element_found = True
        except Exception:
            self.status.element_found = False

        self.status.play_requested = True
        try:
            await page.evaluate(
                """() => {
                    const playBtn = document.querySelector('.playControl:not(.playing), .playButton, button.sc-button-play');
                    if (playBtn) playBtn.click();
                    const audio = document.querySelector('audio');
                    if (audio) {
                        audio.muted = false;
                        audio.play().catch(e => console.log('Audio play error', e));
                    }
                }"""
            )
        except Exception as exc:
            logger.warning("[MEDIA SESSION] SoundCloud play error: %s", exc)

        await asyncio.sleep(1.0)

        dom_state = await page.evaluate(
            """() => {
                const playBtn = document.querySelector('.playControl, .playButton');
                const isPlayingClass = playBtn ? (playBtn.classList.contains('playing') || playBtn.getAttribute('title') === 'Pause') : false;
                const audio = document.querySelector('audio');
                const audioPaused = audio ? audio.paused : null;
                const currentTime = audio ? audio.currentTime : 0;
                const duration = audio ? audio.duration || 0 : 0;
                const muted = audio ? audio.muted : false;
                return {
                    isPlayingClass,
                    audioPaused,
                    currentTime,
                    duration,
                    muted,
                };
            }"""
        )

        if dom_state:
            is_playing = dom_state.get("isPlayingClass") or (dom_state.get("audioPaused") is False)
            self.status.playing = bool(is_playing)
            self.status.current_time = float(dom_state.get("currentTime", 0.0))
            self.status.duration = float(dom_state.get("duration", 0.0))
            self.status.muted = bool(dom_state.get("muted", False))
            self.status.verification = "verified" if is_playing else "started"
        else:
            self.status.verification = "started"

    async def _execute_generic_playback(self, page: Any) -> None:
        """Generic HTML5 video/audio playback."""
        try:
            await page.evaluate("document.querySelector('video, audio')?.play()")
            self.status.play_requested = True
            self.status.verification = "started"
        except Exception:
            self.status.verification = "failed"

    async def refresh_state(self) -> PlaybackStatus:
        """Query live DOM state of the player and update status."""
        if self.page is None or self.page.is_closed():
            self.status.playing = False
            self.status.verification = "failed"
            return self.status

        try:
            dom_state = await self.page.evaluate(
                """() => {
                    const video = document.querySelector('video');
                    if (video) {
                        return {
                            found: true,
                            paused: video.paused,
                            currentTime: video.currentTime,
                            duration: video.duration || 0,
                            muted: video.muted,
                        };
                    }
                    const audio = document.querySelector('audio');
                    if (audio) {
                        return {
                            found: true,
                            paused: audio.paused,
                            currentTime: audio.currentTime,
                            duration: audio.duration || 0,
                            muted: audio.muted,
                        };
                    }
                    const playBtn = document.querySelector('.playControl, .playButton');
                    if (playBtn) {
                        const playing = playBtn.classList.contains('playing') || playBtn.getAttribute('title') === 'Pause';
                        return {
                            found: true,
                            paused: !playing,
                            currentTime: 0,
                            duration: 0,
                            muted: false,
                        };
                    }
                    return { found: false };
                }"""
            )
            if dom_state and dom_state.get("found"):
                is_paused = dom_state.get("paused", True)
                self.status.playing = not is_paused
                self.status.current_time = float(dom_state.get("currentTime", 0.0))
                self.status.duration = float(dom_state.get("duration", 0.0))
                self.status.muted = bool(dom_state.get("muted", False))
            else:
                self.status.playing = False
        except Exception as exc:
            logger.debug("[MEDIA SESSION] refresh_state error: %s", exc)

        return self.status

    async def pause(self) -> bool:
        """Pause current media playback in DOM player."""
        if self.page is None or self.page.is_closed():
            return False

        try:
            await self.page.evaluate(
                """() => {
                    const media = document.querySelector('video, audio');
                    if (media && !media.paused) {
                        media.pause();
                    }
                    const scBtn = document.querySelector('.playControl.playing, button.playControl[title*="Pause"]');
                    if (scBtn) scBtn.click();
                }"""
            )
            await self.refresh_state()
            self.status.playing = False
            logger.info("[MEDIA SESSION] Paused playback")
            return True
        except Exception as exc:
            logger.warning("[MEDIA SESSION] Pause failed: %s", exc)
            return False

    async def resume(self) -> bool:
        """Resume playback in DOM player."""
        if self.page is None or self.page.is_closed():
            return False

        try:
            await self.page.evaluate(
                """() => {
                    const media = document.querySelector('video, audio');
                    if (media && media.paused) {
                        media.play();
                    }
                    const scBtn = document.querySelector('.playControl:not(.playing), button.playControl[title*="Play"]');
                    if (scBtn) scBtn.click();
                }"""
            )
            await self.refresh_state()
            self.status.playing = True
            logger.info("[MEDIA SESSION] Resumed playback")
            return True
        except Exception as exc:
            logger.warning("[MEDIA SESSION] Resume failed: %s", exc)
            return False

    async def stop(self) -> bool:
        """Stop playback and rewind position to 0."""
        if self.page is None or self.page.is_closed():
            return False

        try:
            await self.page.evaluate(
                """() => {
                    const media = document.querySelector('video, audio');
                    if (media) {
                        media.pause();
                        media.currentTime = 0;
                    }
                    const scBtn = document.querySelector('.playControl.playing');
                    if (scBtn) scBtn.click();
                }"""
            )
            self.status.playing = False
            self.status.current_time = 0.0
            logger.info("[MEDIA SESSION] Stopped playback and rewound to 0")
            return True
        except Exception as exc:
            logger.warning("[MEDIA SESSION] Stop failed: %s", exc)
            return False

    async def next(self) -> bool:
        """Trigger next track in player."""
        if self.page is None or self.page.is_closed():
            return False

        try:
            await self.page.evaluate(
                """() => {
                    const ytNext = document.querySelector('.ytp-next-button');
                    if (ytNext) { ytNext.click(); return; }
                    const scNext = document.querySelector('.skipControl__next');
                    if (scNext) { scNext.click(); return; }
                }"""
            )
            await asyncio.sleep(1.0)
            await self.refresh_state()
            logger.info("[MEDIA SESSION] Next track triggered")
            return True
        except Exception as exc:
            logger.warning("[MEDIA SESSION] Next track failed: %s", exc)
            return False

    async def previous(self) -> bool:
        """Rewind current track or trigger previous track."""
        if self.page is None or self.page.is_closed():
            return False

        try:
            await self.page.evaluate(
                """() => {
                    const media = document.querySelector('video, audio');
                    if (media && media.currentTime > 3) {
                        media.currentTime = 0;
                        return;
                    }
                    const ytPrev = document.querySelector('.ytp-prev-button');
                    if (ytPrev) { ytPrev.click(); return; }
                    const scPrev = document.querySelector('.skipControl__previous');
                    if (scPrev) { scPrev.click(); return; }
                    if (media) media.currentTime = 0;
                }"""
            )
            await asyncio.sleep(1.0)
            await self.refresh_state()
            logger.info("[MEDIA SESSION] Previous track triggered")
            return True
        except Exception as exc:
            logger.warning("[MEDIA SESSION] Previous track failed: %s", exc)
            return False

    def dispatch_command_sync(self, action: str) -> bool:
        """Synchronously execute transport command on browser media player if alive."""
        if self.page is None or self.page.is_closed():
            return False

        action_clean = action.lower().strip()

        async def _run():
            if action_clean in {"stop"}:
                return await self.stop()
            if action_clean in {"pause"}:
                return await self.pause()
            if action_clean in {"resume", "play"}:
                return await self.resume()
            if action_clean in {"play_pause"}:
                if self.playing:
                    return await self.pause()
                else:
                    return await self.resume()
            if action_clean in {"next"}:
                return await self.next()
            if action_clean in {"previous", "prev"}:
                return await self.previous()
            return False

        try:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop and loop.is_running():
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    return pool.submit(asyncio.run, _run()).result()
            else:
                return asyncio.run(_run())
        except Exception as exc:
            logger.debug("[MEDIA SESSION] dispatch_command_sync error: %s", exc)
            return False


_global_media_session: MediaSession | None = None


def get_media_session() -> MediaSession:
    """Return the global MediaSession singleton."""
    global _global_media_session
    if _global_media_session is None:
        _global_media_session = MediaSession()
    return _global_media_session
