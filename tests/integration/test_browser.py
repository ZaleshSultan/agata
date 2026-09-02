"""Integration coverage for the shared Playwright layer (no external website)."""

from __future__ import annotations

import asyncio
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

from browser import BrowserManager, Extractor, Navigation


class _PageHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
        body = b"""<html><head><title>Agata browser test</title></head>
        <body><main>This local page proves that managed browser extraction works.
        It contains enough readable text for the main-content selector.</main>
        <a href='/next'>Next page</a></body></html>"""
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        pass


def test_browser_opens_and_extracts_local_page() -> None:
    """Start Chromium, navigate, and extract text/links through shared objects."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), _PageHandler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()

    async def scenario() -> None:
        manager = BrowserManager()
        try:
            navigation = Navigation(manager)
            result = await navigation.open_url(f"http://127.0.0.1:{server.server_port}/")
            assert result.success, result.as_dict()
            assert result.page is not None
            extractor = Extractor()
            text = await extractor.extract_text(result.page)
            links = await extractor.extract_links(result.page)
            assert text.success, text.as_dict()
            assert "managed browser extraction" in text.text
            assert links.success, links.as_dict()
            assert links.links and links.links[0]["url"].endswith("/next")
        finally:
            await manager.close()

    try:
        asyncio.run(scenario())
    finally:
        server.shutdown()
        server.server_close()
