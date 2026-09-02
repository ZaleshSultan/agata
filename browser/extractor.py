"""Page-content extraction with bounded, structured results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import config


@dataclass(slots=True)
class ExtractionResult:
    success: bool
    url: str
    text: str = ""
    links: list[dict[str, str]] | None = None
    error: str | None = None

    def as_dict(self) -> dict[str, object]:
        result: dict[str, object] = {"success": self.success, "url": self.url}
        if self.text:
            result["text"] = self.text
        if self.links is not None:
            result["links"] = self.links
        if self.error:
            result["error"] = self.error
        return result


class Extractor:
    async def extract_text(self, page: Any) -> ExtractionResult:
        try:
            text = await page.evaluate(
                """() => {
                    for (const selector of ['main', 'article', '[role=main]', '.content', '#content', 'body']) {
                        const element = document.querySelector(selector);
                        if (element && element.innerText.trim().length > 100) return element.innerText.trim();
                    }
                    return document.body?.innerText?.trim() || '';
                }"""
            )
            return ExtractionResult(True, page.url, text=(text or "")[:config.BROWSER_MAX_EXTRACT_CHARS])
        except Exception as exc:
            return ExtractionResult(False, getattr(page, "url", ""), error=f"Text extraction failed: {exc}")

    async def extract_links(self, page: Any) -> ExtractionResult:
        try:
            links = await page.eval_on_selector_all(
                "a[href]",
                """anchors => anchors.map(a => ({text: (a.innerText || '').trim(), url: a.href}))
                    .filter(link => link.url)""",
            )
            return ExtractionResult(True, page.url, links=links[:config.BROWSER_MAX_LINKS])
        except Exception as exc:
            return ExtractionResult(False, getattr(page, "url", ""), error=f"Link extraction failed: {exc}")
