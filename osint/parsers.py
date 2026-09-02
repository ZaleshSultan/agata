"""Small deterministic parsers used by public collectors before any AI step."""
from __future__ import annotations
import re
from html.parser import HTMLParser
from urllib.parse import urljoin

EMAIL_RE = re.compile(r"(?<![\w.+-])[\w.+-]+@[\w-]+(?:\.[\w-]+)+(?![\w.-])")

class ProfileParser(HTMLParser):
    def __init__(self, base_url: str) -> None:
        super().__init__(); self.base_url, self.title, self._title, self.meta, self.links = base_url, [], False, {}, []
    def handle_starttag(self, tag, attrs):
        data = dict(attrs)
        if tag == "title": self._title = True
        if tag == "meta" and data.get("name") in {"description", "twitter:description"}: self.meta[data["name"]] = data.get("content", "")
        if tag == "meta" and data.get("property") in {"og:title", "og:description", "og:image"}: self.meta[data["property"]] = data.get("content", "")
        if tag == "a" and data.get("href"): self.links.append(urljoin(self.base_url, data["href"]))
    def handle_endtag(self, tag):
        if tag == "title": self._title = False
    def handle_data(self, data):
        if self._title: self.title.append(data.strip())

def parse_public_page(html: str, url: str) -> dict[str, object]:
    parser = ProfileParser(url); parser.feed(html)
    return {"title": " ".join(part for part in parser.title if part), "bio": parser.meta.get("og:description") or parser.meta.get("description", ""), "avatar_url": parser.meta.get("og:image", ""), "external_links": list(dict.fromkeys(parser.links))[:25], "emails": sorted(set(EMAIL_RE.findall(html)))}
