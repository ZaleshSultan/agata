"""Extract a local, legally obtained book without uploading its contents.

Only PDF needs a third-party reader (pypdf); EPUB/DOCX/TXT use the stdlib.
This module never modifies the source and never bypasses encryption/DRM.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
from html.parser import HTMLParser
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import unicodedata
import xml.etree.ElementTree as ET
import zipfile


def require_private_destination(path: Path) -> None:
    """Refuse book-derived writes to tracked/unignored paths inside this repo."""
    repository = Path(__file__).resolve().parents[1]
    try:
        relative = path.resolve().relative_to(repository).as_posix()
    except ValueError:
        return  # Explicit destinations outside this repository remain local files.
    ignored = subprocess.run(["git", "-C", str(repository), "check-ignore", "--quiet", "--", relative],
                             capture_output=True)
    tracked = subprocess.run(["git", "-C", str(repository), "ls-files", "--error-unmatch", "--", relative],
                             capture_output=True)
    if ignored.returncode != 0 or tracked.returncode == 0:
        raise ValueError("Book-derived output inside AGATA must be Git-ignored and untracked; use finetuning/books/")


def clean_text(text: str) -> str:
    """Keep paragraph boundaries; repair soft hyphens and lowercase line wraps."""
    text = unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00ad", "").replace("\u00a0", " ").replace("\ufeff", "")
    text = re.sub(r"([А-Яа-яЁёA-Za-z])-\n\s*([а-яёa-z])", r"\1\2", text)
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def _safe_zip(path: Path) -> zipfile.ZipFile:
    archive = zipfile.ZipFile(path)
    entries = archive.infolist()
    if len(entries) > 10000 or sum(e.file_size for e in entries) > 256 * 1024**2:
        archive.close()
        raise ValueError("Archive exceeds the 256 MiB / 10,000-entry extraction limit")
    if any(e.flag_bits & 1 for e in entries):
        archive.close()
        raise ValueError("Encrypted archives are not supported; obtain an unprotected authorized file")
    return archive


class _HTMLBlocks(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.blocks = []
        self.buffer = []
        self.kind = "p"
        self.hidden = 0

    def flush(self):
        value = clean_text("".join(self.buffer))
        if value:
            self.blocks.append((self.kind, value))
        self.buffer = []

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style"}:
            self.hidden += 1
        if tag in {"p", "li", "h1", "h2", "h3", "h4", "div"}:
            self.flush()
            self.kind = tag
        elif tag == "br":
            self.buffer.append("\n")

    def handle_endtag(self, tag):
        if tag in {"script", "style"}:
            self.hidden = max(0, self.hidden - 1)
        if tag in {"p", "li", "h1", "h2", "h3", "h4", "div"}:
            self.flush()

    def handle_data(self, data):
        if not self.hidden:
            self.buffer.append(data)


def _epub(path: Path):
    units, meta = [], {}
    with _safe_zip(path) as archive:
        if "META-INF/encryption.xml" in archive.namelist():
            raise ValueError("EPUB declares encrypted resources; DRM is not bypassed")
        container = ET.fromstring(archive.read("META-INF/container.xml"))
        rootfile = next(e.attrib["full-path"] for e in container.iter() if e.tag.endswith("rootfile"))
        package = ET.fromstring(archive.read(rootfile))
        for field in ("title", "creator", "date"):
            values = [e.text for e in package.iter() if e.tag.rsplit("}", 1)[-1] == field and e.text]
            if values:
                meta[field] = "; ".join(values)
        manifest = {e.attrib["id"]: e.attrib["href"] for e in package.iter() if e.tag.endswith("}item")}
        spine = [e.attrib["idref"] for e in package.iter() if e.tag.endswith("}itemref")]
        for spine_number, item in enumerate(spine, 1):
            href = str(PurePosixPath(rootfile).parent / manifest[item]).split("#")[0]
            parser = _HTMLBlocks()
            parser.feed(archive.read(href).decode("utf-8-sig"))
            parser.flush()
            for paragraph, (tag, value) in enumerate(parser.blocks, 1):
                units.append({"text": value, "heading": tag.startswith("h"),
                              "heading_level": int(tag[1:]) if re.fullmatch(r"h[1-6]", tag) else None,
                              "locator": f"{href}#paragraph-{paragraph}", "spine": spine_number})
    return units, meta, []


def _docx(path: Path):
    units, meta = [], {}
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    with _safe_zip(path) as archive:
        root = ET.fromstring(archive.read("word/document.xml"))
        for i, paragraph in enumerate(root.findall(".//w:p", ns), 1):
            value = "".join(node.text or "" for node in paragraph.findall(".//w:t", ns))
            style = paragraph.find("w:pPr/w:pStyle", ns)
            style_name = style.get(f"{{{ns['w']}}}val", "") if style is not None else ""
            if value.strip():
                level = re.search(r"(?:heading|заголовок)\s*([1-6])$", style_name, re.I)
                units.append({"text": value, "heading": bool(re.search(r"heading|заголовок", style_name, re.I)),
                              "heading_level": int(level.group(1)) if level else None,
                              "locator": f"paragraph-{i}"})
        if "docProps/core.xml" in archive.namelist():
            core = ET.fromstring(archive.read("docProps/core.xml"))
            meta = {e.tag.rsplit("}", 1)[-1]: e.text for e in core if e.text}
    return units, meta, []


def _pdf(path: Path):
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("PDF requires pypdf in the isolated fine-tuning environment") from exc
    reader = PdfReader(path)
    if reader.is_encrypted:
        raise ValueError("Encrypted PDF is not supported; obtain an authorized unencrypted file")
    pages = [(page.extract_text() or "") for page in reader.pages]
    outlines = []

    def visit_outline(items, depth=0):
        for item in items:
            if isinstance(item, list):
                visit_outline(item, depth + 1)
            else:
                try:
                    outlines.append({"page": reader.get_destination_page_number(item) + 1,
                                     "title": str(item.title), "depth": depth})
                except (AttributeError, KeyError, ValueError):
                    continue

    visit_outline(reader.outline)
    # Suppress exact repeated top/bottom lines only on sufficiently long books.
    margins = Counter()
    for page in pages:
        lines = [line.strip() for line in page.splitlines() if line.strip()]
        margins.update(set(lines[:1] + lines[-1:]))
    repeated = {line for line, count in margins.items() if len(pages) >= 6 and count >= max(4, len(pages) // 3)}
    units = []
    for page_number, raw in enumerate(pages, 1):
        lines = raw.splitlines()
        raw = "\n".join(line for i, line in enumerate(lines)
                        if not ((i == 0 or i == len(lines)-1) and line.strip() in repeated))
        unit = {"text": raw, "locator": f"pdf-page-{page_number}", "page": page_number}
        previous = [item for item in outlines if item["page"] <= page_number]
        chapters = [item for item in previous if item["depth"] == 0]
        if chapters:
            unit["chapter_title"] = chapters[-1]["title"]
            unit["chapter_key"] = f"outline-page-{chapters[-1]['page']}"
            unit["section_title"] = previous[-1]["title"]
        units.append(unit)
    low_text = [i + 1 for i, text in enumerate(pages) if len(text.strip()) < 40]
    warnings = []
    if low_text:
        warnings.append({"type": "low_text_pages", "pages": low_text,
                         "message": "May contain scans, illustrations or blank pages. OCR was not performed."})
    meta = {str(k).lstrip("/"): str(v) for k, v in (reader.metadata or {}).items()}
    meta["outline"] = outlines
    if any("конец ознакомительного" in page.lower() for page in pages[-3:]):
        warnings.append({"type": "sample_only", "message": "The file declares that it is a preview/sample, not the complete book."})
    return units, meta, warnings


def extract(path: Path, *, title=None, edition=None, year=None, encoding="utf-8-sig") -> dict:
    path = path.resolve()
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        units, metadata, warnings = _pdf(path)
    elif suffix == ".epub":
        units, metadata, warnings = _epub(path)
    elif suffix == ".docx":
        units, metadata, warnings = _docx(path)
    elif suffix == ".txt":
        text = path.read_text(encoding=encoding)
        units = [{"text": p, "locator": f"paragraph-{i}"}
                 for i, p in enumerate(re.split(r"\n\s*\n", text), 1) if p.strip()]
        metadata, warnings = {}, []
    else:
        raise ValueError("Supported formats: PDF, EPUB, TXT, DOCX")
    for i, unit in enumerate(units, 1):
        unit["text"] = clean_text(unit["text"])
        unit["unit_id"] = f"unit-{i:06d}"
    units = [u for u in units if u["text"]]
    if not units:
        raise ValueError("No extractable text; an OCR-produced authorized text source is needed")
    # Filename is an explicit fallback, never presented as verified bibliographic title.
    inferred_title = metadata.get("title") or metadata.get("Title")
    source = {"source_id": sha256(path.read_bytes()).hexdigest(), "filename": path.name,
              "format": suffix[1:], "title": title or inferred_title or path.stem,
              "title_verified": title is not None, "edition": edition, "year": year,
              "raw_metadata": metadata, "sample_only": any(w.get("type") == "sample_only" for w in warnings)}
    return {"schema_version": 1, "source": source, "warnings": warnings, "units": units}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title")
    parser.add_argument("--edition")
    parser.add_argument("--year", type=int)
    parser.add_argument("--encoding", default="utf-8-sig", help="TXT encoding; use cp1251 only when known")
    args = parser.parse_args()
    if args.output.resolve() == args.input.resolve():
        parser.error("Output must not overwrite the original source")
    if args.output.exists():
        parser.error("Output already exists; choose a fresh output file")
    require_private_destination(args.output)
    result = extract(args.input, title=args.title, edition=args.edition, year=args.year, encoding=args.encoding)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "units": len(result["units"]),
                      "warnings": result["warnings"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
