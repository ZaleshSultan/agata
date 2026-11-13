import json
from pathlib import Path
import sys
import zipfile

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from extract_book import clean_text, extract, require_private_destination
from book_pipeline import (candidate, chunks_from_document, export_reviewed,
                           extractive_candidates, prepare, validation_reasons,
                           ollama_candidates)


def document():
    return {"source": {"source_id": "synthetic-001", "title": "Вымышленный сад",
                       "title_verified": True, "edition": "Тестовая", "year": 2020},
            "units": [{"unit_id": "1", "locator": "paragraph-1",
                       "text": "Глава 1\nВ вымышленном саду растут только синие яблоки."},
                      {"unit_id": "2", "locator": "paragraph-2",
                       "text": "Глава 2\nВ вымышленном саду фонтан работает только по вторникам."}]}


def approved_card(chunk, partition="train"):
    card = candidate(chunk, "Какого цвета яблоки в этом вымышленном саду?", chunk["text"], chunk["text"])
    card.update(review_status="approved", reviewer="synthetic-test", review_note="Source checked", partition=partition)
    return card


def test_clean_text_repairs_wrapped_word_not_inline_hyphen():
    assert clean_text("за-\nдача\r\n\r\nпо-прежнему\u00a0важна\u00ad") == "задача\n\nпо-прежнему важна"


def test_book_destinations_cannot_be_tracked_or_unignored():
    repository = Path(__file__).resolve().parents[2]
    if not (repository / ".git").exists():
        pytest.skip("Privacy guard integration requires a Git checkout")
    with pytest.raises(ValueError, match="Git-ignored and untracked"):
        require_private_destination(repository / "prompts.py")
    with pytest.raises(ValueError, match="Git-ignored and untracked"):
        require_private_destination(repository / "finetuning" / "accidental_public_book.jsonl")
    require_private_destination(repository / "finetuning" / "books" / "private_book.jsonl")


def test_chapters_are_distinct_and_chunk_text_preserved():
    chunks = chunks_from_document(document())
    assert len(chunks) == 2
    assert chunks[0]["chapter_id"] != chunks[1]["chapter_id"]
    assert chunks[0]["text"] == "В вымышленном саду растут только синие яблоки."


def test_pdf_outline_group_is_not_split_by_inline_heading():
    doc = document()
    for unit in doc["units"]:
        unit.update(chapter_key="outline-page-1", chapter_title="Общая глава", section_title="Раздел")
    assert len({chunk["chapter_id"] for chunk in chunks_from_document(doc)}) == 1


def test_subsections_are_not_independent_chapter_split_groups():
    doc = document()
    doc["units"] = [
        {"unit_id": "1", "locator": "paragraph-1", "text": "Начальная глава", "heading": True, "heading_level": 1},
        {"unit_id": "2", "locator": "paragraph-2", "text": "Первое содержание."},
        {"unit_id": "3", "locator": "paragraph-3", "text": "Второй раздел", "heading": True, "heading_level": 2},
        {"unit_id": "4", "locator": "paragraph-4", "text": "Второе содержание."},
    ]
    chunks = chunks_from_document(doc)
    assert len(chunks) == 2
    assert chunks[0]["chapter_id"] == chunks[1]["chapter_id"]
    assert chunks[1]["section"] == "Второй раздел"


def test_fabricated_answer_and_evidence_rejected():
    chunk = chunks_from_document(document())[0]
    card = approved_card(chunk)
    card["answer"] = "Яблоки на самом деле красные."
    assert "answer_not_in_evidence" in validation_reasons(card, chunk)
    card["evidence"] = card["answer"]
    assert "evidence_not_in_source" in validation_reasons(card, chunk)


def test_substring_cannot_drop_source_negation():
    chunk = chunks_from_document(document())[0]
    chunk["text"] = "Не применяйте вымышленный препарат для лечения."
    card = candidate(chunk, "Что написано о препарате в источнике?",
                     "применяйте вымышленный препарат для лечения.",
                     "применяйте вымышленный препарат для лечения.")
    assert "evidence_not_complete_source_sentence" in validation_reasons(card, chunk)


def test_review_and_source_provenance_required():
    chunk = chunks_from_document(document())[0]
    card = approved_card(chunk)
    card["locator"] = "made-up-page-999"
    good, rejected = export_reviewed([card], [chunk])
    assert not good and "provenance_mismatch:locator" in rejected[0]["rejection_reasons"]
    card = approved_card(chunk)
    card["review_status"] = "pending"
    assert not export_reviewed([card], [chunk])[0]


def test_chapter_leakage_is_rejected_for_both_rows():
    chunk = chunks_from_document(document())[0]
    first, second = approved_card(chunk, "train"), approved_card(chunk, "test")
    good, rejected = export_reviewed([first, second], [chunk])
    assert not good and len(rejected) == 2
    assert all("chapter_leakage" in row["rejection_reasons"] for row in rejected)


def test_historical_medical_advice_never_exports_as_current():
    chunk = chunks_from_document(document())[0]
    chunk["text"] = "Вымышленное лекарство описано в архивной книге за 1984 год."
    card = approved_card(chunk)
    assert not export_reviewed([card], [chunk])[0]
    good, rejected = export_reviewed([card], [chunk], allow_historical=True)
    assert len(good) == 1 and not rejected
    assert "историческое утверждение" in good[0]["messages"][-1]["content"]


def test_explicit_historical_mode_is_not_bypassed_without_keyword_risk():
    chunk = chunks_from_document(document())[0]
    card = approved_card(chunk)
    card["training_mode"] = "historical"
    assert card["risk_categories"] == []
    assert not export_reviewed([card], [chunk])[0]
    good, rejected = export_reviewed([card], [chunk], allow_historical=True)
    assert len(good) == 1 and "историческое утверждение" in good[0]["messages"][-1]["content"]
    card["source"]["year"] = "unknown"
    assert not export_reviewed([card], [chunk], allow_historical=True)[0]


def test_cloud_alias_rejected_before_any_book_content_is_sent(monkeypatch):
    import book_pipeline
    requests = []

    def local_json(host, endpoint, body):
        requests.append((endpoint, body))
        return {"remote_model": "some-cloud-model", "remote_host": "https://example.com"}

    monkeypatch.setattr(book_pipeline, "_local_json", local_json)
    chunk = chunks_from_document(document())[0]
    with pytest.raises(ValueError, match="before sending"):
        ollama_candidates(chunk, "innocent-local-alias")
    assert requests == [("/api/show", {"model": "innocent-local-alias"})]
    with pytest.raises(ValueError, match="Cloud models"):
        ollama_candidates(chunk, "some-model:cloud")
    assert len(requests) == 1


def test_local_model_preflight_requires_positive_weight_metadata(monkeypatch):
    import book_pipeline
    monkeypatch.setattr(book_pipeline, "_local_json", lambda *args: {})
    with pytest.raises(ValueError, match="Cannot verify local"):
        ollama_candidates(chunks_from_document(document())[0], "unknown-model")


def test_conflicting_exact_supported_answers_are_quarantined():
    chunks = chunks_from_document(document())
    first, second = approved_card(chunks[0]), approved_card(chunks[1])
    second["question"] = first["question"]
    good, rejected = export_reviewed([first, second], chunks)
    assert not good
    assert all("conflicting_answers" in row["rejection_reasons"] for row in rejected)


def test_txt_pipeline_never_auto_approves(tmp_path):
    source = tmp_path / "source.txt"
    source.write_text("Глава 1\nВ вымышленном саду растут только синие яблоки.", encoding="utf-8")
    doc = extract(source, title="Вымышленный сад", year=2020)
    summary = prepare(doc, tmp_path / "prepared")
    assert summary["status"] == "review_required"
    cards = [json.loads(line) for line in (tmp_path / "prepared/candidates.jsonl").read_text(encoding="utf-8").splitlines()]
    assert cards and all(card["review_status"] == "pending" for card in cards)
    with pytest.raises(ValueError, match="nonempty"):
        prepare(doc, tmp_path / "prepared")


def test_docx_heading_and_locator(tmp_path):
    path = tmp_path / "book.docx"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("word/document.xml", '''<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>
        <w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr><w:r><w:t>Глава 1</w:t></w:r></w:p>
        <w:p><w:r><w:t>Вымышленный дом стоит у синего озера.</w:t></w:r></w:p></w:body></w:document>''')
    doc = extract(path)
    assert doc["units"][0]["heading"]
    assert doc["units"][1]["locator"] == "paragraph-2"


def test_epub_reads_spine_order_not_zip_order(tmp_path):
    path = tmp_path / "book.epub"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("META-INF/container.xml", '<container><rootfiles><rootfile full-path="OEBPS/content.opf"/></rootfiles></container>')
        archive.writestr("OEBPS/content.opf", '''<package xmlns="http://www.idpf.org/2007/opf"><metadata/><manifest>
        <item id="a" href="a.xhtml"/><item id="b" href="b.xhtml"/></manifest><spine><itemref idref="b"/><itemref idref="a"/></spine></package>''')
        archive.writestr("OEBPS/a.xhtml", "<p>Вторая запись</p>")
        archive.writestr("OEBPS/b.xhtml", "<h1>Первая глава</h1><p>Первая запись</p>")
    doc = extract(path)
    assert doc["units"][0]["text"] == "Первая глава"
    assert doc["units"][-1]["text"] == "Вторая запись"
    with zipfile.ZipFile(path, "a") as archive:
        archive.writestr("META-INF/encryption.xml", "<encryption/>")
    with pytest.raises(ValueError, match="DRM"):
        extract(path)


def test_pdf_optional_dependency_and_metadata(tmp_path):
    pypdf = pytest.importorskip("pypdf")
    # Real local book integration was executed separately; blank PDF must fail clearly.
    path = tmp_path / "blank.pdf"
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=200, height=200)
    with path.open("wb") as stream:
        writer.write(stream)
    with pytest.raises(ValueError, match="No extractable text"):
        extract(path)
