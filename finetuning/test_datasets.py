"""Meaningful leakage and tool-protocol regression tests; no GPU or AGATA imports."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from prepare_dataset import (assert_no_leakage, build_seed, prepare, read_jsonl,
                             runtime_tool_schemas, split_records, validate_benchmark, validate_record,
                             ensure_private_output_paths, book_records, REPO)


class DatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.datasets, cls.benchmark = build_seed(runtime_tool_schemas())
        cls.rows = [row for members in cls.datasets.values() for row in members]

    def test_seed_protocol_and_leakage(self):
        splits = split_records(self.rows)
        assert_no_leakage(splits)
        validate_benchmark(self.benchmark, splits)
        self.assertEqual({item["category"] for item in self.benchmark}, set("ABCDEF"))
        self.assertTrue(all(not row["metadata"].get("excluded_entire_chapter") for row in splits["train"]))

    def test_missing_tool_result_rejected(self):
        row = copy.deepcopy(self.datasets["tool_calling"][0])
        row["messages"] = [message for message in row["messages"] if message["role"] != "tool"]
        with self.assertRaisesRegex(ValueError, "missing tool result"):
            validate_record(row)

    def test_chapter_leakage_rejected(self):
        first = copy.deepcopy(self.rows[0])
        second = copy.deepcopy(self.rows[1])
        second["group_id"] = first["group_id"]
        with self.assertRaisesRegex(ValueError, "Group leakage"):
            assert_no_leakage({"train": [first], "test": [second]})

    def test_conflicting_chapter_partitions_rejected(self):
        first, second = copy.deepcopy(self.rows[:2])
        first["group_id"] = second["group_id"] = "book:source:chapter"
        first["metadata"]["partition"] = "train"
        second["metadata"]["partition"] = "test"
        with self.assertRaisesRegex(ValueError, "Conflicting partitions"):
            split_records([first, second])

    def test_benchmark_paraphrase_leakage_rejected(self):
        benchmark = copy.deepcopy(self.benchmark)
        benchmark[0]["category"] = "C"
        with self.assertRaisesRegex(ValueError, "Unintended benchmark leakage"):
            validate_benchmark(benchmark, split_records(self.rows))

    def test_reproducible_safe_output_and_smoke_subset(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            prepare(root / "a", root / "a_benchmark.jsonl")
            prepare(root / "b", root / "b_benchmark.jsonl")
            for partition in ("train", "validation", "test", "smoke"):
                self.assertEqual((root / "a" / f"{partition}.jsonl").read_bytes(),
                                 (root / "b" / f"{partition}.jsonl").read_bytes())
            train = {row["id"] for row in read_jsonl(root / "a" / "train.jsonl")}
            smoke = {row["id"] for row in read_jsonl(root / "a" / "smoke.jsonl")}
            self.assertTrue(smoke <= train)
            with self.assertRaises(FileExistsError):
                prepare(root / "a", root / "a_benchmark.jsonl")

    def test_smoke_book_cap_and_heldout_chapter(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            rows = []
            for index in range(14):
                partition = "train" if index < 12 else "test"
                rows.append({"id": f"book_fixture_{index}", "messages": [
                    {"role": "user", "content": f"Что сказано в учебной карточке номер {index}?"},
                    {"role": "assistant", "content": f"В книге «Учебная выдумка» (2026) сказано: Тестовая цитата номер {index}."}],
                    "metadata": {"review_status": "approved", "evidence": f"Тестовая цитата номер {index}.",
                                 "source_id": "fictional_unit_test", "chapter_id": partition,
                                 "title": "Учебная выдумка", "title_verified": True, "year": 2026,
                                 "chapter": partition, "locator": f"fixture-{index}", "reviewer": "unit test",
                                 "review_note": "Fictional fixture only", "training_mode": "source_attributed",
                                 "partition": partition}})
            source = root / "book.jsonl"
            source.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8")
            prepare(root / "data", root / "benchmark.jsonl", book=source)
            smoke = read_jsonl(root / "data" / "smoke.jsonl")
            book = [row for row in smoke if row["category"] == "book_knowledge"]
            self.assertEqual(len(book), 8)
            self.assertTrue(all(row["metadata"]["partition"] == "train" for row in book))

    def test_excluded_benchmark_chapter_leakage_rejected(self):
        benchmark = copy.deepcopy(self.benchmark)
        excluded = next(row for row in benchmark if row["category"] == "D")
        excluded["metadata"]["chapter"] = "Огни"
        with self.assertRaisesRegex(ValueError, "Excluded benchmark chapter"):
            validate_benchmark(benchmark, split_records(self.rows))

    def test_paraphrase_must_point_to_seen_fact(self):
        benchmark = copy.deepcopy(self.benchmark)
        paraphrase = next(row for row in benchmark if row["category"] == "C")
        paraphrase["metadata"]["training_fact_id"] = "missing-fact"
        with self.assertRaisesRegex(ValueError, "absent training fact"):
            validate_benchmark(benchmark, split_records(self.rows))

    def test_private_outputs_cannot_use_code_directory(self):
        with self.assertRaisesRegex(ValueError, "Git-ignored and untracked"):
            ensure_private_output_paths([REPO / "finetuning" / "unignored_book_excerpt.jsonl"])
        ensure_private_output_paths([REPO / "finetuning" / "datasets" / "private_fixture.jsonl"])

    def test_book_quote_subset_keeps_sentence_context(self):
        row = {"id": "book_quote_fixture", "messages": [
            {"role": "user", "content": "Что сказано о бумагах в учебной выдумке?"},
            {"role": "assistant", "content": "В книге «Учебная выдумка» (2026) сказано: Храните бумаги в папке."}],
            "metadata": {"review_status": "approved", "evidence": "Разберите стол. Храните бумаги в папке.",
                         "answer_quote": "Храните бумаги в папке.",
                         "source_id": "fictional_unit_test", "chapter_id": "paper",
                         "title": "Учебная выдумка", "title_verified": True, "year": 2026,
                         "chapter": "Бумаги", "locator": "fixture-quote", "reviewer": "unit test",
                         "review_note": "Fictional fixture only", "training_mode": "source_attributed",
                         "partition": "train"}}
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "book.jsonl"
            source.write_text(json.dumps(row, ensure_ascii=False), encoding="utf-8")
            self.assertEqual(len(book_records(source)), 1)
            row["metadata"]["evidence"] = "Не храните бумаги в папке."
            row["metadata"]["answer_quote"] = "храните бумаги в папке."
            row["messages"][-1]["content"] = "В книге «Учебная выдумка» (2026) сказано: храните бумаги в папке."
            source.write_text(json.dumps(row, ensure_ascii=False), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "exact reviewed evidence"):
                book_records(source)


if __name__ == "__main__":
    unittest.main()
