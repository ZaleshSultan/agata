"""Audit guards for frozen protocols and semantic comparisons; reviewed 2026-11-10 (no GPU)."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from report_abc import aggregate, load_reports, paired_interval, PRIMARY_GROUPS, MODELS
from run_abc import freeze_manifest


def review_fixture():
    items, mapping, ratings = [], {}, []
    # Unequal domain sizes must not turn the primary metric into micro accuracy.
    for group in PRIMARY_GROUPS:
        for index in range(4 if group == "persona" else 1):
            item_id = f"{group}-{index}"
            item = {"suite": "benchmark", "id": item_id, "metadata": {"primary_group": group}, "responses": []}
            items.append(item)
            for model in MODELS:
                response_id = f"{item_id}-{model}"
                item["responses"].append({"response_id": response_id, "response": "test answer"})
                mapping[response_id] = {"suite": "benchmark", "item_id": item_id, "model": model}
                ratings.append({"response_id": response_id, "score": int(model == "A" and group == "persona"),
                                "reason": "Reference condition evaluated", "rubric_evidence": "Full required fact",
                                "unsupported_claim": False})
    return {"items": items}, {"response_mapping": mapping}, {
        "reviewer": "test reviewer", "review_method": "semantic", "reviewed_unix": 1, "ratings": ratings}


class ABCReportingTest(unittest.TestCase):
    def test_protocol_mutation_cannot_reuse_experiment(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            freeze_manifest(path, {"seed": 3407, "dataset_hash": "first"})
            original = path.read_bytes()
            freeze_manifest(path, {"seed": 3407, "dataset_hash": "first"})
            with self.assertRaises(ValueError):
                freeze_manifest(path, {"seed": 3407, "dataset_hash": "edited"})
            self.assertEqual(path.read_bytes(), original)

    def test_macro_metric_keeps_equal_domain_weight(self):
        document, key, ratings = review_fixture()
        with patch("report_abc.paired_interval", return_value={"low": 0, "high": 1}):
            result = aggregate(document, key, ratings)
        self.assertEqual(result["summary"]["A"]["primary_macro_accuracy"], .25)
        self.assertEqual(result["summary"]["A"]["paired_items"]["improved"], 4)
        self.assertEqual(result["highest_observed_adapter_score"], ["A"])

    def test_missing_duplicate_and_unreasoned_ratings_rejected(self):
        document, key, ratings = review_fixture()
        mutations = []
        missing = copy.deepcopy(ratings)
        missing["ratings"].pop()
        mutations.append(missing)
        duplicate = copy.deepcopy(ratings)
        duplicate["ratings"].append(duplicate["ratings"][0])
        mutations.append(duplicate)
        unreasoned = copy.deepcopy(ratings)
        unreasoned["ratings"][0]["rubric_evidence"] = ""
        mutations.append(unreasoned)
        for bad in mutations:
            with self.assertRaises(ValueError):
                aggregate(document, key, bad)

    def test_paired_bootstrap_preserves_identical_predictions(self):
        interval = paired_interval({group: [0, 0, 0] for group in PRIMARY_GROUPS}, draws=100)
        self.assertEqual((interval["low"], interval["high"]), (0, 0))

    def test_unsupported_claim_cannot_receive_full_credit(self):
        document, key, ratings = review_fixture()
        row = next(row for row in ratings["ratings"] if row["score"] == 1)
        row["unsupported_claim"] = True
        with self.assertRaisesRegex(ValueError, "unsupported claim"):
            aggregate(document, key, ratings)

    def test_missing_or_nonstring_rationale_is_not_evidence(self):
        document, key, ratings = review_fixture()
        for field in ("reason", "rubric_evidence"):
            for bad_value in (None, 12, False, [], "   "):
                broken = copy.deepcopy(ratings)
                broken["ratings"][0][field] = bad_value
                with self.assertRaises(ValueError):
                    aggregate(document, key, broken)

    def test_response_key_cannot_duplicate_or_reassign_models(self):
        document, key, ratings = review_fixture()
        first_id = next(iter(key["response_mapping"]))
        duplicate_model = copy.deepcopy(key)
        duplicate_model["response_mapping"][first_id]["model"] = "A"
        with self.assertRaisesRegex(ValueError, "exactly one response"):
            aggregate(document, duplicate_model, ratings)
        wrong_item = copy.deepcopy(key)
        wrong_item["response_mapping"][first_id]["item_id"] = "another-question"
        with self.assertRaisesRegex(ValueError, "another suite/item"):
            aggregate(document, wrong_item, ratings)

    def test_packet_response_ids_must_exactly_match_key(self):
        document, key, ratings = review_fixture()
        duplicate = copy.deepcopy(document)
        duplicate["items"][1]["responses"][0]["response_id"] = duplicate["items"][0]["responses"][0]["response_id"]
        with self.assertRaisesRegex(ValueError, "unique nonempty"):
            aggregate(duplicate, key, ratings)
        extra_key = copy.deepcopy(key)
        extra_key["response_mapping"]["unlisted-answer"] = dict(next(iter(extra_key["response_mapping"].values())))
        with self.assertRaisesRegex(ValueError, "different response IDs"):
            aggregate(document, extra_key, ratings)

    def test_report_mismatch_rejected_before_semantic_aggregation(self):
        base = {"status": "complete", "limit": None, "benchmark_sha256": "frozen",
                "base_model": {"revision": "pinned"}, "generation": {"max_new_tokens": 192},
                "results": [{"id": "q", "prompt": "question", "reference": "answer", "metadata": {}}]}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for model in MODELS:
                report = dict(base, adapter=None if model == "baseline" else f"saved/{model}")
                if model == "C":
                    report["generation"] = {"max_new_tokens": 96}
                (root / f"{model}.json").write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_reports(root)


if __name__ == "__main__":
    unittest.main()
