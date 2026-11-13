"""Guards for the new diagnostic pipeline; CPU only, no model downloads."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import file_hash
from evaluate_diagnostics import generation_row, lexical_scores, teacher_forced_nll
from train_diagnostics import safe_output, variant_settings, verify_one_fact_gate
from report_diagnostics import summarize_groups, checkpoint_step


class DiagnosticRuntimeTest(unittest.TestCase):
    def test_reference_and_rubric_are_not_generation_inputs(self):
        row = {"messages": [{"role": "system", "content": "brief"}, {"role": "user", "content": "What code?"}],
               "metadata": {"reference": "SECRET_IDENTIFIER", "expected_answer": "SECRET_IDENTIFIER"}}
        actual = generation_row(row)
        self.assertNotIn("SECRET_IDENTIFIER", json.dumps(actual))
        bad = dict(row, messages=row["messages"] + [{"role": "assistant", "content": "SECRET_IDENTIFIER"}])
        with self.assertRaises(ValueError):
            generation_row(bad)

    def test_exact_and_inclusion_are_separate(self):
        target = {"expected_answer": "AB-91XZ"}
        scores = lexical_scores("Код: AB-91XZ.", target)
        self.assertFalse(scores["exact_normalized"])
        self.assertTrue(scores["answer_inclusion"])
        self.assertFalse(lexical_scores("AB-91XZextra", target)["answer_inclusion"])
        self.assertFalse(lexical_scores("AB-91XZ-extra", target)["answer_inclusion"])
        self.assertTrue(lexical_scores(" ab-91xz. ", target)["exact_normalized"])
        self.assertFalse(lexical_scores("AB91XZ", target)["exact_normalized"])
        self.assertIsNone(lexical_scores("Не знаю", {})["exact_normalized"])
        self.assertFalse(lexical_scores("17", {"expected_answer": "7"})["answer_inclusion"])

    def test_output_cannot_target_abc_or_runtime(self):
        for path in ("outputs/abc_v1/A", "../data", "outputs/smoke"):
            with self.assertRaises(ValueError):
                safe_output(path)

    def test_mini_gate_rejects_unverified_or_changed_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            benchmark = root / "one_fact_benchmark.jsonl"
            benchmark.write_text('{}\n', encoding="utf-8")
            path = root / "gate.json"
            report = {"status": "complete", "adapter": "saved", "adapter_reload_verified": True,
                      "benchmark_sha256": file_hash(benchmark), "one_fact_gate": {"passed": True}}
            path.write_text(json.dumps(report), encoding="utf-8")
            self.assertTrue(verify_one_fact_gate(path, root)["gate"]["passed"])
            benchmark.write_text('{"changed":true}\n', encoding="utf-8")
            with self.assertRaises(ValueError):
                verify_one_fact_gate(path, root)

    def test_fixed_milestones_and_style_budget(self):
        self.assertEqual(variant_settings("one_fact")["milestones"], [20, 50, 100])
        self.assertEqual(variant_settings("mini")["warmup_steps"], 2)
        self.assertEqual(variant_settings("style")["max_steps"], 50)
        self.assertEqual(variant_settings("style")["learning_rate"], 1e-4)

    def test_teacher_forcing_uses_shifted_assistant_labels(self):
        import torch
        class Model:
            device = "cpu"
            def __call__(self, **kwargs):
                # Position 0 predicts target 1; position 1 predicts target 2.
                return SimpleNamespace(logits=torch.tensor([[[0., 3., 0.], [0., 0., 3.], [9., 0., 0.]]]))
        row = {"id": "q", "messages": [{"role": "user", "content": "Q"}], "metadata": {"reference": "A"}}
        encoded = {"input_ids": [0, 1, 2], "labels": [-100, 1, 2]}
        with patch("evaluate_diagnostics.encode_assistant_only", return_value=encoded):
            result = teacher_forced_nll(Model(), None, row, 10)
        self.assertEqual(result["target_token_ids"], [1, 2])
        self.assertEqual(result["target_tokens"], 2)
        self.assertLess(result["mean_nll"], .1)

    def test_unknown_style_and_format_do_not_enter_fact_recall(self):
        rows = []
        for group, exact in (("exact", True), ("paraphrase", False), ("unknown", None), ("format", True), ("style", None)):
            rows.append({"id": group, "metadata": {"task_type": group},
                         "lexical": {"exact_normalized": exact, "answer_inclusion": exact},
                         "teacher_forced": {"target_tokens": 2, "total_nll": 3.0},
                         "hit_token_limit": False, "unclosed_think": False})
        groups = summarize_groups({"results": rows}, 0)
        self.assertEqual(groups["exact"]["n"], 1)
        self.assertEqual(groups["paraphrase"]["exact_normalized_count"], 0)
        self.assertIsNone(groups["unknown"]["exact_normalized_count"])
        self.assertIsNone(groups["style"]["semantic_correct"])
        self.assertEqual(groups["format"]["exact_normalized_count"], 1)

    def test_checkpoint_labels_require_verified_fresh_adapter(self):
        self.assertEqual(checkpoint_step({"adapter": None}), 0)
        self.assertEqual(checkpoint_step({"adapter": "outputs/diagnostics_v1/mini/checkpoint-50/adapter", "adapter_reload_verified": True}), 50)
        with self.assertRaises(ValueError):
            checkpoint_step({"adapter": "outputs/diagnostics_v1/mini/checkpoint-50/adapter", "adapter_reload_verified": False})


if __name__ == "__main__":
    unittest.main()
