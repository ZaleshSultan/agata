"""CPU-only tests for prospective budgets, stopping and validation-only selection."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import file_hash
from evaluate_large import check_rows, lexical_answers, nll_row
from evaluate_diagnostics import generation_row
from train_large import ValidationStopper, arm_settings, checked_config, planned_steps, safe_output
from report_large import evaluation_completeness, load_ratings, select_checkpoint, validation_guard


def validation(arm="baseline", step=0):
    return {"split": "validation", "arm": arm, "step": step,
            "results": [{"id": kind, "metadata": {"task_type": kind}} for kind in
                        ("general", "unknown", "false_action", "persona")]}


def complete_ratings(reports):
    return {(key, row["id"]): {"score": 1, "unsupported_claim": False, "reason": "All rubric criteria met"}
            for key, record in reports.items() for row in record["report"]["results"]}


class LargeRuntimeTests(unittest.TestCase):
    def test_epoch_plan_uses_ceiling_not_stale_step_override(self):
        self.assertEqual(planned_steps(121), {"steps_per_epoch": 31, "maximum_steps": 93,
                                           "maximum_presentations": 363, "warmup_steps": 5})
        for arm, lr in (("lr5e-5", 5e-5), ("lr1e-4", 1e-4)):
            settings = arm_settings(arm)
            self.assertEqual((settings["epochs"], settings["max_steps"]), (3, -1))
            self.assertEqual(settings["learning_rate"], lr)
            self.assertEqual(settings["lr_scheduler_type"], "linear")
            self.assertEqual(settings["warmup_ratio"], .05)
            self.assertIsNone(settings["save_total_limit"])
        with self.assertRaises(ValueError):
            planned_steps(0)

    def test_checked_config_replaces_stale_yaml_budget(self):
        config = {"model": {"name": "unsloth/Qwen3-1.7B-bnb-4bit",
                             "revision": "d1d7781199f38692496fab70012c39598f0c858f",
                             "max_seq_length": 1024, "load_in_4bit": True},
                  "lora": {"r": 16, "alpha": 16, "dropout": 0,
                           "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]},
                  "training": {"max_steps": 32, "epochs": 9}}
        with patch("train_large.read_config", return_value=config):
            result = checked_config("unused", "lr5e-5")
        self.assertEqual(result["training"]["max_steps"], -1)
        self.assertEqual(result["training"]["epochs"], 3)
        self.assertEqual(config["training"]["max_steps"], 32)

    def test_stopping_needs_two_consecutive_nonsignificant_epochs(self):
        stopper = ValidationStopper(2.)
        first = stopper.observe(1, 1.99)
        self.assertEqual(first["stale_epochs"], 1)
        self.assertIsNone(first["stop_reason"])
        # Improvement is measured against significant best (2), not last epoch.
        second = stopper.observe(2, 1.98)
        self.assertTrue(second["significant_improvement"])
        self.assertEqual(second["stale_epochs"], 0)
        self.assertEqual(stopper.observe(3, 1.975)["stop_reason"], "maximum_three_epochs")
        with self.assertRaises(ValueError):
            stopper.observe(4, 1.)
        stalled = ValidationStopper(2.)
        stalled.observe(1, 1.999)
        self.assertEqual(stalled.observe(2, 1.998)["stop_reason"], "two_epochs_without_one_percent_improvement")
        self.assertEqual(stalled.best_absolute, 1.998)
        self.assertEqual(stalled.best_significant, 2.)

    def test_catastrophic_boundary_nan_and_sequential_epochs(self):
        stopper = ValidationStopper(2.)
        self.assertFalse(stopper.observe(1, 3.)["catastrophic"])
        stopper = ValidationStopper(2.)
        self.assertTrue(stopper.observe(1, 3.0001)["catastrophic"])
        with self.assertRaises(FloatingPointError):
            ValidationStopper(2.).observe(1, float("nan"))
        with self.assertRaises(ValueError):
            ValidationStopper(float("inf"))
        with self.assertRaises(ValueError):
            ValidationStopper(2.).observe(2, 1.)

    def test_outputs_preserve_previous_experiments_and_runtime(self):
        for path in ("outputs/abc_v1/A", "outputs/diagnostics_v1/style", "../memory", "outputs/smoke"):
            with self.assertRaises(ValueError):
                safe_output(path)

    def test_references_are_used_only_for_nll_not_generation(self):
        row = {"id": "a", "messages": [{"role": "user", "content": "Question"}],
               "metadata": {"task_type": "known_exact", "reference_answer": "SECRET", "expected_answers": ["SECRET"]}}
        self.assertEqual(nll_row(row)["metadata"]["reference"], "SECRET")
        self.assertNotIn("reference", row["metadata"])
        self.assertNotIn("SECRET", json.dumps(generation_row(row)))
        check_rows([row])
        with self.assertRaises(ValueError):
            check_rows([row, row])

    def test_answer_alternatives_and_boundaries_are_not_semantic_truth(self):
        expected = {"expected_answers": ["семь", "7"]}
        self.assertTrue(lexical_answers("Семь.", expected)["exact_normalized"])
        self.assertFalse(lexical_answers("17", expected)["answer_inclusion"])
        result = lexical_answers("Ответ 7, но на самом деле восемь.", expected)
        self.assertFalse(result["exact_normalized"])
        self.assertTrue(result["answer_inclusion"])
        self.assertIsNone(lexical_answers("Короткая шутка", {})["exact_normalized"])
        with self.assertRaises(ValueError):
            lexical_answers("x", {"expected_answers": "x"})

    def test_guard_needs_all_validation_ratings_not_only_guard_subset(self):
        base, cand = validation(), validation("lr5e-5", 10)
        reports = {"validation:baseline:0": {"report": base}, "validation:lr5e-5:10": {"report": cand}}
        ratings = complete_ratings(reports)
        ratings.pop(("validation:lr5e-5:10", "persona"))
        result = validation_guard(base, cand, ratings)
        self.assertFalse(result["passed"])
        self.assertEqual(result["status"], "pending_complete_validation_ratings")

    def test_unknown_failure_is_not_hidden_by_honesty_success(self):
        base, cand = validation(), validation("lr5e-5", 10)
        reports = {"validation:baseline:0": {"report": base}, "validation:lr5e-5:10": {"report": cand}}
        ratings = complete_ratings(reports)
        ratings[("validation:lr5e-5:10", "unknown")].update(score=0, unsupported_claim=True)
        guard = validation_guard(base, cand, ratings)
        self.assertFalse(guard["passed"])
        self.assertTrue(guard["groups"]["honesty"]["passed"])
        self.assertFalse(guard["groups"]["unknown"]["passed"])
        with self.assertRaises(ValueError):
            validation_guard(dict(base, split="benchmark"), cand, ratings)

    def test_selection_uses_loss_then_step_then_lr_and_ignores_test_scores(self):
        candidates = [{"arm": "lr1e-4", "step": 20, "validation_loss": .4},
                      {"arm": "lr5e-5", "step": 20, "validation_loss": .4},
                      {"arm": "lr5e-5", "step": 10, "validation_loss": .4}]
        reports = {"validation:baseline:0": {"report": validation()}}
        for c in candidates:
            key = f"validation:{c['arm']}:{c['step']}"
            reports[key] = {"report": validation(c["arm"], c["step"])}
        reports["benchmark:lr1e-4:20"] = {"report": dict(validation("lr1e-4", 20), split="benchmark"), "test_score": 100}
        ratings = complete_ratings(reports)
        selected = select_checkpoint(candidates, reports, ratings)
        self.assertEqual(selected["recommended"], "lr5e-5:10")
        self.assertEqual(select_checkpoint(candidates[:2], reports, ratings)["recommended"], "lr5e-5:20")
        ratings[("validation:lr5e-5:10", "unknown")]["score"] = 0
        self.assertEqual(select_checkpoint(candidates, reports, ratings)["recommended"], "lr5e-5:20")
        for c in candidates:
            ratings[(f"validation:{c['arm']}:{c['step']}", "general")]["score"] = 0
        self.assertEqual(select_checkpoint(candidates, reports, ratings)["recommended"], "baseline")

    def test_ratings_bind_hashes_and_reject_unsupported_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ratings.json"
            reports = {"validation:baseline:0": {"report": validation(), "sha256": "frozen"}}
            rating = {"report_id": "validation:baseline:0", "id": "general", "score": 1,
                      "reason": "OK", "unsupported_claim": False}
            document = {"reviewer": "reviewer", "review_method": "blind rubrics",
                        "report_sha256": {"validation:baseline:0": "frozen"}, "ratings": [rating]}
            path.write_text(json.dumps(document), encoding="utf-8")
            self.assertEqual(len(load_ratings(path, reports)[0]), 1)
            for mutation in ({"unsupported_claim": True}, {"reason": " "}, {"score": True}):
                document["ratings"] = [dict(rating, **mutation)]
                path.write_text(json.dumps(document), encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_ratings(path, reports)
            document["ratings"] = [rating]
            document["report_sha256"]["validation:baseline:0"] = "changed"
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_ratings(path, reports)

    def test_validation_selection_does_not_claim_test_completion(self):
        reports = {"validation:baseline:0": {"report": validation()},
                   "validation:lr5e-5:10": {"report": validation("lr5e-5", 10)}}
        ratings = complete_ratings(reports)
        selection = {"status": "complete", "recommended": "lr5e-5:10"}
        candidates = [{"arm": "lr5e-5", "step": 10}]
        completeness = evaluation_completeness(selection, reports, ratings, candidates)
        self.assertFalse(completeness["complete"])
        self.assertIn("benchmark:lr5e-5:10", completeness["missing_reports"])
        self.assertIn("recall:baseline:0", completeness["missing_reports"])
        self.assertEqual(completeness["missing_semantic_rating_count"], 0)

    def test_rejected_checkpoint_still_needs_all_heldout_outcomes(self):
        candidates = [{"arm": "lr5e-5", "step": 10}, {"arm": "lr1e-4", "step": 10, "admissible": False}]
        reports = {}
        for split in ("validation", "benchmark", "recall"):
            for arm, step in (("baseline", 0), ("lr5e-5", 10), ("lr1e-4", 10)):
                if split == "benchmark" and arm == "lr1e-4":
                    continue
                reports[f"{split}:{arm}:{step}"] = {"report": dict(validation(arm, step), split=split)}
        selection = {"status": "complete", "recommended": "lr5e-5:10"}
        completeness = evaluation_completeness(selection, reports, complete_ratings(reports), candidates)
        self.assertFalse(completeness["complete"])
        self.assertEqual(completeness["missing_reports"], ["benchmark:lr1e-4:10"])


if __name__ == "__main__":
    unittest.main()
