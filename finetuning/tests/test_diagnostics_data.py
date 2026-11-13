"""Provenance and inference-boundary regressions for fresh diagnostics data."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prepare_diagnostics import audit_rows, build_mini, build_one_fact, build_style


class CharTokenizer:
    def apply_chat_template(self, messages, **_):
        return "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)

    def __call__(self, text, **_):
        return {"input_ids": list(map(ord, text)), "offset_mapping": [(i, i + 1) for i in range(len(text))]}


def parts():
    one = build_one_fact("СЕВРА-TESTCODE")
    mini = build_mini()
    style = build_style()
    return {domain: dict(zip(("train", "validation", "benchmark"), source[:3]))
            for domain, source in (("one_fact", one), ("mini", mini), ("style", style))}


class DiagnosticsDataTest(unittest.TestCase):
    def test_counts_exact_probes_and_no_reference_in_model_inputs(self):
        data = parts()
        audit = audit_rows(data, CharTokenizer(), 5000)
        self.assertEqual(audit["one_fact"]["benchmark_task_counts"], {"exact": 3, "paraphrase": 3, "unknown": 2, "format": 1})
        self.assertEqual(audit["mini"]["benchmark_task_counts"], {"exact": 16, "paraphrase": 8, "unknown": 2, "format": 1})
        self.assertEqual(audit["style"]["benchmark_task_counts"], {"style": 12, "general": 6, "honesty": 4})
        gate = [r for r in data["one_fact"]["benchmark"] if r["metadata"]["gate_required"]]
        self.assertEqual([r["id"] for r in gate], ["diag_one_exact_01"])
        self.assertEqual(gate[0]["metadata"]["expected_answer"], "СЕВРА-TESTCODE")

    def test_rejects_expected_answer_in_user_prompt(self):
        data = parts()
        data["one_fact"]["benchmark"][3]["messages"][-1]["content"] += " Подсказка: СЕВРА-TESTCODE."
        with self.assertRaisesRegex(ValueError, "Answer leaked"):
            audit_rows(data, CharTokenizer(), 5000)

    def test_rejects_evaluation_assistant_target(self):
        data = parts()
        data["one_fact"]["benchmark"][0]["messages"].append({"role": "assistant", "content": "СЕВРА-TESTCODE"})
        with self.assertRaisesRegex(ValueError, "Evaluation answer"):
            audit_rows(data, CharTokenizer(), 5000)

    def test_rejects_unlabelled_train_question_in_paraphrases(self):
        data = parts()
        data["mini"]["benchmark"][0]["metadata"]["task_type"] = "paraphrase"
        with self.assertRaisesRegex(ValueError, "labelled exact"):
            audit_rows(data, CharTokenizer(), 5000)

    def test_rejects_style_scenario_reuse_and_system_domain_hint(self):
        data = parts()
        data["style"]["benchmark"][0]["family_id"] = data["style"]["train"][0]["family_id"]
        with self.assertRaisesRegex(ValueError, "Style scenario"):
            audit_rows(data, CharTokenizer(), 5000)
        data = parts()
        data["style"]["train"][0]["messages"][0]["content"] += " Always cite the book."
        with self.assertRaisesRegex(ValueError, "System/tools differ"):
            audit_rows(data, CharTokenizer(), 5000)

    def test_fact_entities_and_targets_are_consistent_across_training_variants(self):
        for domain in ("one_fact", "mini"):
            data = parts()[domain]
            targets = {}
            for row in data["train"] + data["validation"]:
                fact_id = row["metadata"]["fact_id"]
                answer = row["messages"][-1]["content"]
                targets.setdefault(fact_id, set()).add(answer)
            self.assertTrue(all(len(answers) == 1 for answers in targets.values()))
            self.assertEqual(len(targets), 1 if domain == "one_fact" else 8)


if __name__ == "__main__":
    unittest.main()
