"""Leak/provenance tests independent of CUDA and private book files."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from abc_persona_seed import SYSTEM_PROMPT, persona_training
from prepare_abc import validate_book, validate_sets


def row(key, prompt, *, train=False, domain="persona", chapter=None, primary=None, secondary=None):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
    if train:
        messages.append({"role": "assistant", "content": "An answer."})
    return {"id": key, "family_id": key, "messages": messages, "tools": [],
            "metadata": {"primary_group": primary, "secondary_group": secondary,
                         "chapter_id": chapter, "fictional": False, "real_book_evidence": domain == "book",
                         "rubric": {"score_binary": True, "must_include": ["Correct answer"], "must_not": ["Fabrication"]}}}


def fixture():
    pt = [row("persona_train", "An original training situation", train=True)]
    pv = [row("persona_val", "A distinct validation situation", train=True)]
    bt = [row("book_train", "A factual training question", train=True, domain="book", chapter="one")]
    bv = [row("book_val", "A new formulation about that fact", train=True, domain="book", chapter="one")]
    bench = [row("persona_test", "A new persona scenario", primary="persona"),
             row("book_test", "A held-out chapter question", domain="book", chapter="intro", secondary="heldchapter")]
    memory = [row("memory", "A factual training question", domain="book", chapter="one")]
    return [pt, pv, bt, bv, bench, memory]


class ABCDataTest(unittest.TestCase):
    def test_accepts_explicit_known_fact_validation_and_exact_memory(self):
        result = validate_sets(*fixture())
        self.assertEqual(result["exact_train_primary_overlap"], 0)
        self.assertEqual(result["memory_diagnostics_are_intentionally_seen_questions"], 1)

    def test_case_and_punctuation_cannot_hide_exact_test_leak(self):
        data = fixture()
        data[4][0]["messages"][-1]["content"] = "AN ORIGINAL, TRAINING SITUATION!"
        with self.assertRaisesRegex(ValueError, "Exact question leakage"):
            validate_sets(*data)

    def test_persona_family_cannot_leak_with_new_wording(self):
        data = fixture()
        data[4][0]["family_id"] = data[0][0]["family_id"]
        with self.assertRaisesRegex(ValueError, "scenario family"):
            validate_sets(*data)

    def test_held_chapter_cannot_enter_training(self):
        data = fixture()
        data[2][0]["metadata"]["chapter_id"] = "intro"
        with self.assertRaisesRegex(ValueError, "Held-out book chapter"):
            validate_sets(*data)

    def test_fictional_knowledge_rejected_from_B(self):
        data = fixture()
        data[2][0]["metadata"]["fictional"] = True
        with self.assertRaisesRegex(ValueError, "verified real-source"):
            validate_sets(*data)

    def test_different_condition_system_rejected(self):
        data = fixture()
        data[2][0]["messages"][0]["content"] += " Extra personality hints."
        with self.assertRaisesRegex(ValueError, "identical common system"):
            validate_sets(*data)

    def test_memory_probe_must_be_exact_training_prompt(self):
        data = fixture()
        data[5][0]["messages"][-1]["content"] = "An unseen question"
        with self.assertRaisesRegex(ValueError, "Exact-memory"):
            validate_sets(*data)

    def test_evidence_checks_source_hash_page_and_negation_boundary(self):
        extracted = {"source": {"source_id": "sha"}, "units": [
            {"locator": "page-1", "text": "Do not treat a guess as evidence. Another sentence."}]}
        item = {"id": "example", "metadata": {"source_id": "sha", "locator": "page-1",
                "evidence": "Do not treat a guess as evidence.", "review_status": "approved",
                "reviewer": "test", "real_book_evidence": True, "fictional": False,
                "training_mode": "source_attributed"}}
        validate_book(item, extracted)
        for changes in ({"review_status": "draft"}, {"source_id": "wrong"}, {"locator": "page-2"},
                        {"evidence": "treat a guess as evidence."}, {"fictional": True}):
            wrong = deepcopy(item)
            wrong["metadata"].update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_book(wrong, extracted)

    def test_expansion_is_20_new_scenarios_not_question_permutations(self):
        rows = persona_training()
        self.assertEqual(len(rows), 48)
        self.assertEqual(sum(r[0] == "new" for r in rows), 20)
        self.assertEqual(len({r[1] for r in rows}), 48)


if __name__ == "__main__":
    unittest.main()
