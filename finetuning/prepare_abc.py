"""Freeze a small, auditable ABC protocol before training; never imports AGATA.

Book material is read only from reviewed, ignored local artifacts. No model is
loaded: only the cached pinned tokenizer is used for the pre-training size audit.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
from importlib import metadata as packages
import json
import math
from pathlib import Path
import re
import subprocess
import sys

from abc_persona_seed import (GENERAL_BENCHMARK, PERSONA_BENCHMARK, SYSTEM_PROMPT,
                              UNKNOWN_BENCHMARK, VALIDATION, persona_training)
from common import (LAB, encode_assistant_only, file_hash, lab_path, read_config,
                    read_jsonl, runtime_environment, write_json)

GROUPS = ("persona", "known_paraphrase", "uncertainty", "general_skills")


def normal(text):
    return " ".join(re.findall(r"\w+", text.casefold().replace("ё", "е")))


def user_text(row):
    return "\n".join(m["content"] for m in row["messages"] if m["role"] == "user")


def private_destination(path):
    path = path.resolve()
    try:
        relative = path.relative_to(LAB.parent.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError("ABC artifacts must stay inside the ignored lab workspace") from exc
    ignored = subprocess.run(["git", "check-ignore", "--no-index", "--quiet", relative], cwd=LAB.parent)
    tracked = subprocess.run(["git", "ls-files", "--", relative], cwd=LAB.parent, capture_output=True, text=True)
    if ignored.returncode or tracked.returncode or tracked.stdout.strip():
        raise ValueError(f"Private output must be ignored and untracked: {path}")
    return path


def persona_row(kind, family, prompt, answer, partition):
    key = f"abc_persona_{partition}_{family}"
    return {"id": key, "group_id": f"persona:{family}", "family_id": family,
            "category": "personality", "tools": [], "messages": [
                {"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt},
                {"role": "assistant", "content": answer}],
            "metadata": {"source": f"authored_persona_{kind}", "review_status": "approved",
                         "reviewer": "Codex authored-scenario review; not independent human review",
                         "family_id": family, "partition": partition, "private_data": False,
                         "fictional": False, "domain": "persona", "tools_unavailable": True,
                         "review_note": "Distinct scenario, concise response, no factual curriculum, no executed action."}}


def benchmark_row(group, category, item):
    family, prompt, reference, rubric = item
    return {"id": f"ABC_{category}_{family}", "family_id": family, "category": category,
            "prompt": prompt, "reference": reference, "expected_keywords": [], "tools": [],
            "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
            "metadata": {"primary_group": group, "family_id": family, "rubric": rubric,
                         "seen_question": False, "source": "independently_authored_abc_holdout",
                         "fictional": False, "real_book_evidence": False,
                         "unanswerable": group == "uncertainty", "evaluation_partition": "test"}}


def standardize_book(rows, training=False):
    result = deepcopy(rows)
    for row in result:
        # A shared minimal instruction removes a system-prompt condition confound.
        row["messages"] = [{"role": "system", "content": SYSTEM_PROMPT}] + [
            m for m in row["messages"] if m["role"] != "system"]
        meta = row["metadata"]
        row["group_id"] = f"book:{meta['source_id']}:{meta.get('chapter_id') or meta['family_id']}"
        meta["domain"] = "book"
        meta.setdefault("fictional", False)
        if training:
            row["category"] = "book_knowledge"
    return result


def validate_book(row, extracted):
    """Reject unreviewed/fictional evidence and check the actual extracted page."""
    from book_pipeline import complete_source_quote
    meta = row["metadata"]
    if meta.get("review_status") != "approved" or not meta.get("reviewer") or not meta.get("real_book_evidence"):
        raise ValueError(f"{row['id']}: book record is not approved")
    for item in meta.get("supporting_evidence") or [meta]:
        if (item.get("review_status") != "approved" or not item.get("reviewer")
                or item.get("fictional") is True or not item.get("real_book_evidence")
                or item.get("training_mode") not in ("source_attributed", "evidence_bound_uncertainty")):
            raise ValueError(f"{row['id']}: unreviewed or unsupported book material")
        if item.get("source_id") != extracted["source"]["source_id"]:
            raise ValueError(f"{row['id']}: source hash differs")
        evidence = item.get("evidence", "")
        source = "\n".join(unit["text"] for unit in extracted["units"] if unit["locator"] == item.get("locator"))
        if not evidence or not source or not complete_source_quote(evidence, source):
            raise ValueError(f"{row['id']}: quoted evidence is not a full sentence from its source page")


def validate_sets(persona_train, persona_val, book_train, book_val, benchmark, memorization):
    """Exact question splits and explicit known-fact exceptions, never blanket dedup."""
    train = persona_train + book_train
    validation = persona_val + book_val
    for label, rows in (("train", train), ("validation", validation), ("benchmark", benchmark), ("memorization", memorization)):
        if not rows:
            raise ValueError(f"Empty {label}")
        ids = [r["id"] for r in rows]
        prompts = [normal(user_text(r)) for r in rows]
        if len(ids) != len(set(ids)) or len(prompts) != len(set(prompts)):
            raise ValueError(f"Duplicate IDs/questions within {label}")
        for row in rows:
            if row["messages"][0] != {"role": "system", "content": SYSTEM_PROMPT} or row.get("tools"):
                raise ValueError("All conditions must use the identical common system and no tools")
            if any(m.get("tool_calls") for m in row["messages"]):
                raise ValueError("ABC does not train tool trajectories")
            expected_last_role = "assistant" if label in ("train", "validation") else "user"
            if row["messages"][-1]["role"] != expected_last_role:
                raise ValueError("Unexpected transcript role")
            if label in ("benchmark", "memorization") and not row.get("metadata", {}).get("rubric", {}).get("score_binary"):
                raise ValueError("Every evaluation item needs a preregistered binary rubric")
    prompts = {name: {normal(user_text(r)) for r in rows} for name, rows in
               (("train", train), ("validation", validation), ("benchmark", benchmark))}
    for first, second in (("train", "validation"), ("train", "benchmark"), ("validation", "benchmark")):
        if prompts[first] & prompts[second]:
            raise ValueError(f"Exact question leakage: {first}/{second}")
    family = lambda rows: {r["family_id"] for r in rows}
    persona_bench = [r for r in benchmark if r["metadata"].get("primary_group") == "persona"]
    if (family(persona_train) & family(persona_val) or family(persona_train) & family(persona_bench)
            or family(persona_val) & family(persona_bench)):
        raise ValueError("Persona scenario family leaked across partitions")
    train_chapters = {r["metadata"].get("chapter_id") for r in book_train}
    held = [r for r in benchmark if r["metadata"].get("secondary_group") == "heldchapter"]
    if train_chapters & {r["metadata"].get("chapter_id") for r in held}:
        raise ValueError("Held-out book chapter leaked into training")
    if any(r["metadata"].get("fictional") or not r["metadata"].get("real_book_evidence") for r in book_train):
        raise ValueError("B must contain only verified real-source material")
    train_questions = {normal(user_text(r)): r for r in book_train}
    for row in memorization:
        if normal(user_text(row)) not in train_questions:
            raise ValueError("Exact-memory probe must use an actual book training question")
    return {"exact_train_validation_overlap": 0, "exact_train_primary_overlap": 0,
            "persona_scenario_family_overlap": 0, "held_chapter_overlap": 0,
            "known_fact_overlap_is_intentional": True,
            "book_validation_is_known_fact_new_formulation": True,
            "memory_diagnostics_are_intentionally_seen_questions": len(memorization)}


def token_stats(tokenizer, rows, max_length):
    encoded = [encode_assistant_only(tokenizer, row, max_length) for row in rows]
    tokens = [len(row["input_ids"]) for row in encoded]
    labels = [sum(t != -100 for t in row["labels"]) for row in encoded]
    return {"rows": len(rows), "sequence_tokens": sum(tokens), "supervised_tokens": sum(labels),
            "max_sequence_tokens": max(tokens), "mean_supervised_tokens": sum(labels) / len(labels),
            "domain_counts": dict(Counter(r["metadata"]["domain"] for r in rows)),
            "unique_facts_or_scenarios": len({r["metadata"].get("fact_id") or r["family_id"] for r in rows}),
            "overlength_rows": 0}


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--book-dir", default="books/processed/abc_v1")
    parser.add_argument("--extracted", default="books/polivalina_2013.extracted.json")
    parser.add_argument("--output", default="datasets/abc_v1")
    parser.add_argument("--steps", type=int, default=32)
    args = parser.parse_args()
    if args.steps != 32:
        raise ValueError("ABC v1 preregisters exactly 32 optimizer steps; use a new protocol for other budgets")
    output = private_destination(lab_path(args.output))
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Frozen ABC artifacts already exist; choose a new private protocol directory")
    book_dir = lab_path(args.book_dir)
    extracted = json.loads(lab_path(args.extracted).read_text(encoding="utf-8"))
    parts = {part: read_jsonl(book_dir / f"{part}.jsonl") for part in
             ("train", "validation", "benchmark_part", "memorization_part")}
    for rows in parts.values():
        for row in rows:
            validate_book(row, extracted)
    pt = [persona_row(kind, family, prompt, answer, "train") for kind, family, prompt, answer in persona_training()]
    pv = [persona_row("new", family, prompt, answer, "validation") for family, prompt, answer in VALIDATION]
    bt, bv = (standardize_book(parts[key], training=True) for key in ("train", "validation"))
    benchmark = ([benchmark_row("persona", "A", item) for item in PERSONA_BENCHMARK[:10]] +
                 [benchmark_row("general_skills", "F", item) for item in GENERAL_BENCHMARK[:6]] +
                 [benchmark_row("uncertainty", "E", item) for item in UNKNOWN_BENCHMARK] +
                 standardize_book(parts["benchmark_part"]))
    memory = standardize_book(parts["memorization_part"])
    quality = validate_sets(pt, pv, bt, bv, benchmark, memory)
    # Previous smoke evaluation has been read; do not silently reuse its questions.
    old_bench = LAB / "evaluation" / "benchmark.jsonl"
    if old_bench.exists() and ({normal(user_text(r)) for r in read_jsonl(old_bench)} &
                               {normal(user_text(r)) for r in benchmark}):
        raise ValueError("New benchmark repeats an old benchmark question")
    config = read_config()
    runtime_environment()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"], revision=config["model"]["revision"],
                                              local_files_only=True, trust_remote_code=False)
    variants = {"A": (pt, pv), "B": (bt, bv), "C": (pt + bt, pv + bv)}
    stats = {key: {"train": token_stats(tokenizer, rows[0], config["model"]["max_seq_length"]),
                   "validation": token_stats(tokenizer, rows[1], config["model"]["max_seq_length"])}
             for key, rows in variants.items()}
    effective_batch = config["training"]["batch_size"] * config["training"]["gradient_accumulation_steps"]
    for key, value in stats.items():
        n = value["train"]["rows"]
        value["budget"] = {"optimizer_steps": args.steps, "effective_batch_size": effective_batch,
                            "nominal_example_presentations": args.steps * effective_batch,
                            "estimated_epochs": args.steps * effective_batch / n,
                            "steps_per_full_epoch": math.ceil(n / effective_batch),
                            "expected_domain_presentations": {d: args.steps * effective_batch * count / n
                                 for d, count in value["train"]["domain_counts"].items()},
                            "expected_supervised_token_presentations": args.steps * effective_batch * value["train"]["mean_supervised_tokens"],
                            "note": "Nominal estimates, not measured sampler exposure; no token-budget matching."}
    for key, (train, val) in variants.items():
        write_jsonl(output / f"{key}_train.jsonl", train)
        write_jsonl(output / f"{key}_validation.jsonl", val)
        cfg = deepcopy(config)
        cfg["training"]["max_steps"] = args.steps
        cfg["training"]["save_steps"] = 32
        cfg["data"] = {"train": (output / f"{key}_train.jsonl").relative_to(LAB).as_posix(),
                       "validation": (output / f"{key}_validation.jsonl").relative_to(LAB).as_posix(),
                       "smoke": config["data"]["smoke"]}
        cfg["evaluation"] = {"benchmark": (output / "benchmark.jsonl").relative_to(LAB).as_posix(), "max_new_tokens": 192}
        import yaml
        (output / "configs").mkdir(exist_ok=True)
        (output / "configs" / f"{key}.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    write_jsonl(output / "benchmark.jsonl", benchmark)
    write_jsonl(output / "memorization.jsonl", memory)
    protocol = {"protocol_id": "agata_abc_v1", "created_utc": datetime.now(timezone.utc).isoformat(),
                "status": "frozen_before_model_outputs", "seed": 3407, "max_steps": args.steps,
                "system_prompt": SYSTEM_PROMPT, "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
                "base_model": config["model"], "lora": config["lora"], "variants": stats,
                "quality_audit": quality,
                "benchmark": {"n": len(benchmark), "memorization_n": len(memory),
                              "primary_counts": dict(Counter(r["metadata"].get("primary_group") for r in benchmark if r["metadata"].get("primary_group"))),
                              "secondary_counts": dict(Counter(r["metadata"].get("secondary_group") for r in benchmark if r["metadata"].get("secondary_group"))),
                              "primary_metric": "Unweighted arithmetic mean of the four binary semantic group accuracies",
                              "primary_groups": list(GROUPS), "rubric_location": "benchmark.jsonl metadata.rubric",
                              "passing_rule": "All must_include semantics and no must_not violation; unsupported substantive extra claim fails. Surface wording may differ.",
                              "review": "Blind model labels before semantic scoring; AI-assisted review, not independent human evaluation.",
                              "selection": "Report highest measured primary macro-average. Ties remain ties. No significance claim from a single small run."},
                "limitations": [
                    "48 persona scenarios and 16 book claims plus four source-boundary examples are a small pilot, not broad training data.",
                    "The 2013 source is a 98-page preview, not the full book. Only reviewed neutral chapter-I claims are trained.",
                    "Known-fact paraphrases share facts with training by design: measure retrieval under new wording, not acquisition of unseen facts.",
                    "Book validation shares known facts and is loss-only monitoring; no checkpoint/hyperparameter selection uses it.",
                    "C is the exact union without duplicated rows: equal optimizer steps imply fewer exposures per domain and unequal target-token budgets.",
                    "Held-out Introduction scores are secondary and cannot show learning of information never supplied.",
                    "A common system already prompts honesty and brevity, so baseline has these instructions too.",
                    "No test assesses real tool execution, current factual knowledge or deployment readiness.",
                    "One seed, no independent human curation/scoring and a small benchmark limit statistical conclusions."],
                "source_files": {p.relative_to(LAB).as_posix(): file_hash(p) for p in
                     [Path(__file__), LAB / "abc_persona_seed.py", LAB / "seed_data.py", lab_path(args.extracted)] + sorted(book_dir.glob("*.json*"))},
                "library_versions": {"python": sys.version, **{name: packages.version(name) for name in
                    ("torch", "transformers", "trl", "peft", "datasets", "bitsandbytes", "unsloth", "unsloth_zoo", "huggingface-hub")}},
                "artifact_sha256": {p.relative_to(output).as_posix(): file_hash(p) for p in sorted(output.rglob("*")) if p.is_file()}}
    write_json(output / "protocol.json", protocol)
    write_json(output / "protocol.sha256.json", {"protocol.json": file_hash(output / "protocol.json")})
    print(json.dumps({"output": str(output), "stats": stats, "benchmark": protocol["benchmark"], "quality_audit": quality}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
