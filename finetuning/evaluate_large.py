"""Frozen large-v3 generation, raw-token evidence and separate lexical/NLL probes."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import re
import time

from common import LAB, NvidiaMonitor, file_hash, lab_path, load_model, read_jsonl, render_chat, write_json
from evaluate_diagnostics import (generation_row, lexical_scores, raw_generate,
                                  teacher_forced_nll, tokenizer_audit)
from train_large import PROTOCOL_RULES, checked_config

SPLITS = {"evaluation_validation.jsonl": "validation", "benchmark.jsonl": "benchmark",
          "recall_probes.jsonl": "recall"}


def lexical_answers(response, metadata):
    answers = metadata.get("expected_answers")
    if answers is None:
        one = metadata.get("expected_answer")
        answers = [] if one is None else [one]
    if not isinstance(answers, list) or any(not isinstance(a, str) or not a.strip() for a in answers):
        raise ValueError("expected_answers must be a list of nonempty string alternatives")
    if not answers:
        return lexical_scores(response, {})
    scores = [lexical_scores(response, {"expected_answer": answer,
                                        "case_sensitive": metadata.get("case_sensitive", False)}) for answer in answers]
    return {"exact_normalized": any(s["exact_normalized"] for s in scores),
            "answer_inclusion": any(s["answer_inclusion"] for s in scores),
            "alternatives": scores, "expected_answers_semantics": "alternative acceptable short answers, not a conjunction",
            "note": "Lexical diagnostics only; extra claims and substantive correctness require semantic review."}


def nll_row(row):
    result = deepcopy(row)
    meta = result.setdefault("metadata", {})
    reference = meta.get("reference_answer", meta.get("reference"))
    if not isinstance(reference, str) or not reference.strip():
        raise ValueError(f"Missing nonempty metadata.reference_answer/reference: {row.get('id')}")
    meta["reference"] = reference
    return result


def check_rows(rows):
    if not rows or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Evaluation needs nonempty, uniquely identified rows")
    for row in rows:
        generation_row(row)
        nll_row(row)
        if not isinstance(row.get("metadata", {}).get("task_type"), str):
            raise ValueError("Each evaluation row needs metadata.task_type")
        lexical_answers("", row["metadata"])


def adapter_proof(adapter, config):
    adapter = lab_path(adapter).resolve()
    proof_path = adapter.parent / "large_checkpoint.json"
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    if proof.get("status") != "saved" or Path(proof["adapter"]).resolve() != adapter:
        raise ValueError("Adapter provenance is absent or belongs to another path")
    hashes = {path.name: file_hash(path) for path in adapter.iterdir() if path.is_file()}
    if hashes != proof.get("adapter_files_sha256") or proof["base_model"] != config["model"]:
        raise ValueError("Saved adapter hashes/base differ from training provenance")
    if not proof.get("changed_tensors") or proof.get("initial_lora_sha256") == proof.get("lora_sha256"):
        raise ValueError("Adapter has no verified training update")
    return proof, {"path": str(proof_path), "sha256": file_hash(proof_path)}, hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--output", required=True, help="New output stem relative to finetuning")
    args = parser.parse_args()
    output = lab_path(args.output).resolve()
    relative = output.relative_to((LAB / "evaluation" / "reports").resolve())
    if not relative.parts or not relative.parts[0].startswith("large"):
        raise ValueError("Evaluation output must be inside evaluation/reports/large*/")
    paths = [output.with_suffix(suffix) for suffix in (".json", ".md")]
    paths.append(output.with_name(output.name + "_vram.csv"))
    if any(path.exists() for path in paths):
        raise FileExistsError("Evaluation artifacts exist; choose a new output stem")
    benchmark = lab_path(args.benchmark)
    if benchmark.name not in SPLITS:
        raise ValueError("Use the frozen evaluation_validation.jsonl, benchmark.jsonl, or recall_probes.jsonl")
    protocol_path = benchmark.parent / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if not str(protocol.get("status", "")).startswith("frozen"):
        raise ValueError("Evaluation protocol must be frozen")
    declared = protocol.get("artifact_sha256", {}).get(benchmark.name)
    if declared is not None and declared != file_hash(benchmark):
        raise ValueError("Frozen evaluation data changed")
    rows = read_jsonl(benchmark)
    check_rows(rows)
    config = checked_config(args.config, "lr5e-5")
    proof, proof_source, hashes = adapter_proof(args.adapter, config) if args.adapter else ({}, None, None)
    model, tokenizer = load_model(config, adapter=args.adapter)
    active = []
    if args.adapter:
        from peft import PeftModel
        if not isinstance(model, PeftModel) or not model.active_adapters:
            raise RuntimeError("Evaluation requires a freshly loaded active PEFT adapter")
        active = list(model.active_adapters)
    report = {"status": "running", "created_unix": time.time(), "base_model": config["model"],
              "arm": proof.get("arm", "baseline"), "step": proof.get("step", 0), "epoch": proof.get("epoch", 0),
              "split": SPLITS[benchmark.name], "adapter": str(lab_path(args.adapter).resolve()) if args.adapter else None,
              "adapter_reload_verified": bool(args.adapter), "active_adapters": active,
              "adapter_files_sha256": hashes, "checkpoint_provenance": proof_source,
              "checkpoint_validation": proof.get("validation"),
              "benchmark": str(benchmark.resolve()), "benchmark_sha256": file_hash(benchmark),
              "protocol_sha256": file_hash(protocol_path), "evaluator_sha256": file_hash(Path(__file__)),
              "code_sha256": {name: file_hash(LAB / name) for name in ("evaluate_large.py", "evaluate_diagnostics.py", "common.py", "train_large.py")},
              "tokenizer": tokenizer_audit(tokenizer), "generation": PROTOCOL_RULES["generation"],
              "exact_normalization": "NFKC; collapse whitespace; remove terminal .!?; casefold unless case_sensitive; preserve internal punctuation",
              "results": []}
    write_json(paths[0], report)
    with NvidiaMonitor(paths[2]) as monitor:
        for row in rows:
            prompt = render_chat(tokenizer, generation_row(row), generation=True)
            if len(tokenizer(prompt, add_special_tokens=False)["input_ids"]) > 1024:
                raise ValueError(f"Generation input exceeds the frozen sequence limit: {row['id']}")
            generated = raw_generate(model, tokenizer, row, 128)
            metrics = lexical_answers(generated["response"], row["metadata"])
            nll = teacher_forced_nll(model, tokenizer, nll_row(row), 1024)
            report["results"].append({"id": row["id"], "category": row.get("category"),
                                      "prompt": row.get("prompt"), "metadata": row["metadata"],
                                      **generated, "lexical": metrics, "teacher_forced": nll})
            write_json(paths[0], report)
            print(json.dumps({"id": row["id"], "response": generated["response"],
                              "lexical_exact": metrics["exact_normalized"]}, ensure_ascii=False), flush=True)
    report.update(status="complete", nvidia=monitor.summary(),
                  mean_teacher_forced_nll=sum(r["teacher_forced"]["total_nll"] for r in report["results"])
                  / sum(r["teacher_forced"]["target_tokens"] for r in report["results"]))
    write_json(paths[0], report)
    lines = [f"# Large experiment: {report['arm']}, step {report['step']}, {report['split']}", "",
             "Lexical/NLL probes do not supply semantic, honesty, or style ratings.", ""]
    for row in report["results"]:
        lines.extend([f"## {row['id']}", "", str(row["prompt"]), "", row["response"], "",
                      "```json", json.dumps({key: row[key] for key in ("raw_decoded", "finish_reason", "lexical")},
                                            ensure_ascii=False, indent=2), "```", ""])
    paths[1].write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
