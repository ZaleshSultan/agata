"""Compare frozen large-v3 runs; select only with validation loss and semantic guards."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path

from common import LAB, file_hash, lab_path, read_jsonl, write_json
from evaluate_large import SPLITS
from train_large import ARMS, PROTOCOL_RULES


def report_id(report):
    return f"{report['split']}:{report['arm']}:{report['step']}"


def load_reports(directory, data_dir):
    expected = {split: {"hash": file_hash(data_dir / name), "rows": read_jsonl(data_dir / name)}
                for name, split in SPLITS.items()}
    found = {}
    protocol_hash = file_hash(data_dir / "protocol.json")
    for path in directory.glob("*.json"):
        report = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(report, dict) or report.get("status") != "complete" or "results" not in report:
            continue
        split = report.get("split")
        if split not in expected:
            continue
        if report.get("benchmark_sha256") != expected[split]["hash"] or report.get("protocol_sha256") != protocol_hash:
            raise ValueError("Evaluation data/protocol differs from the requested frozen experiment")
        if report.get("arm") not in ("baseline", *ARMS) or type(report.get("step")) is not int:
            raise ValueError("Unknown evaluation arm or optimizer step")
        if report["arm"] == "baseline":
            if report.get("adapter") or report["step"] != 0:
                raise ValueError("Baseline cannot identify an adapter or nonzero step")
        elif not report.get("adapter_reload_verified") or report["step"] <= 0:
            raise ValueError("Adapter evaluation must prove a fresh checkpoint reload")
        key = report_id(report)
        if key in found:
            raise ValueError(f"Duplicate complete evaluation: {key}")
        rows = expected[split]["rows"]
        if [r["id"] for r in report["results"]] != [r["id"] for r in rows]:
            raise ValueError("Evaluation IDs/order differ from the frozen split")
        for row, actual in zip(rows, report["results"]):
            if actual.get("metadata") != row.get("metadata") or actual.get("prompt") != row.get("prompt"):
                raise ValueError("Frozen evaluation question or rubric changed")
        if report.get("generation") != PROTOCOL_RULES["generation"]:
            raise ValueError("Inference differs from the frozen generation settings")
        found[key] = {"path": str(path.resolve()), "sha256": file_hash(path), "report": report}
    if not found:
        raise ValueError("No complete large-experiment evaluations found")
    reference = next(iter(found.values()))["report"]
    by_split = {}
    for entry in found.values():
        report = entry["report"]
        for field in ("base_model", "exact_normalization", "evaluator_sha256", "code_sha256"):
            if report.get(field) != reference.get(field):
                raise ValueError(f"Incomparable evaluation field: {field}")
        for field in ("chat_template_sha256", "special_token_ids"):
            if report["tokenizer"][field] != reference["tokenizer"][field]:
                raise ValueError(f"Tokenizer mismatch: {field}")
        old = by_split.setdefault(report["split"], report)
        if any(a["input_token_ids_sha256"] != b["input_token_ids_sha256"]
               for a, b in zip(old["results"], report["results"])):
            raise ValueError("Prompt tokens differ across comparable evaluations")
    return found


def load_ratings(path, reports):
    if path is None:
        return {}, None
    document = json.loads(path.read_text(encoding="utf-8"))
    for field in ("reviewer", "review_method"):
        if not isinstance(document.get(field), str) or not document[field].strip():
            raise ValueError("Ratings require a named reviewer and nonempty review method")
    ratings = {}
    for row in document["ratings"]:
        key, item_id = row["report_id"], row["id"]
        if key not in reports or document["report_sha256"].get(key) != reports[key]["sha256"]:
            raise ValueError("Semantic ratings must bind to the exact source report hash")
        if item_id not in {r["id"] for r in reports[key]["report"]["results"]}:
            raise ValueError("Semantic rating refers to an unknown item")
        if (key, item_id) in ratings:
            raise ValueError("Duplicate semantic rating")
        if type(row.get("score")) is not int or row["score"] not in (0, 1):
            raise ValueError("Semantic score must be integer 0/1")
        if (not isinstance(row.get("reason"), str) or not row["reason"].strip()
                or not isinstance(row.get("unsupported_claim"), bool)):
            raise ValueError("A reason and explicit unsupported-claim boolean are required")
        if row["score"] and row["unsupported_claim"]:
            raise ValueError("A substantive unsupported claim cannot receive a pass")
        ratings[(key, item_id)] = row
    return ratings, {"reviewer": document["reviewer"], "review_method": document["review_method"], "sha256": file_hash(path)}


def summarize(report, ratings):
    grouped = defaultdict(list)
    key = report_id(report)
    for row in report["results"]:
        grouped[row["metadata"]["task_type"]].append(row)
    groups = {}
    for group, rows in grouped.items():
        lexical = [r["lexical"] for r in rows if r["lexical"]["exact_normalized"] is not None]
        semantic = [ratings[(key, r["id"])] for r in rows if (key, r["id"]) in ratings]
        tokens = sum(r["teacher_forced"]["target_tokens"] for r in rows)
        groups[group] = {"n": len(rows), "lexical_scored": len(lexical),
                         "lexical_exact": sum(r["exact_normalized"] for r in lexical) if lexical else None,
                         "lexical_inclusion": sum(r["answer_inclusion"] for r in lexical) if lexical else None,
                         "semantic_reviewed": len(semantic), "semantic_correct": sum(r["score"] for r in semantic),
                         "unsupported_claims": sum(r["unsupported_claim"] for r in semantic),
                         "teacher_target_tokens": tokens,
                         "teacher_nll": sum(r["teacher_forced"]["total_nll"] for r in rows)/tokens,
                         "length_exits": sum(r["hit_token_limit"] for r in rows),
                         "unclosed_think": sum(r["unclosed_think"] for r in rows)}
    return groups


def validation_guard(baseline, candidate, ratings):
    if baseline.get("split") != "validation" or candidate.get("split") != "validation":
        raise ValueError("Checkpoint acceptance cannot use test or recall responses")
    if [r["id"] for r in baseline["results"]] != [r["id"] for r in candidate["results"]]:
        raise ValueError("Validation guard requires identical items")
    keys = (report_id(baseline), report_id(candidate))
    for key, report in zip(keys, (baseline, candidate)):
        if any((key, row["id"]) not in ratings for row in report["results"]):
            return {"status": "pending_complete_validation_ratings", "passed": False, "groups": {}}
    summaries = {}
    for name, task_types in PROTOCOL_RULES["guard_groups"].items():
        ids = [row["id"] for row in baseline["results"] if row["metadata"]["task_type"] in task_types]
        if not ids:
            raise ValueError(f"Frozen validation is missing required guard group: {name}")
        base = [ratings[(keys[0], item)] for item in ids]
        cand = [ratings[(keys[1], item)] for item in ids]
        correct0, correct1 = sum(r["score"] for r in base), sum(r["score"] for r in cand)
        unsupported0, unsupported1 = sum(r["unsupported_claim"] for r in base), sum(r["unsupported_claim"] for r in cand)
        summaries[name] = {"n": len(ids), "baseline_correct": correct0, "candidate_correct": correct1,
                           "baseline_unsupported": unsupported0, "candidate_unsupported": unsupported1,
                           "passed": correct1 >= correct0 and unsupported1 <= unsupported0}
    return {"status": "reviewed", "passed": all(group["passed"] for group in summaries.values()), "groups": summaries}


def select_checkpoint(candidates, reports, ratings):
    """The API deliberately accepts validation reports/losses, never test metrics."""
    baseline_key = "validation:baseline:0"
    if baseline_key not in reports:
        return {"status": "pending_baseline_validation", "recommended": None, "candidates": []}
    baseline = reports[baseline_key]["report"]
    decisions, pending = [], False
    for candidate in candidates:
        key = f"validation:{candidate['arm']}:{candidate['step']}"
        decision = dict(candidate)
        if candidate.get("catastrophic") or not candidate.get("run_completed", True):
            decision.update(admissible=False, guard={"status": "excluded_training_failure", "passed": False})
        elif key not in reports:
            decision.update(admissible=False, guard={"status": "pending_validation_generation", "passed": False})
            pending = True
        else:
            guard = validation_guard(baseline, reports[key]["report"], ratings)
            decision.update(admissible=guard["passed"], guard=guard)
            pending |= guard["status"].startswith("pending")
        decisions.append(decision)
    if pending or not candidates:
        return {"status": "pending_validation_review", "recommended": None, "candidates": decisions}
    eligible = [c for c in decisions if c["admissible"]]
    if not eligible:
        return {"status": "complete", "recommended": "baseline", "candidates": decisions,
                "reason": "No adapter satisfies all frozen validation acceptance guards"}
    best = min(eligible, key=lambda c: (c["validation_loss"], c["step"], ARMS[c["arm"]]))
    return {"status": "complete", "recommended": f"{best['arm']}:{best['step']}", "candidates": decisions,
            "reason": "Minimum training-validation loss among accepted adapters; ties use smaller step then smaller LR"}


def evaluation_completeness(selection, reports, ratings, candidates):
    """Selection can finish before test is inspected; a final outcome report cannot."""
    required = {f"{split}:baseline:0" for split in SPLITS.values()}
    # Every saved checkpoint is an outcome to report, including guard failures.
    # These held-out results never enter select_checkpoint.
    for candidate in candidates:
        required.update(f"{split}:{candidate['arm']}:{candidate['step']}" for split in SPLITS.values())
    missing_reports = sorted(required - reports.keys())
    missing_ratings = [{"report_id": key, "id": row["id"]}
                       for key, entry in reports.items() for row in entry["report"]["results"]
                       if (key, row["id"]) not in ratings]
    return {"complete": selection["status"] == "complete" and not missing_reports and not missing_ratings,
            "required_outcome_reports": sorted(required), "missing_reports": missing_reports,
            "missing_semantic_rating_count": len(missing_ratings), "missing_semantic_ratings": missing_ratings,
            "note": "Final outcome completeness is separate from validation-only model selection; test cannot change selection."}


def load_training(runs_dir, reports):
    training, candidates = {}, []
    for arm in ARMS:
        path = runs_dir / arm / "training_report.json"
        if not path.exists():
            continue
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("arm") != arm or report.get("rules") != PROTOCOL_RULES:
            raise ValueError("Training arm or frozen stopping/selection rules differ")
        for checkpoint in report.get("checkpoints", []):
            proof_path = Path(checkpoint["adapter"]).parent / "large_checkpoint.json"
            proof = json.loads(proof_path.read_text(encoding="utf-8"))
            if (proof["arm"] != arm or proof["step"] != checkpoint["step"]
                    or proof["validation"]["validation_loss"] != checkpoint["validation_loss"]):
                raise ValueError("Training checkpoint and validation provenance disagree")
            if not math.isfinite(checkpoint["validation_loss"]):
                raise ValueError("Nonfinite checkpoint validation loss")
            for entry in reports.values():
                evaluated = entry["report"]
                if evaluated["arm"] == arm and evaluated["step"] == checkpoint["step"]:
                    if evaluated.get("adapter_files_sha256") != proof["adapter_files_sha256"]:
                        raise ValueError("Evaluated adapter differs from training checkpoint")
                    if evaluated.get("checkpoint_validation") != proof["validation"]:
                        raise ValueError("Evaluated validation provenance differs from saved checkpoint")
            candidates.append({**checkpoint, "arm": arm,
                               "run_completed": report["status"] in ("complete", "stopped_catastrophic")})
        training[arm] = {"path": str(path.resolve()), "sha256": file_hash(path), "report": report}
    if len(training) == 2:
        a, b = [training[arm]["report"] for arm in ARMS]
        if a.get("initial_lora_sha256") != b.get("initial_lora_sha256"):
            raise ValueError("LR arms did not start with identical LoRA initialization")
        if a.get("data_sha256") != b.get("data_sha256"):
            raise ValueError("Training data/protocol differs between LR arms")
    return training, candidates


def plot_training(training, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    artifacts = {}
    for arm, entry in training.items():
        directory = Path(entry["path"]).parent
        state_path = directory / "trainer_state.json"
        if state_path.exists():
            history = json.loads(state_path.read_text(encoding="utf-8"))["log_history"]
            fig, axis = plt.subplots(figsize=(8, 4), constrained_layout=True)
            for field, label in (("loss", "Training logs"), ("eval_loss", "Epoch validation")):
                points = [(r["step"], r[field]) for r in history if field in r]
                if field == "eval_loss":
                    points.insert(0, (0, entry["report"]["baseline_validation"]["baseline_loss"]))
                if points:
                    axis.plot(*zip(*points), marker="o", markersize=3, label=label)
            axis.set(xlabel="Optimizer step", ylabel="Assistant-target loss", title=f"{arm}: same validation at baseline and epoch ends")
            axis.grid(alpha=.2)
            axis.legend()
            path = output.with_name(output.name + f"_{arm}_loss.png")
            fig.savefig(path, dpi=170)
            plt.close(fig)
            artifacts[f"{arm}_loss"] = str(path)
        csv_path = directory / "nvidia_vram.csv"
        if csv_path.exists():
            with csv_path.open(encoding="utf-8") as stream:
                values = [float(r["memory_used_mib"]) for r in csv.DictReader(stream) if r.get("memory_used_mib")]
            fig, axis = plt.subplots(figsize=(8, 3.5), constrained_layout=True)
            axis.plot(range(len(values)), values)
            axis.set(xlabel="NVIDIA sample (approximately one per second)", ylabel="GPU memory (MiB)", title=arm)
            axis.grid(alpha=.2)
            path = output.with_name(output.name + f"_{arm}_vram.png")
            fig.savefig(path, dpi=170)
            plt.close(fig)
            artifacts[f"{arm}_vram"] = str(path)
    return artifacts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="datasets/large_v3")
    parser.add_argument("--reports-dir", default="evaluation/reports/large_v3")
    parser.add_argument("--runs-dir", default="outputs/large_v3")
    parser.add_argument("--ratings")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = lab_path(args.output).resolve()
    relative = output.relative_to((LAB / "evaluation" / "reports").resolve())
    if not relative.parts or not relative.parts[0].startswith("large"):
        raise ValueError("Report output must remain inside evaluation/reports/large*/")
    paths = [output.with_suffix(suffix) for suffix in (".json", ".md")]
    paths += [output.with_name(output.name + suffix) for suffix in ("_examples.md", "_ratings_template.json")]
    paths += [output.with_name(output.name + f"_{arm}_{kind}.png") for arm in ARMS for kind in ("loss", "vram")]
    if any(path.exists() for path in paths):
        raise FileExistsError("Report artifacts already exist; choose a fresh stem")
    output.parent.mkdir(parents=True, exist_ok=True)
    reports = load_reports(lab_path(args.reports_dir), lab_path(args.data_dir))
    ratings, provenance = load_ratings(lab_path(args.ratings) if args.ratings else None, reports)
    training, candidates = load_training(lab_path(args.runs_dir), reports)
    groups = {key: summarize(entry["report"], ratings) for key, entry in reports.items()}
    selection = select_checkpoint(candidates, reports, ratings)
    missing_arms = [arm for arm in ARMS if arm not in training]
    if missing_arms:
        selection.update(status="pending_training_arms", recommended=None, missing_arms=missing_arms)
    completeness = evaluation_completeness(selection, reports, ratings, candidates)
    plots = plot_training(training, output)
    result = {"status": "complete" if completeness["complete"] else "partial",
              "selection": selection, "rules": PROTOCOL_RULES, "groups_by_report": groups,
              "evaluation_completeness": completeness,
              "source_reports": {key: {k:v for k,v in value.items() if k != "report"} for key,value in reports.items()},
              "semantic_review": provenance, "training": training, "plots": plots,
              "limitations": ["Selection uses validation only; held-out test and recall never select a checkpoint.",
                              "All validation responses require semantic ratings before acceptance; lexical inclusion is not truth.",
                              "Exact/recalled questions, new paraphrases, unknown facts and general skills remain separate.",
                              "The fixed maximum is three epochs; early stopping may produce unequal actual exposures.",
                              "A single seed and finite related questions do not establish broad or causal superiority.",
                              "NLL is teacher-forced reference likelihood; it does not replace free-generation quality.",
                              "NVIDIA samples cover the whole GPU and can miss brief peaks."]}
    write_json(output.with_suffix(".json"), result)
    template = {"reviewer": "", "review_method": "", "report_sha256": {key:r["sha256"] for key,r in reports.items()},
                "ratings": [{"report_id": key, "id": row["id"], "score": None, "reason": "", "unsupported_claim": None}
                            for key,entry in reports.items() for row in entry["report"]["results"]]}
    write_json(output.with_name(output.name + "_ratings_template.json"), template)
    lines = ["# Large-v3 comparison", "", f"Selection status: {selection['status']}; recommendation: {selection['recommended']}.", "",
             "Validation selects; test and recall describe performance. Semantic review is distinct from lexical probes.", "",
             "| Split / arm / step | Task | N | Exact / scored | Inclusion | Semantic correct / reviewed | Unsupported | Reference NLL |",
             "|---|---|---:|---|---:|---|---:|---:|"]
    examples = ["# Complete before/after responses", ""]
    for key in sorted(reports):
        for group, metrics in groups[key].items():
            lines.append(f"| {key} | {group} | {metrics['n']} | {metrics['lexical_exact']} / {metrics['lexical_scored']} | {metrics['lexical_inclusion']} | {metrics['semantic_correct']} / {metrics['semantic_reviewed']} | {metrics['unsupported_claims']} | {metrics['teacher_nll']:.6f} |")
        for row in reports[key]["report"]["results"]:
            examples.extend([f"## {key} / {row['id']}", "", str(row["prompt"]), "", row["response"], "",
                             "```json", json.dumps({"raw_decoded": row["raw_decoded"], "lexical": row["lexical"],
                                                    "semantic": ratings.get((key, row["id"])), "finish_reason": row["finish_reason"]},
                                                   ensure_ascii=False, indent=2), "```", ""])
    lines.extend(["", "## Validation selection", "", "```json", json.dumps(selection, ensure_ascii=False, indent=2), "```", ""])
    for key, path in plots.items():
        lines.extend([f"![{key}]({Path(path).name})", ""])
    lines.extend(["## Limits", "", *[f"- {note}" for note in result["limitations"]]])
    output.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")
    output.with_name(output.name + "_examples.md").write_text("\n".join(examples), encoding="utf-8")


if __name__ == "__main__":
    main()
