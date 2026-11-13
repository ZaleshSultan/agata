"""Summarize diagnostic trajectories without conflating recall and semantic quality."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import re

from common import LAB, file_hash, lab_path, read_jsonl, write_json

GROUPS = ("exact", "paraphrase", "unknown", "format", "style", "general", "honesty")


def checkpoint_step(report):
    adapter = report.get("adapter")
    if not adapter:
        return 0
    match = re.search(r"(?:^|[\\/])checkpoint-(\d+)(?:[\\/]|$)", adapter)
    if not match or not report.get("adapter_reload_verified"):
        raise ValueError("A diagnostic adapter report must identify its freshly reloaded checkpoint step")
    return int(match.group(1))


def load_trajectory(reports_dir, benchmark):
    expected_hash = file_hash(benchmark)
    expected_rows = read_jsonl(benchmark)
    expected_ids = [row["id"] for row in expected_rows]
    found = {}
    for path in reports_dir.glob("*.json"):
        report = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(report, dict):
            continue
        if report.get("benchmark_sha256") != expected_hash or "results" not in report:
            continue
        if report.get("status") != "complete":
            continue
        step = checkpoint_step(report)
        if step in found:
            raise ValueError(f"Two complete reports represent step {step}; choose an unambiguous report directory")
        if [row["id"] for row in report["results"]] != expected_ids:
            raise ValueError("Evaluation IDs/order differ from the frozen benchmark")
        for expected, observed in zip(expected_rows, report["results"]):
            if observed.get("metadata") != expected.get("metadata") or observed.get("prompt") != expected.get("prompt"):
                raise ValueError("Evaluation reference metadata or question changed")
        found[step] = {"path": str(path.resolve()), "sha256": file_hash(path), "report": report}
    if not found:
        raise ValueError("No complete evaluation found for the requested diagnostic benchmark")
    reference = found[min(found)]["report"]
    for record in found.values():
        report = record["report"]
        for field in ("base_model", "generation", "exact_normalization"):
            if report.get(field) != reference.get(field):
                raise ValueError(f"Incomparable checkpoint evaluation: {field}")
        for field in ("chat_template_sha256", "special_token_ids"):
            if report["tokenizer"][field] != reference["tokenizer"][field]:
                raise ValueError(f"Tokenizer changed across checkpoints: {field}")
        for a, b in zip(reference["results"], report["results"]):
            if a["input_token_ids_sha256"] != b["input_token_ids_sha256"]:
                raise ValueError("Generation input token IDs differ across checkpoints")
    return found


def load_semantic_ratings(path, trajectory):
    if path is None:
        return {}, None
    document = json.loads(path.read_text(encoding="utf-8"))
    if not document.get("reviewer") or not document.get("review_method"):
        raise ValueError("Semantic review must identify its reviewer and method")
    lookup = {}
    for row in document["ratings"]:
        step, item_id = int(row["step"]), row["id"]
        if step not in trajectory or document["report_sha256_by_step"].get(str(step)) != trajectory[step]["sha256"]:
            raise ValueError("Semantic rating source report changed or is absent")
        if item_id not in {item["id"] for item in trajectory[step]["report"]["results"]}:
            raise ValueError("Semantic rating item is absent from the benchmark")
        if (step, item_id) in lookup:
            raise ValueError("Duplicate semantic rating")
        if type(row.get("score")) is not int or row["score"] not in (0, 1):
            raise ValueError("Semantic score must be integer 0/1")
        if not isinstance(row.get("reason"), str) or not row["reason"].strip() or not isinstance(row.get("unsupported_claim"), bool):
            raise ValueError("Semantic score needs a reason and an explicit unsupported-claim flag")
        if row["score"] and row["unsupported_claim"]:
            raise ValueError("Unsupported substantive claims cannot receive full semantic credit")
        lookup[(step, item_id)] = row
    return lookup, {"reviewer": document["reviewer"], "review_method": document["review_method"], "sha256": file_hash(path)}


def summarize_groups(report, step, semantic=None):
    semantic = semantic or {}
    grouped = defaultdict(list)
    for row in report["results"]:
        group = row["metadata"].get("task_type")
        if group not in GROUPS:
            raise ValueError(f"Unknown diagnostic task_type: {group}")
        grouped[group].append(row)
    summary = {}
    for group, rows in grouped.items():
        scored = [row for row in rows if row["lexical"]["exact_normalized"] is not None]
        reviewed = [semantic[(step, row["id"])] for row in rows if (step, row["id"]) in semantic]
        tokens = sum(row["teacher_forced"]["target_tokens"] for row in rows)
        summary[group] = {"n": len(rows), "lexically_scored": len(scored),
                          "exact_normalized_count": sum(row["lexical"]["exact_normalized"] for row in scored) if scored else None,
                          "answer_inclusion_count": sum(row["lexical"]["answer_inclusion"] for row in scored) if scored else None,
                          "mean_teacher_forced_nll": sum(row["teacher_forced"]["total_nll"] for row in rows)/tokens,
                          "teacher_target_tokens": tokens,
                          "token_limit_count": sum(row["hit_token_limit"] for row in rows),
                          "unclosed_think_count": sum(row["unclosed_think"] for row in rows),
                          "semantic_reviewed": len(reviewed), "semantic_missing": len(rows)-len(reviewed),
                          "semantic_correct": sum(row["score"] for row in reviewed) if reviewed else None,
                          "semantic_unsupported_claims": sum(row["unsupported_claim"] for row in reviewed) if reviewed else None}
    return summary


def plot_traces(run_dir, output, trajectory, groups):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    paths = {}
    state_path = run_dir / "trainer_state.json"
    if state_path.exists():
        history = json.loads(state_path.read_text(encoding="utf-8"))["log_history"]
        fig, axis = plt.subplots(figsize=(8, 4), constrained_layout=True)
        for field, label in (("loss", "Training loss (10-step logs)"), ("eval_loss", "Validation loss")):
            points = [(row["step"], row[field]) for row in history if field in row]
            if points:
                axis.plot(*zip(*points), marker="o", markersize=3, label=label)
        axis.set(xlabel="Optimizer step", ylabel="Assistant-target cross-entropy", title=f"{run_dir.name}: deliberate overfit trajectory")
        axis.legend()
        axis.grid(alpha=.2)
        path = output.with_name(output.name + "_loss.png")
        fig.savefig(path, dpi=170)
        plt.close(fig)
        paths["loss"] = str(path)
    csv_path = run_dir / "nvidia_vram.csv"
    if csv_path.exists():
        with csv_path.open(encoding="utf-8") as stream:
            samples = list(csv.DictReader(stream))
        values = [float(row["memory_used_mib"]) for row in samples if row.get("memory_used_mib")]
        fig, axis = plt.subplots(figsize=(8, 3.5), constrained_layout=True)
        axis.plot(range(len(values)), values)
        axis.set(xlabel="NVIDIA driver sample (approximately one per second)", ylabel="GPU memory used (MiB)", title=f"{run_dir.name}: complete training process")
        axis.grid(alpha=.2)
        path = output.with_name(output.name + "_vram.png")
        fig.savefig(path, dpi=170)
        plt.close(fig)
        paths["vram"] = str(path)
    fig, axis = plt.subplots(figsize=(8, 4), constrained_layout=True)
    for group in GROUPS:
        points = [(step, groups[step][group]["mean_teacher_forced_nll"]) for step in sorted(trajectory) if group in groups[step]]
        if points:
            axis.plot(*zip(*points), marker="o", markersize=4, label=group)
    axis.set(xlabel="Checkpoint step (0 = base)", ylabel="Reference assistant-token NLL", title="Frozen prompts/reference targets; groups remain separate")
    axis.legend(fontsize=8)
    axis.grid(alpha=.2)
    path = output.with_name(output.name + "_nll.png")
    fig.savefig(path, dpi=170)
    plt.close(fig)
    paths["teacher_forced_nll"] = str(path)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True, choices=("one_fact", "mini", "style"))
    parser.add_argument("--data-dir", default="datasets/diagnostics_v1")
    parser.add_argument("--reports-dir", default="evaluation/reports/diagnostics_v1")
    parser.add_argument("--runs-dir", default="outputs/diagnostics_v1")
    parser.add_argument("--ratings", help="Optional separate semantic ratings, keyed by checkpoint step and item ID")
    parser.add_argument("--output", help="New report stem; existing artifacts are never replaced")
    args = parser.parse_args()
    data_dir, reports_dir, runs_dir = map(lab_path, (args.data_dir, args.reports_dir, args.runs_dir))
    output = lab_path(args.output) if args.output else reports_dir / f"{args.variant}_comparison"
    paths = [output.with_suffix(suffix) for suffix in (".json", ".md")]
    paths += [output.with_name(output.name + suffix) for suffix in ("_examples.md", "_ratings_template.json", "_loss.png", "_vram.png", "_nll.png")]
    if any(path.exists() for path in paths):
        raise FileExistsError("Comparison output exists; choose a new --output stem to preserve it")
    output.parent.mkdir(parents=True, exist_ok=True)
    benchmark = data_dir / f"{args.variant}_benchmark.jsonl"
    trajectory = load_trajectory(reports_dir, benchmark)
    semantic, semantic_provenance = load_semantic_ratings(lab_path(args.ratings) if args.ratings else None, trajectory)
    groups = {step: summarize_groups(record["report"], step, semantic) for step, record in trajectory.items()}
    expected_steps = [0, 20, 50] if args.variant == "style" else [0, 20, 50, 100]
    missing = [step for step in expected_steps if step not in trajectory]
    training_path = runs_dir / args.variant / "training_report.json"
    training = json.loads(training_path.read_text(encoding="utf-8")) if training_path.exists() else None
    checkpoint_proofs = {}
    for step in sorted(trajectory):
        proof_path = runs_dir / args.variant / f"checkpoint-{step}" / "diagnostic_checkpoint.json"
        if step and proof_path.exists():
            proof = json.loads(proof_path.read_text(encoding="utf-8"))
            if Path(proof["adapter"]).resolve() != Path(trajectory[step]["report"]["adapter"]).resolve():
                raise ValueError("Training checkpoint proof points to another evaluated adapter")
            if proof["adapter_files_sha256"] != trajectory[step]["report"].get("adapter_files_sha256"):
                raise ValueError("Saved checkpoint adapter hashes differ from the freshly evaluated adapter")
            checkpoint_proofs[step] = proof
    plots = plot_traces(runs_dir / args.variant, output, trajectory, groups)
    result = {"status": "partial" if missing else "complete", "variant": args.variant,
              "benchmark_sha256": file_hash(benchmark), "expected_steps": expected_steps, "missing_steps": missing,
              "source_reports": {step: {k:v for k,v in record.items() if k != "report"} for step, record in trajectory.items()},
              "groups_by_step": groups, "semantic_review": semantic_provenance, "training": training,
              "checkpoint_proofs": checkpoint_proofs, "plots": plots,
              "one_fact_gate_by_step": {step: record["report"]["one_fact_gate"] for step, record in trajectory.items()},
              "limitations": ["A deliberate overfit diagnostic is not a quality estimate and is not comparable with the 32-step ABC budget.",
                              "Exact training questions and unseen paraphrases share the same supplied facts; unknown and format probes are separate.",
                              "Identifier/answer inclusion can occur inside a false or contradictory answer; semantic correctness requires review.",
                              "Style, general skills, honesty and unknown-answer quality are not inferred from exact-match or NLL values.",
                              "NLL uses reference assistant targets including template/empty-think/EOS tokens; low NLL is not a free-generation success.",
                              "One seed, tiny data, related examples; checkpoint inspection is diagnostic and does not establish a broadly superior model.",
                              "NVIDIA samples cover the entire GPU and can miss brief peaks."]}
    write_json(output.with_suffix(".json"), result)
    lines = [f"# Diagnostics: {args.variant}", "", f"Status: {result['status']}; missing evaluated steps: {missing}.", "",
             "Exact-match and inclusion are lexical probes. They are not semantic truth or style scores.", "",
             "| Step | Group | N | Exact target | Target included | Reference NLL | Semantic rated / correct | Token-limit exits |",
             "|---:|---|---:|---|---|---:|---|---:|"]
    for step in expected_steps:
        if step not in groups:
            lines.append(f"| {step} | not evaluated | — | — | — | — | — | — |")
            continue
        for group in GROUPS:
            if group not in groups[step]:
                continue
            item = groups[step][group]
            exact = "n/a" if item["exact_normalized_count"] is None else f"{item['exact_normalized_count']}/{item['lexically_scored']}"
            included = "n/a" if item["answer_inclusion_count"] is None else f"{item['answer_inclusion_count']}/{item['lexically_scored']}"
            reviewed = "pending" if not item["semantic_reviewed"] else f"{item['semantic_reviewed']}/{item['n']} rated; {item['semantic_correct']} correct"
            lines.append(f"| {step} | {group} | {item['n']} | {exact} | {included} | {item['mean_teacher_forced_nll']:.4f} | {reviewed} | {item['token_limit_count']} |")
    lines += ["", "## Actual training exposure", "", "| Step | Examples consumed | Assistant target tokens consumed | Changed LoRA tensors |", "|---:|---:|---:|---:|"]
    for step, proof in checkpoint_proofs.items():
        consumed = proof["consumed"]
        lines.append(f"| {step} | {consumed['example_presentations']} | {consumed['assistant_target_tokens']} | {proof['changed_tensors']} |")
    lines += ["", "Counts come from completed training microsteps, not max_steps × nominal batch size. Target tokens include the training assistant template and EOS.", "", "## Plots", ""]
    for name, path in plots.items():
        lines.append(f"![{name}]({Path(path).name})")
    lines += ["", "## Limits", ""] + ["- " + item for item in result["limitations"]]
    output.with_suffix(".md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    examples = [f"# {args.variant}: every frozen question, all available checkpoints", "", "No answer is omitted or selected by success.", ""]
    template = {"reviewer": "", "review_method": "", "report_sha256_by_step": {str(step): record["sha256"] for step,record in trajectory.items()}, "ratings": []}
    for index, item in enumerate(next(iter(trajectory.values()))["report"]["results"]):
        examples += [f"## {item['id']} · {item['metadata']['task_type']}", "", str(item["prompt"]), "",
                     "Reference: " + item["metadata"]["reference"], "", "Rubric: " + json.dumps(item["metadata"].get("rubric"), ensure_ascii=False), ""]
        for step in sorted(trajectory):
            row = trajectory[step]["report"]["results"][index]
            examples += [f"### Step {step}", "", row["response"], "", f"Finish: {row['finish_reason']}; generated tokens: {row['generated_tokens']}.", "",
                         "Raw decoded:", "```text", row["raw_decoded"], "```", ""]
            template["ratings"].append({"step": step, "id": row["id"], "score": None, "reason": "", "unsupported_claim": None})
    output.with_name(output.name+"_examples.md").write_text("\n".join(examples), encoding="utf-8")
    write_json(output.with_name(output.name+"_ratings_template.json"), template)
    print(json.dumps({"status": result["status"], "report": str(output.with_suffix('.md')), "missing_steps": missing}, ensure_ascii=False))


if __name__ == "__main__":
    main()
