"""Frozen, sequential local GPU runs for the controlled AGATA A/B/C comparison.

Preparation and token-budget validation must complete before this entry point.
Paths are relative to finetuning. Existing successful actions are validated and
skipped; incomplete actions are never silently resumed or overwritten.
"""
from __future__ import annotations

import argparse
from collections import Counter
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

from common import LAB, file_hash, lab_path, read_config, read_jsonl, write_json

MODELS = ("A", "B", "C")
STEPS = 32
MAX_NEW_TOKENS = 192
PACKAGES = ("torch", "torchvision", "unsloth", "unsloth_zoo", "transformers", "trl",
            "peft", "datasets", "bitsandbytes", "triton-windows", "huggingface-hub",
            "safetensors", "numpy", "matplotlib")


def versions():
    installed = {}
    for name in PACKAGES:
        try:
            installed[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            installed[name] = None
    return {"python": platform.python_version(), "executable": sys.executable,
            "platform": platform.platform(), "packages": installed}


def make_manifest(data_dir, runs_dir, reports_dir):
    protocol = json.loads((data_dir / "protocol.json").read_text(encoding="utf-8"))
    sealed = json.loads((data_dir / "protocol.sha256.json").read_text(encoding="utf-8"))
    if sealed.get("protocol.json") != file_hash(data_dir / "protocol.json"):
        raise ValueError("Prepared protocol seal differs")
    if protocol.get("status") != "frozen_before_model_outputs" or protocol.get("max_steps") != STEPS or protocol.get("seed") != 3407:
        raise ValueError("A completed, frozen quality/token audit for the fixed budget is required")
    for field in ("exact_train_validation_overlap", "exact_train_primary_overlap", "persona_scenario_family_overlap", "held_chapter_overlap"):
        if protocol["quality_audit"].get(field) != 0:
            raise ValueError(f"Dataset leakage audit failed or missing: {field}")
    for relative, expected in protocol["artifact_sha256"].items():
        if file_hash(data_dir / relative) != expected:
            raise ValueError(f"Prepared artifact changed after quality review: {relative}")
    actual_groups = Counter(row.get("metadata", {}).get("primary_group") for row in read_jsonl(data_dir / "benchmark.jsonl"))
    actual_groups.pop(None, None)
    if dict(actual_groups) != protocol["benchmark"]["primary_counts"]:
        raise ValueError("Actual primary benchmark domains differ from the preregistered counts")
    configs = {name: read_config(data_dir / "configs" / f"{name}.yaml") for name in MODELS}
    expected_training = {"batch_size": 1, "gradient_accumulation_steps": 4,
                         "learning_rate": 0.0002, "seed": 3407}
    reference = configs["A"]
    for name, config in configs.items():
        for section in ("model", "lora", "training", "evaluation"):
            if config[section] != reference[section]:
                raise ValueError(f"Controlled configuration differs: {name}.{section}")
        if any(config["training"].get(k) != v for k, v in expected_training.items()):
            raise ValueError(f"Unexpected training budget/seed in {name}")
        if config["model"]["max_seq_length"] != 1024 or config["lora"]["r"] != 16:
            raise ValueError("Protocol requires sequence length 1024 and LoRA rank 16")
        if not config["model"].get("revision") or config["model"]["revision"] == "main":
            raise ValueError("The base model must be pinned to an immutable revision")
        if config["evaluation"]["max_new_tokens"] != MAX_NEW_TOKENS:
            raise ValueError("Protocol requires the same fixed 192-token generation budget")
        for split in ("train", "validation"):
            rows = read_jsonl(lab_path(config["data"][split]))
            audit = protocol["variants"][name][split]
            if (len(rows) != audit["rows"] or audit["overlength_rows"] != 0 or
                    not 0 < audit["max_sequence_tokens"] <= config["model"]["max_seq_length"] or
                    audit["supervised_tokens"] <= 0):
                raise ValueError(f"Missing or failed tokenizer quality check: {name} {split}")
            if not rows:
                raise ValueError(f"Empty {name} {split} data")
    files = {data_dir / "protocol.json", data_dir / "benchmark.jsonl", data_dir / "memorization.jsonl"}
    for name, config in configs.items():
        files.add(data_dir / "configs" / f"{name}.yaml")
        files.update(lab_path(config["data"][split]) for split in ("train", "validation"))
    # Freeze all preparation evidence, including manifests and token checks, if
    # present. Model outputs live in other directories and cannot affect this.
    files.update(path for path in data_dir.rglob("*") if path.is_file())
    files.update(LAB / name for name in ("common.py", "train.py", "evaluate.py", "run_abc.py"))
    frozen = {str(path.resolve()): file_hash(path) for path in sorted(files)}
    return {"schema_version": 1, "data_dir": str(data_dir.resolve()),
            "runs_dir": str(runs_dir.resolve()), "reports_dir": str(reports_dir.resolve()),
            "base_model": reference["model"], "lora": reference["lora"],
            "training": reference["training"], "optimizer_steps": STEPS,
            "sample_presentations_nominal": STEPS * 4,
            "generation": {"do_sample": False, "max_new_tokens": MAX_NEW_TOKENS, "enable_thinking": False},
            "config_paths": {name: str((data_dir / "configs" / f"{name}.yaml").resolve()) for name in MODELS},
            "configs": configs, "files_sha256": frozen, "runtime": versions(),
            "prepared_protocol_sha256": file_hash(data_dir / "protocol.json"),
            "prepared_token_budgets": protocol["variants"], "benchmark_design": protocol["benchmark"],
            "design_limitations": ["One seed and short runs; no statistical superiority claim.",
                                   "Equal optimizer steps do not imply equal supervised token exposure.",
                                   "The mixed run has lower per-domain exposure than a pure run.",
                                   "Validation is diagnostic; no early stopping or hyperparameter selection."]}


def freeze_manifest(path, candidate):
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        comparable = {key: value for key, value in previous.items() if key != "frozen_unix"}
        if comparable != candidate:
            raise ValueError("Frozen protocol, data, code, runtime, or destinations changed. Use a new experiment version.")
        return previous
    candidate = dict(candidate, frozen_unix=time.time())
    write_json(path, candidate)
    return candidate


def verify_training(path, config, steps=STEPS):
    report = json.loads(path.read_text(encoding="utf-8"))
    if (report.get("status") != "passed" or report.get("smoke") or
            report.get("starting_global_step") != 0 or report.get("global_step") != steps or
            report.get("resumed_from") or report.get("config") != config or
            not report.get("lora_update", {}).get("changed_tensors")):
        raise ValueError(f"Existing training is incomplete or incompatible: {path}")
    if report["dataset"]["sha256"] != file_hash(lab_path(config["data"]["train"])):
        raise ValueError("Training dataset hash changed")
    adapter = Path(report["adapter"])
    if not (adapter / "adapter_model.safetensors").is_file():
        raise ValueError("Completed training report has no saved adapter")
    if not report.get("initial_lora_sha256") or report.get("initial_lora_sha256") == report.get("final_lora_sha256"):
        raise ValueError("Fresh initialization and changed final LoRA state must have distinct recorded hashes")
    return report


def verify_evaluation(path, benchmark, config, adapter):
    report = json.loads(path.read_text(encoding="utf-8"))
    expected_adapter = str(adapter) if adapter else None
    if (report.get("status") != "complete" or report.get("limit") is not None or
            report.get("benchmark_sha256") != file_hash(benchmark) or
            report.get("base_model") != config["model"] or report.get("adapter") != expected_adapter or
            report.get("generation") != {"do_sample": False, "max_new_tokens": MAX_NEW_TOKENS, "enable_thinking": False}):
        raise ValueError(f"Existing evaluation is incomplete or incompatible: {path}")
    if [r["id"] for r in report["results"]] != [r["id"] for r in read_jsonl(benchmark)]:
        raise ValueError("Benchmark answer IDs or order differ")
    if adapter and not report.get("adapter_reload_verified"):
        raise ValueError("Evaluation did not verify fresh PEFT adapter loading")
    return report


def execute(argv, log_path, journal):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    event = {"command": argv, "log": str(log_path), "started_unix": time.time()}
    print("RUN " + " ".join(argv), flush=True)
    environment = dict(os.environ, PYTHONUTF8="1", PYTHONUNBUFFERED="1")
    with log_path.open("x", encoding="utf-8") as stream:
        result = subprocess.run(argv, cwd=LAB.parent, env=environment, stdout=stream, stderr=subprocess.STDOUT)
    event.update(returncode=result.returncode, ended_unix=time.time())
    with journal.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, ensure_ascii=False) + "\n")
    if result.returncode:
        raise RuntimeError(f"Action failed ({result.returncode}); inspect {log_path}. No automatic retry or protocol change.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("preflight", "baseline", "train", "evaluate", "all"), default="all")
    parser.add_argument("--data-dir", default="datasets/abc_v1")
    parser.add_argument("--runs-dir", default="outputs/abc_v1")
    parser.add_argument("--reports-dir", default="evaluation/reports/abc_v1")
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--smoke-report", default="outputs/smoke/training_report.json")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and print frozen manifest without writing or GPU use")
    args = parser.parse_args()
    data_dir, runs_dir, reports_dir = map(lab_path, (args.data_dir, args.runs_dir, args.reports_dir))
    candidate = make_manifest(data_dir, runs_dir, reports_dir)
    if args.dry_run:
        print(json.dumps(candidate, ensure_ascii=False, indent=2))
        return
    reports_dir.mkdir(parents=True, exist_ok=True)
    frozen = freeze_manifest(reports_dir / "run_manifest.json", candidate)
    journal = reports_dir / "actions.jsonl"
    configs = frozen["configs"]
    log_dir = LAB / "logs" / runs_dir.name

    def benchmark(name, suffix, benchmark_path):
        config_name = "A" if name == "baseline" else name
        adapter = None if name == "baseline" else (runs_dir / name / "adapter").resolve()
        output = reports_dir / (name + suffix)
        report_path = output.with_suffix(".json")
        if report_path.exists():
            verify_evaluation(report_path, benchmark_path, configs[config_name], adapter)
            print(f"SKIP verified complete evaluation: {report_path}", flush=True)
            return
        argv = [sys.executable, str(LAB / "evaluate.py"), "--config", frozen["config_paths"][config_name],
                "--benchmark", str(benchmark_path), "--output", str(output), "--max-new-tokens", str(MAX_NEW_TOKENS)]
        if adapter:
            argv.extend(("--adapter", str(adapter)))
        execute(argv, log_dir / f"{name}{suffix}_evaluate.log", journal)
        verify_evaluation(report_path, benchmark_path, configs[config_name], adapter)

    def evaluate(name):
        benchmark(name, "", data_dir / "benchmark.jsonl")
        benchmark(name, "_memorization", data_dir / "memorization.jsonl")

    def train(name):
        output = runs_dir / name
        report_path = output / "training_report.json"
        if report_path.exists():
            verify_training(report_path, configs[name])
            print(f"SKIP verified complete training: {report_path}", flush=True)
            return
        if output.exists() and any(output.iterdir()):
            raise ValueError(f"Incomplete nonempty run directory: {output}")
        execute([sys.executable, str(LAB / "train.py"), "--config", frozen["config_paths"][name],
                 "--max-steps", str(STEPS), "--output-dir", str(output), "--smoke-report", args.smoke_report],
                log_dir / f"{name}_train.log", journal)
        verify_training(report_path, configs[name])

    def verify_independent_initialization():
        hashes = {}
        for name in MODELS:
            report_path = runs_dir / name / "training_report.json"
            if report_path.is_file():
                hashes[name] = verify_training(report_path, configs[name])["initial_lora_sha256"]
        if len(set(hashes.values())) > 1:
            raise ValueError("A/B/C did not begin with identical freshly initialized LoRA weights")

    if args.stage in ("baseline", "all"):
        evaluate("baseline")
    for name in args.models:
        if args.stage in ("train", "all"):
            train(name)
            verify_independent_initialization()
        if args.stage in ("evaluate", "all"):
            verify_training(runs_dir / name / "training_report.json", configs[name])
            evaluate(name)
    print(f"Stage {args.stage} completed; frozen manifest: {reports_dir / 'run_manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
