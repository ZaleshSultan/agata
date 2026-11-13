"""Deliberate local overfit diagnostics; never resumes or alters ABC artifacts."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import inspect
import json
import math
from pathlib import Path
import sys
import time
import traceback

from common import (LAB, LabelPreservingCollator, NvidiaMonitor, encode_assistant_only,
                    file_hash, lab_path, load_model, read_config, read_jsonl, write_json)
from run_abc import versions
from train import trainable_state_hash
from evaluate_diagnostics import tokenizer_audit, token_hash


def variant_settings(variant):
    if variant not in ("one_fact", "mini", "style"):
        raise ValueError("Unknown diagnostic variant")
    return {"max_steps": 50 if variant == "style" else 100,
            "milestones": [20, 50] if variant == "style" else [20, 50, 100],
            "learning_rate": 1e-4 if variant == "style" else 2e-4,
            "lr_scheduler_type": "constant_with_warmup", "warmup_steps": 2,
            "batch_size": 1, "gradient_accumulation_steps": 4, "seed": 3407}


def safe_output(value):
    output = lab_path(value).resolve()
    relative = output.relative_to((LAB / "outputs").resolve())
    if not relative.parts or not relative.parts[0].startswith("diagnostic"):
        raise ValueError("Diagnostic outputs must be inside outputs/diagnostic*/; ABC outputs are protected")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Preserving existing diagnostic run: {output}")
    return output


def verify_one_fact_gate(path, data_dir):
    if not path:
        raise ValueError("mini requires --gate-report from a successful one_fact exact probe")
    report = json.loads(lab_path(path).read_text(encoding="utf-8"))
    if (report.get("status") != "complete" or not report.get("adapter") or
            report.get("benchmark_sha256") != file_hash(data_dir / "one_fact_benchmark.jsonl") or
            report.get("one_fact_gate", {}).get("passed") is not True or
            not report.get("adapter_reload_verified")):
        raise ValueError("one_fact gate must pass in a complete fresh-adapter evaluation of the frozen one_fact benchmark")
    return {"report": str(lab_path(path).resolve()), "sha256": file_hash(lab_path(path)), "gate": report["one_fact_gate"]}


def save_adapter(model, tokenizer, destination, config):
    destination.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(str(destination), safe_serialization=True)
    tokenizer.save_pretrained(str(destination))
    path = destination / "adapter_config.json"
    saved = json.loads(path.read_text(encoding="utf-8"))
    saved.update(base_model_name_or_path=config["model"]["name"], revision=config["model"]["revision"])
    write_json(path, saved)
    return {path.name: file_hash(path) for path in sorted(destination.iterdir()) if path.is_file()}


def run(args, output, report):
    settings = variant_settings(args.variant)
    data_dir = lab_path(args.data_dir)
    config = deepcopy(read_config(args.config))
    if config["model"]["revision"] == "main" or config["lora"]["r"] != 16 or config["lora"]["alpha"] != 16 or config["lora"]["dropout"] != 0:
        raise ValueError("Require the pinned base and LoRA rank/alpha/dropout 16/16/0")
    config["training"].update(settings)
    config["training"].pop("warmup_ratio", None)
    config["training"].update(save_steps=10, save_only_milestones=settings["milestones"],
                              save_total_limit=None, logging_steps=10, eval_steps=10,
                              load_best_model_at_end=False)
    config["training"]["epochs"] = 1
    config["data"] = {split: str(data_dir / f"{args.variant}_{split}.jsonl") for split in ("train", "validation")}
    gate = verify_one_fact_gate(args.gate_report, data_dir) if args.variant == "mini" else None
    sources = [data_dir / f"{args.variant}_{split}.jsonl" for split in ("train", "validation", "benchmark")]
    sources.extend(data_dir / name for name in ("protocol.json", "data_audit.json"))
    source_hashes = {str(path.resolve()): file_hash(path) for path in sources}
    code_hashes = {str(path.resolve()): file_hash(path) for path in (Path(__file__), LAB / "common.py", LAB / "train.py", LAB / "evaluate_diagnostics.py")}
    report.update(variant=args.variant, config=config, settings=settings, gate=gate, data_sha256=source_hashes, code_sha256=code_hashes)
    write_json(output / "run_manifest.json", {"created_unix": time.time(), "config": config,
               "data_sha256": source_hashes, "code_sha256": code_hashes, "runtime": versions(), "mini_gate": gate,
               "interpretation": "Intentional memorization diagnostic; not a quality estimate or an ABC extension."})
    write_json(output / "effective_config.json", config)
    model, tokenizer = load_model(config, training=True)
    report["tokenizer"] = tokenizer_audit(tokenizer)
    write_json(output / "tokenizer_audit.json", report["tokenizer"])
    from unsloth import FastLanguageModel
    import torch
    from datasets import Dataset
    from transformers import set_seed, TrainerCallback
    from trl import SFTConfig, SFTTrainer
    set_seed(settings["seed"])
    model = FastLanguageModel.get_peft_model(
        model, r=16, target_modules=config["lora"]["target_modules"], lora_alpha=16, lora_dropout=0,
        bias="none", use_gradient_checkpointing="unsloth", random_state=settings["seed"],
        use_rslora=False, loftq_config=None)
    model.config.use_cache = False
    lookup = {}
    dataset_info = {}

    def encoded_dataset(split):
        rows = read_jsonl(lab_path(config["data"][split]))
        if not rows:
            raise ValueError(f"Empty {split}")
        encoded = [encode_assistant_only(tokenizer, row, config["model"]["max_seq_length"]) for row in rows]
        dataset_info[split] = {"rows": len(rows), "supervised_tokens_per_pass": sum(sum(t != -100 for t in e["labels"][1:]) for e in encoded),
                              "max_sequence_tokens": max(len(e["input_ids"]) for e in encoded), "overlength_dropped": 0,
                              "examples": [{"id": row["id"], "target_tokens": sum(t != -100 for t in e["labels"][1:]),
                                            "input_token_ids_sha256": token_hash(e["input_ids"]),
                                            "fact_id": row.get("metadata", {}).get("fact_id")} for row, e in zip(rows, encoded)]}
        if split == "train":
            for row, e in zip(rows, encoded):
                key = tuple(e["input_ids"])
                if key in lookup:
                    raise ValueError("Duplicate tokenized training conversation; consumed-example identities would be ambiguous")
                lookup[key] = row["id"]
        return Dataset.from_list(encoded)

    train, validation = encoded_dataset("train"), encoded_dataset("validation")
    report["dataset"] = dataset_info
    write_json(output / "token_audit.json", dataset_info)
    consumed = {"successful_microbatches": 0, "example_presentations": 0, "assistant_target_tokens": 0, "per_example": Counter()}
    initial = {}
    snapshots = {}

    def exposure_snapshot():
        return {key: dict(value) if isinstance(value, Counter) else value for key, value in consumed.items()}

    class CountingTrainer(SFTTrainer):
        def training_step(self, model, inputs, *positional, **kwargs):
            # Count only completed backward microsteps, never validation forwards.
            # labels[:, 1:] matches causal next-token prediction, including the
            # same assistant template/EOS targets used by the training loss.
            target_count = int((inputs["labels"][:, 1:] != -100).sum().item())
            row_ids = []
            for ids, mask in zip(inputs["input_ids"], inputs["attention_mask"]):
                key = tuple(ids[mask.bool()].detach().cpu().tolist())
                if key not in lookup:
                    raise RuntimeError("Training sampler yielded an unrecognized conversation")
                row_ids.append(lookup[key])
            loss = super().training_step(model, inputs, *positional, **kwargs)
            consumed["successful_microbatches"] += 1
            consumed["example_presentations"] += len(row_ids)
            consumed["assistant_target_tokens"] += target_count
            consumed["per_example"].update(row_ids)
            return loss

    class Milestones(TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            if state.global_step:
                raise RuntimeError("Diagnostics require fresh initialization, never checkpoint continuation")
            parameters = {name: tensor for name, tensor in kwargs["model"].named_parameters() if tensor.requires_grad}
            if not parameters or any("lora_" not in name for name in parameters):
                raise RuntimeError("Only LoRA parameters may be trainable")
            initial.update({name: tensor.detach().cpu().clone() for name, tensor in parameters.items()})
            report["initial_lora_sha256"] = trainable_state_hash(initial)
            report["trainable_parameters"] = sum(t.numel() for t in parameters.values())

        def on_step_end(self, args, state, control, **kwargs):
            if consumed["successful_microbatches"] < state.global_step:
                raise RuntimeError("Actual training-step exposure accounting was bypassed")
            control.should_save = state.global_step in settings["milestones"]
            if state.global_step % 10 == 0:
                snapshots[state.global_step] = exposure_snapshot()
                write_json(output / "exposure_by_step.json", snapshots)
            return control

        def on_save(self, args, state, control, **kwargs):
            step = state.global_step
            checkpoint = output / f"checkpoint-{step}"
            adapter = checkpoint / "adapter"
            hashes = save_adapter(kwargs["model"], tokenizer, adapter, config)
            parameters = {name: tensor for name, tensor in kwargs["model"].named_parameters() if tensor.requires_grad}
            changes = {name: float((tensor.detach().cpu().float() - initial[name].float()).abs().max()) for name, tensor in parameters.items()}
            proof = {"status": "saved", "variant": args_variant, "step": step, "adapter": str(adapter),
                     "initial_lora_sha256": report["initial_lora_sha256"], "lora_sha256": trainable_state_hash(parameters),
                     "changed_tensors": sum(v > 0 for v in changes.values()), "max_absolute_delta": max(changes.values()),
                     "adapter_files_sha256": hashes, "consumed": exposure_snapshot(), "base_model": config["model"],
                     "data_sha256": source_hashes, "config_sha256": file_hash(output / "effective_config.json")}
            write_json(checkpoint / "diagnostic_checkpoint.json", proof)
            print(json.dumps({"saved_step": step, "adapter": str(adapter), "consumed": proof["consumed"]}), flush=True)

    args_variant = args.variant
    bf16 = torch.cuda.is_bf16_supported()
    kwargs = dict(output_dir=str(output), per_device_train_batch_size=1, gradient_accumulation_steps=4,
                  max_steps=settings["max_steps"], num_train_epochs=1, learning_rate=settings["learning_rate"],
                  lr_scheduler_type="constant_with_warmup", warmup_steps=2,
                  weight_decay=config["training"]["weight_decay"], optim=config["training"]["optimizer"],
                  seed=3407, data_seed=3407, fp16=not bf16, bf16=bf16, gradient_checkpointing=True,
                  logging_steps=10, logging_nan_inf_filter=False, eval_strategy="steps", eval_steps=10,
                  save_strategy="steps", save_steps=10, save_total_limit=None, load_best_model_at_end=False,
                  per_device_eval_batch_size=1, eval_accumulation_steps=1, prediction_loss_only=True,
                  report_to="none", dataloader_num_workers=0, dataset_num_proc=1, packing=False,
                  dataset_kwargs={"skip_prepare_dataset": True}, remove_unused_columns=False)
    length_key = "max_length" if "max_length" in inspect.signature(SFTConfig).parameters else "max_seq_length"
    kwargs[length_key] = config["model"]["max_seq_length"]
    trainer = CountingTrainer(model=model, processing_class=tokenizer, train_dataset=train, eval_dataset=validation,
                              data_collator=LabelPreservingCollator(tokenizer.pad_token_id), args=SFTConfig(**kwargs), callbacks=[Milestones()])
    torch.cuda.reset_peak_memory_stats()
    result = trainer.train()
    trainer.save_state()
    loss_values = [float(row["loss"]) for row in trainer.state.log_history if "loss" in row]
    if trainer.state.global_step != settings["max_steps"] or not loss_values or not all(math.isfinite(v) for v in loss_values):
        raise RuntimeError("Training did not complete the fixed budget with finite losses")
    expected_consumed = sum(row["target_tokens"] * consumed["per_example"][row["id"]] for row in dataset_info["train"]["examples"])
    if expected_consumed != consumed["assistant_target_tokens"] or expected_consumed <= 0:
        raise RuntimeError("Consumed target-token count is inconsistent with actual sampled example identities")
    for step in settings["milestones"]:
        proof = json.loads((output / f"checkpoint-{step}" / "diagnostic_checkpoint.json").read_text(encoding="utf-8"))
        if not proof["changed_tensors"] or proof["initial_lora_sha256"] == proof["lora_sha256"]:
            raise RuntimeError("Checkpoint has no verified LoRA change")
    report.update(status="passed", global_step=trainer.state.global_step, ending_epoch=trainer.state.epoch,
                  metrics=result.metrics, losses=loss_values, consumed=exposure_snapshot(),
                  checkpoints={str(step): str(output / f"checkpoint-{step}" / "adapter") for step in settings["milestones"]},
                  cuda_peak_allocated_mib=round(torch.cuda.max_memory_allocated()/2**20, 2),
                  cuda_peak_reserved_mib=round(torch.cuda.max_memory_reserved()/2**20, 2), precision="bfloat16" if bf16 else "float16")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("one_fact", "mini", "style"), required=True)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--data-dir", default="datasets/diagnostics_v1")
    parser.add_argument("--output-dir")
    parser.add_argument("--gate-report")
    args = parser.parse_args()
    output = safe_output(args.output_dir or f"outputs/diagnostics_v1/{args.variant}")
    output.mkdir(parents=True, exist_ok=True)
    report = {"status": "running", "started_unix": time.time(), "command": sys.argv}
    started, code = time.perf_counter(), 0
    with NvidiaMonitor(output / "nvidia_vram.csv") as monitor:
        try:
            run(args, output, report)
        except Exception as exc:
            report.update(status="failed", error=str(exc), traceback=traceback.format_exc())
            traceback.print_exc()
            code = 1
    report.update(nvidia=monitor.summary(), elapsed_seconds=round(time.perf_counter()-started, 2))
    write_json(output / "training_report.json", report)
    print(json.dumps({k: report.get(k) for k in ("status", "global_step", "consumed", "error")}), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
