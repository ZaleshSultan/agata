"""Three-epoch local QLoRA comparison with prospective validation-only stopping."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass, field
import inspect
import json
import math
from pathlib import Path
import sys
import time
import traceback

from common import (LAB, LabelPreservingCollator, NvidiaMonitor, encode_assistant_only,
                    file_hash, lab_path, load_model, read_config, read_jsonl, write_json)
from evaluate_diagnostics import token_hash, tokenizer_audit
from run_abc import versions
from train import trainable_state_hash
from train_diagnostics import save_adapter

ARMS = {"lr5e-5": 5e-5, "lr1e-4": 1e-4}
BASE = {"name": "unsloth/Qwen3-1.7B-bnb-4bit",
        "revision": "d1d7781199f38692496fab70012c39598f0c858f"}
PROTOCOL_RULES = {
    "maximum_epochs": 3, "max_steps": -1, "seed": 3407,
    "learning_rates": ARMS, "batch_size": 1, "gradient_accumulation_steps": 4,
    "max_seq_length": 1024, "lora": {"r": 16, "alpha": 16, "dropout": 0},
    "scheduler": "linear", "warmup_ratio": 0.05,
    "early_stopping": {"patience_epochs": 2, "minimum_relative_improvement": 0.01,
                       "reference": "best significant loss, initialized with pre-training validation loss",
                       "catastrophic_validation_multiplier": 1.5,
                       "stop_on_nonfinite_loss_or_oom": True},
    "generation": {"do_sample": False, "max_new_tokens": 128, "enable_thinking": False},
    "guard_groups": {"general": ["general"], "unknown": ["unknown"],
                     "honesty": ["false_action", "honesty"]},
    "selection": {"source": "validation only", "complete_semantic_ratings_required": True,
                  "minimum_guard_pass_rate": "baseline for each guard group",
                  "maximum_guard_unsupported_count": "baseline for each guard group",
                  "order": ["minimum validation loss", "smaller optimizer step", "lr5e-5"],
                  "fallback": "baseline if no admissible adapter"},
}


def arm_settings(arm):
    if arm not in ARMS:
        raise ValueError("Unknown learning-rate arm")
    return {"epochs": 3, "max_steps": -1, "learning_rate": ARMS[arm], "seed": 3407,
            "batch_size": 1, "gradient_accumulation_steps": 4, "warmup_ratio": .05,
            "warmup_steps": 0, "lr_scheduler_type": "linear", "weight_decay": .01,
            "optimizer": "adamw_8bit", "save_total_limit": None}


def planned_steps(rows):
    if type(rows) is not int or rows <= 0:
        raise ValueError("Training must have a positive integer row count")
    per_epoch = math.ceil(rows / 4)
    return {"steps_per_epoch": per_epoch, "maximum_steps": 3 * per_epoch,
            "maximum_presentations": 3 * rows, "warmup_steps": math.ceil(3 * per_epoch * .05)}


@dataclass
class ValidationStopper:
    baseline: float
    best_significant: float = field(init=False)
    best_absolute: float = field(init=False)
    stale_epochs: int = 0
    history: list = field(default_factory=list)
    stop_reason: str | None = None

    def __post_init__(self):
        if not math.isfinite(self.baseline) or self.baseline <= 0:
            raise ValueError("Pre-training validation loss must be finite and positive")
        self.best_significant = self.best_absolute = self.baseline

    def observe(self, epoch, loss):
        if self.stop_reason:
            raise ValueError("Cannot evaluate another epoch after a stopping decision")
        if epoch != len(self.history) + 1 or not 1 <= epoch <= 3:
            raise ValueError("Epoch validation must be sequential within the three-epoch budget")
        if not math.isfinite(loss):
            self.stop_reason = "nonfinite_validation_loss"
            raise FloatingPointError(self.stop_reason)
        previous = self.best_significant
        relative = (previous - loss) / previous
        significant = loss <= previous * .99
        self.stale_epochs = 0 if significant else self.stale_epochs + 1
        if significant:
            self.best_significant = loss
        self.best_absolute = min(self.best_absolute, loss)
        catastrophic = loss > 1.5 * self.baseline
        if catastrophic:
            self.stop_reason = "validation_loss_above_1.5_times_baseline"
        elif self.stale_epochs >= 2:
            self.stop_reason = "two_epochs_without_one_percent_improvement"
        elif epoch == 3:
            self.stop_reason = "maximum_three_epochs"
        event = {"epoch": epoch, "validation_loss": loss, "previous_significant_best": previous,
                 "relative_improvement": relative, "significant_improvement": significant,
                 "stale_epochs": self.stale_epochs, "catastrophic": catastrophic,
                 "stop_reason": self.stop_reason}
        self.history.append(event)
        return event


def safe_output(value):
    output = lab_path(value).resolve()
    relative = output.relative_to((LAB / "outputs").resolve())
    if not relative.parts or not relative.parts[0].startswith("large"):
        raise ValueError("Large experiment outputs must be inside outputs/large*/")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Preserving existing experiment: {output}")
    return output


def checked_config(path, arm):
    config = deepcopy(read_config(path))
    if any(config["model"].get(key) != value for key, value in BASE.items()):
        raise ValueError("The exact pinned Qwen3 base is required")
    if config["model"]["max_seq_length"] != 1024 or not config["model"].get("load_in_4bit"):
        raise ValueError("Require sequence length 1024 and four-bit base loading")
    if any(config["lora"].get(key) != value for key, value in PROTOCOL_RULES["lora"].items()):
        raise ValueError("Require LoRA rank/alpha/dropout 16/16/0")
    expected = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    if set(config["lora"]["target_modules"]) != expected:
        raise ValueError("All seven frozen LoRA projection targets are required")
    config["training"] = arm_settings(arm)
    return config


def run(args, output, report):
    config = checked_config(args.config, args.arm)
    data_dir = lab_path(args.data_dir)
    protocol_path = data_dir / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if not str(protocol.get("status", "")).startswith("frozen"):
        raise ValueError("Protocol must be frozen before training")
    # Training never reads held-out benchmark/recall content.
    sources = [protocol_path, data_dir / "train.jsonl", data_dir / "validation.jsonl"]
    source_hashes = {str(p.resolve()): file_hash(p) for p in sources}
    for name in ("train.jsonl", "validation.jsonl"):
        declared = protocol.get("artifact_sha256", {}).get(name)
        if declared is not None and declared != file_hash(data_dir / name):
            raise ValueError(f"Frozen dataset changed: {name}")
    config["data"] = {split: str(data_dir / f"{split}.jsonl") for split in ("train", "validation")}
    code_names = ["train_large.py", "evaluate_large.py", "report_large.py", "common.py", "train.py", "train_diagnostics.py",
                  "evaluate_diagnostics.py", "run_abc.py"]
    code_hashes = {str((LAB / name).resolve()): file_hash(LAB / name) for name in code_names}
    rows_by_split = {split: read_jsonl(config["data"][split]) for split in ("train", "validation")}
    if any(not rows for rows in rows_by_split.values()):
        raise ValueError("Nonempty training and validation sets are required")
    for split, rows in rows_by_split.items():
        if len({row["id"] for row in rows}) != len(rows):
            raise ValueError(f"Duplicate {split} row IDs")
    plan = planned_steps(len(rows_by_split["train"]))
    report.update(arm=args.arm, config=config, rules=PROTOCOL_RULES, planned=plan,
                  data_sha256=source_hashes, code_sha256=code_hashes)
    write_json(output / "run_manifest.json", {"created_unix": time.time(), "config": config,
               "rules": PROTOCOL_RULES, "planned": plan, "data_sha256": source_hashes,
               "code_sha256": code_hashes, "runtime": versions(),
               "selection_note": "No benchmark is used for stopping or checkpoint selection."})
    write_json(output / "effective_config.json", config)
    model, tokenizer = load_model(config, training=True)
    from unsloth import FastLanguageModel
    import torch
    from datasets import Dataset
    from transformers import TrainerCallback, set_seed
    from trl import SFTConfig, SFTTrainer
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("This frozen comparison requires BF16 support")
    report["tokenizer"] = tokenizer_audit(tokenizer)
    write_json(output / "tokenizer_audit.json", report["tokenizer"])
    set_seed(3407)
    model = FastLanguageModel.get_peft_model(
        model, r=16, target_modules=config["lora"]["target_modules"], lora_alpha=16,
        lora_dropout=0, bias="none", use_gradient_checkpointing="unsloth", random_state=3407,
        use_rslora=False, loftq_config=None)
    model.config.use_cache = False
    before_parameters = {name: tensor for name, tensor in model.named_parameters() if tensor.requires_grad}
    lora_b = [tensor for name, tensor in before_parameters.items() if "lora_B" in name]
    if not lora_b or any(bool(torch.count_nonzero(tensor)) for tensor in lora_b):
        raise RuntimeError("Baseline validation requires zero-initialized LoRA-B")
    report["baseline_initial_lora_sha256"] = trainable_state_hash(before_parameters)
    report["baseline_lora_b_all_zero"] = True
    lookup, dataset_info, encoded_sets = {}, {}, {}
    for split, rows in rows_by_split.items():
        encoded = [encode_assistant_only(tokenizer, row, 1024) for row in rows]
        examples = [{"id": row["id"], "target_tokens": sum(t != -100 for t in entry["labels"][1:]),
                     "input_token_ids_sha256": token_hash(entry["input_ids"])}
                    for row, entry in zip(rows, encoded)]
        dataset_info[split] = {"rows": len(rows), "examples": examples,
                              "supervised_tokens_per_pass": sum(e["target_tokens"] for e in examples),
                              "max_sequence_tokens": max(len(e["input_ids"]) for e in encoded),
                              "overlength_dropped": 0}
        if split == "train":
            for row, entry in zip(rows, encoded):
                key = tuple(entry["input_ids"])
                if key in lookup:
                    raise ValueError("Duplicate tokenized training conversation")
                lookup[key] = row["id"]
        encoded_sets[split] = Dataset.from_list(encoded)
    report["dataset"] = dataset_info
    write_json(output / "token_audit.json", dataset_info)
    consumed = {"successful_microbatches": 0, "example_presentations": 0,
                "assistant_target_tokens": 0, "per_example": Counter()}
    report["consumed"] = consumed
    report["checkpoints"] = []
    initial, stopping = {}, {}

    def exposure():
        return {key: dict(value) if isinstance(value, Counter) else value for key, value in consumed.items()}

    class CountingTrainer(SFTTrainer):
        def training_step(self, model, inputs, *positional, **kwargs):
            targets = int((inputs["labels"][:, 1:] != -100).sum().item())
            row_ids = [lookup[tuple(ids[mask.bool()].detach().cpu().tolist())]
                       for ids, mask in zip(inputs["input_ids"], inputs["attention_mask"])]
            loss = super().training_step(model, inputs, *positional, **kwargs)
            if not bool(torch.isfinite(loss).all()):
                raise FloatingPointError("Nonfinite training microstep loss")
            consumed["successful_microbatches"] += 1
            consumed["example_presentations"] += len(row_ids)
            consumed["assistant_target_tokens"] += targets
            consumed["per_example"].update(row_ids)
            return loss

    class EpochControl(TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            if state.global_step or args.max_steps != -1 or args.num_train_epochs != 3:
                raise RuntimeError("Fresh initialization and exactly three scheduled epochs required")
            if args.world_size != 1 or state.max_steps != plan["maximum_steps"]:
                raise RuntimeError("Trainer step plan differs from the preflight single-GPU calculation")
            parameters = {n: t for n, t in kwargs["model"].named_parameters() if t.requires_grad}
            if not parameters or any("lora_" not in n for n in parameters):
                raise RuntimeError("Only LoRA tensors may be trained")
            initial.update({n: t.detach().cpu().clone() for n, t in parameters.items()})
            report["initial_lora_sha256"] = trainable_state_hash(initial)
            if report["initial_lora_sha256"] != report["baseline_initial_lora_sha256"]:
                raise RuntimeError("Baseline evaluation changed trainable adapter weights")
            report["trainable_parameters"] = sum(t.numel() for t in parameters.values())
            write_json(output / "initial_adapter_proof.json", {"sha256": report["initial_lora_sha256"],
                       "trainable_parameters": report["trainable_parameters"], "trainer_max_steps": state.max_steps})

        def on_epoch_end(self, args, state, control, **kwargs):
            control.should_log = control.should_evaluate = control.should_save = True
            return control

        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            if "eval_loss" not in (metrics or {}) or not state.global_step:
                return control
            epoch = round(state.epoch)
            if abs(state.epoch - epoch) > 1e-7:
                raise RuntimeError("Validation must occur at a complete epoch boundary")
            event = stopping["state"].observe(epoch, float(metrics["eval_loss"]))
            event["step"] = state.global_step
            report["validation_history"] = stopping["state"].history
            report["stop_reason"] = event["stop_reason"]
            write_json(output / "validation_history.json", {"baseline": report["baseline_validation"],
                       "epochs": report["validation_history"]})
            if event["stop_reason"]:
                control.should_training_stop = True
            return control

        def on_save(self, args, state, control, **kwargs):
            epoch, step = round(state.epoch), state.global_step
            event = stopping["state"].history[-1]
            if event["epoch"] != epoch or event["step"] != step:
                raise RuntimeError("Checkpoint lacks its corresponding epoch validation")
            if any(consumed["per_example"][row["id"]] != epoch for row in dataset_info["train"]["examples"]):
                raise RuntimeError("A completed epoch did not expose each training row exactly once")
            checkpoint = output / f"checkpoint-{step}"
            adapter = checkpoint / "adapter"
            parameters = {n: t for n, t in kwargs["model"].named_parameters() if t.requires_grad}
            if any(not bool(torch.isfinite(t).all()) for t in parameters.values()):
                raise FloatingPointError("Nonfinite LoRA weights")
            hashes = save_adapter(kwargs["model"], tokenizer, adapter, config)
            changes = [float((t.detach().cpu().float() - initial[n].float()).abs().max())
                       for n, t in parameters.items()]
            proof = {"status": "saved", "arm": args_arm, "epoch": epoch, "step": step,
                     "adapter": str(adapter.resolve()), "initial_lora_sha256": report["initial_lora_sha256"],
                     "lora_sha256": trainable_state_hash(parameters), "changed_tensors": sum(v > 0 for v in changes),
                     "max_absolute_delta": max(changes), "adapter_files_sha256": hashes,
                     "validation": event, "consumed": exposure(), "base_model": config["model"],
                     "data_sha256": source_hashes, "config_sha256": file_hash(output / "effective_config.json")}
            if not proof["changed_tensors"]:
                raise RuntimeError("Saved LoRA did not change")
            write_json(checkpoint / "large_checkpoint.json", proof)
            report["checkpoints"].append({"epoch": epoch, "step": step, "adapter": str(adapter),
                                          "validation_loss": event["validation_loss"],
                                          "catastrophic": event["catastrophic"]})
            write_json(output / "training_progress.json", report)
            print(json.dumps({"saved_epoch": epoch, "step": step, "validation": event}), flush=True)

    args_arm = args.arm
    kwargs = dict(output_dir=str(output), per_device_train_batch_size=1, gradient_accumulation_steps=4,
                  max_steps=-1, num_train_epochs=3, learning_rate=ARMS[args.arm], lr_scheduler_type="linear",
                  warmup_ratio=.05, warmup_steps=0, weight_decay=.01, optim="adamw_8bit",
                  seed=3407, data_seed=3407, fp16=False, bf16=True, gradient_checkpointing=True,
                  logging_steps=10, logging_nan_inf_filter=False, eval_strategy="epoch", save_strategy="epoch",
                  save_total_limit=None, save_only_model=False, load_best_model_at_end=False, per_device_eval_batch_size=1,
                  eval_accumulation_steps=1, prediction_loss_only=True, report_to="none",
                  dataloader_num_workers=0, dataset_num_proc=1, packing=False,
                  dataset_kwargs={"skip_prepare_dataset": True}, remove_unused_columns=False)
    length_key = "max_length" if "max_length" in inspect.signature(SFTConfig).parameters else "max_seq_length"
    kwargs[length_key] = 1024
    trainer = CountingTrainer(model=model, processing_class=tokenizer, train_dataset=encoded_sets["train"],
                              eval_dataset=encoded_sets["validation"], args=SFTConfig(**kwargs),
                              data_collator=LabelPreservingCollator(tokenizer.pad_token_id), callbacks=[EpochControl()])
    torch.cuda.reset_peak_memory_stats()
    # Zero-initialized LoRA-B leaves the pinned base unchanged. This evaluation is
    # teacher-forced validation, with identical collator/model precision to epoch evaluation.
    baseline = trainer.evaluate(metric_key_prefix="baseline")
    stopping["state"] = ValidationStopper(float(baseline["baseline_loss"]))
    report["baseline_validation"] = baseline
    write_json(output / "baseline_validation.json", baseline)
    set_seed(3407)
    try:
        result = trainer.train()
        expected = sum(e["target_tokens"] * consumed["per_example"][e["id"]]
                       for e in dataset_info["train"]["examples"])
        if expected != consumed["assistant_target_tokens"] or not report["checkpoints"]:
            raise RuntimeError("Missing checkpoint or inconsistent actual target-token consumption")
        report.update(status="stopped_catastrophic" if any(e["catastrophic"] for e in stopping["state"].history) else "complete",
                      global_step=trainer.state.global_step, ending_epoch=trainer.state.epoch,
                      metrics=result.metrics, consumed=exposure(), precision="bfloat16",
                      selection_status="pending_full_validation_semantic_guard")
    finally:
        trainer.save_state()
        report.update(cuda_peak_allocated_mib=round(torch.cuda.max_memory_allocated()/2**20, 2),
                      cuda_peak_reserved_mib=round(torch.cuda.max_memory_reserved()/2**20, 2),
                      consumed=exposure())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=tuple(ARMS), required=True)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--data-dir", default="datasets/large_v3")
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    output = safe_output(args.output_dir or f"outputs/large_v3/{args.arm}")
    output.mkdir(parents=True, exist_ok=True)
    report = {"status": "running", "started_unix": time.time(), "command": sys.argv}
    started, code = time.perf_counter(), 0
    with NvidiaMonitor(output / "nvidia_vram.csv") as monitor:
        try:
            run(args, output, report)
        except Exception as exc:
            catastrophic = isinstance(exc, FloatingPointError) or "out of memory" in str(exc).lower()
            report.update(status="failed_catastrophic" if catastrophic else "failed", error=str(exc), traceback=traceback.format_exc())
            traceback.print_exc()
            code = 1
    report.update(nvidia=monitor.summary(), elapsed_seconds=round(time.perf_counter()-started, 2))
    write_json(output / "training_report.json", report)
    print(json.dumps({key: report.get(key) for key in ("status", "global_step", "stop_reason", "error")}), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
