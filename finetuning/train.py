"""Isolated Unsloth QLoRA training with measurable smoke test and resume support."""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
from pathlib import Path
import sys
import time
import traceback

from common import (LAB, LabelPreservingCollator, NvidiaMonitor, encode_assistant_only,
                    file_hash, lab_path, load_model, read_config, read_jsonl, write_json)


def trainable_state_hash(tensors):
    """Stable ordered digest; canonical FP32 bytes avoid serialization metadata."""
    digest = hashlib.sha256()
    for name, tensor in sorted(tensors.items()):
        value = tensor.detach().cpu().float().contiguous()
        header = {"name": name, "shape": list(value.shape), "source_dtype": str(tensor.dtype),
                  "canonical_dtype": "float32"}
        digest.update(json.dumps(header, sort_keys=True).encode("utf-8"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--max-seq-length", type=int)
    parser.add_argument("--output-dir")
    parser.add_argument("--resume", help="Explicit checkpoint directory; trainer/optimizer state is restored")
    parser.add_argument("--smoke-report", default="outputs/smoke/training_report.json")
    return parser.parse_args()


def run(args, output, report):
    config = read_config(args.config)
    if args.max_seq_length:
        config["model"]["max_seq_length"] = args.max_seq_length
    report["config"] = config
    if not args.smoke:
        smoke = json.loads(lab_path(args.smoke_report).read_text(encoding="utf-8"))
        if smoke.get("status") != "passed" or not smoke.get("lora_update", {}).get("changed_tensors"):
            raise RuntimeError("A successful QLoRA smoke-test report is required before full training")
        if not smoke.get("adapter_reload_verified"):
            raise RuntimeError("Reload the smoke adapter with inference.py before full training")
        if any(smoke["config"]["model"].get(key) != config["model"].get(key) for key in ("name", "revision")):
            raise RuntimeError("Smoke test used another base model or revision")
        if smoke["config"]["lora"] != config["lora"]:
            raise RuntimeError("LoRA settings differ from the verified smoke test; run a new smoke test")
    source = lab_path(config["data"]["smoke" if args.smoke else "train"])
    report["dataset"] = {"path": str(source), "sha256": file_hash(source)}
    model, tokenizer = load_model(config, training=True)
    from unsloth import FastLanguageModel
    import torch
    from datasets import Dataset
    from transformers import set_seed, TrainerCallback
    from trl import SFTConfig, SFTTrainer
    seed = config["training"]["seed"]
    set_seed(seed)
    model = FastLanguageModel.get_peft_model(
        model, r=config["lora"]["r"], target_modules=config["lora"]["target_modules"],
        lora_alpha=config["lora"]["alpha"], lora_dropout=config["lora"]["dropout"],
        bias="none", use_gradient_checkpointing="unsloth", random_state=seed,
        use_rslora=False, loftq_config=None,
    )
    model.config.use_cache = False
    trainable = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
    if not trainable or any("lora_" not in name for name in trainable):
        raise RuntimeError("Only LoRA parameters must be trainable")
    before = {}
    report["trainable_parameters"] = sum(t.numel() for t in trainable.values())
    dropped = []

    accepted = {"train": [], "validation": []}

    def encode(rows, split):
        encoded = []
        for row in rows:
            try:
                encoded.append(encode_assistant_only(tokenizer, row, config["model"]["max_seq_length"]))
                accepted[split].append({"id": row.get("id"), "group_id": row.get("group_id"),
                                        "fact_id": row.get("metadata", {}).get("fact_id") or row.get("id")})
            except OverflowError as exc:
                dropped.append(str(exc))
        if not encoded:
            raise ValueError("No complete conversations fit the sequence length; inspect dataset lengths")
        return Dataset.from_list(encoded)

    train = encode(read_jsonl(source), "train")
    validation = None if args.smoke else encode(read_jsonl(lab_path(config["data"]["validation"])), "validation")
    report["dataset"].update({"accepted_rows": len(train), "overlength_dropped": dropped,
                              "supervised_tokens": sum(sum(t != -100 for t in row["labels"]) for row in train),
                              "accepted_examples": accepted["train"]})
    manifest = {"config": config, "dataset_sha256": report["dataset"]["sha256"],
                "accepted_examples": accepted["train"]}
    if args.resume:
        previous_path = lab_path(args.resume).resolve().parent / "training_manifest.json"
        if not previous_path.is_file():
            raise ValueError("Checkpoint has no training_manifest.json beside it; dataset/config compatibility cannot be verified")
        previous = json.loads(previous_path.read_text(encoding="utf-8"))
        if previous != manifest:
            raise ValueError("Resume dataset/config differs from the checkpoint manifest. Start a separate experiment.")
    write_json(output / "training_manifest.json", manifest)
    write_json(output / "effective_config.json", config)
    # Keep the reason visible; no example is partly trained by silent truncation.
    if dropped:
        print(f"Excluded {len(dropped)} overlength conversations; details in training report", flush=True)
    train_config = config["training"]
    max_steps = args.max_steps if args.max_steps is not None else (5 if args.smoke else -1)
    bf16 = torch.cuda.is_bf16_supported()
    kwargs = dict(
        output_dir=str(output), per_device_train_batch_size=train_config["batch_size"],
        gradient_accumulation_steps=train_config["gradient_accumulation_steps"],
        learning_rate=train_config["learning_rate"], num_train_epochs=train_config["epochs"],
        max_steps=max_steps,
        weight_decay=train_config["weight_decay"], optim=train_config["optimizer"],
        lr_scheduler_type="linear", seed=seed, data_seed=seed, fp16=not bf16, bf16=bf16,
        gradient_checkpointing=True, logging_steps=1, logging_nan_inf_filter=False, save_strategy="steps",
        save_steps=1 if args.smoke else train_config["save_steps"], save_total_limit=2,
        report_to="none", dataloader_num_workers=0, dataset_num_proc=1,
        packing=False, dataset_kwargs={"skip_prepare_dataset": True},
        remove_unused_columns=False, eval_strategy="no" if args.smoke else "epoch",
        per_device_eval_batch_size=1, eval_accumulation_steps=1, prediction_loss_only=True,
    )
    parameters = inspect.signature(SFTConfig).parameters
    length_key = "max_length" if "max_length" in parameters else "max_seq_length"
    kwargs[length_key] = config["model"]["max_seq_length"]
    # Transformers 5.17 replaced warmup_ratio with fractional warmup_steps.
    warmup_key = "warmup_ratio" if "warmup_ratio" in parameters else "warmup_steps"
    kwargs[warmup_key] = train_config["warmup_ratio"]

    class CaptureRestoredWeights(TrainerCallback):
        def on_train_begin(self, args, state, control, **callback_kwargs):
            report["starting_global_step"] = state.global_step
            report["starting_epoch"] = state.epoch or 0.0
            # Capture after Trainer preparation and (if requested) checkpoint
            # restoration. Neither setup-time transformations nor restoration
            # should be counted as optimizer updates, including on fresh runs.
            active_model = callback_kwargs.get("model", model)
            trainable.clear()
            trainable.update({name: parameter for name, parameter in active_model.named_parameters()
                              if parameter.requires_grad})
            if not trainable or any("lora_" not in name for name in trainable):
                raise RuntimeError("Trainer preparation changed the intended LoRA-only trainable parameters")
            before.clear()
            before.update({name: tensor.detach().cpu().clone() for name, tensor in trainable.items()})
            report["initial_lora_sha256"] = trainable_state_hash(before)
            report["initial_lora_hash_method"] = "sorted name/shape/source dtype plus canonical contiguous FP32 tensor bytes before optimizer updates"

    report["resumed_from"] = args.resume
    trainer = SFTTrainer(
        model=model, processing_class=tokenizer, train_dataset=train, eval_dataset=validation,
        data_collator=LabelPreservingCollator(tokenizer.pad_token_id), args=SFTConfig(**kwargs),
        callbacks=[CaptureRestoredWeights()],
    )
    torch.cuda.reset_peak_memory_stats()
    result = trainer.train(resume_from_checkpoint=str(lab_path(args.resume)) if args.resume else None)
    starting_step = report.get("starting_global_step", 0)
    if trainer.state.global_step <= starting_step:
        raise RuntimeError("No new optimizer steps were performed (checkpoint may already meet max_steps)")
    losses = [float(row["loss"]) for row in trainer.state.log_history
              if "loss" in row and row.get("step", 0) > starting_step]
    if not losses or not all(math.isfinite(loss) for loss in losses) or not math.isfinite(result.training_loss):
        raise RuntimeError("Training did not produce finite loss values")
    changes = {name: float((tensor.detach().cpu().float() - before[name].float()).abs().max())
               for name, tensor in trainable.items()}
    changed = sum(delta > 0 for delta in changes.values())
    if not changed:
        raise RuntimeError("Optimizer steps did not change any LoRA tensor")
    adapter = output / "adapter"
    model.save_pretrained(str(adapter), safe_serialization=True)
    tokenizer.save_pretrained(str(adapter))
    adapter_config = json.loads((adapter / "adapter_config.json").read_text(encoding="utf-8"))
    adapter_config["base_model_name_or_path"] = config["model"]["name"]
    adapter_config["revision"] = config["model"]["revision"]
    write_json(adapter / "adapter_config.json", adapter_config)
    trainer.save_state()
    report.update({
        "status": "passed", "global_step": trainer.state.global_step, "losses": losses,
        "ending_epoch": trainer.state.epoch,
        # On resume, the unchanged manifest ties the cumulative epoch counter to
        # the same examples; those already learned by the checkpoint still count.
        "dataset_full_pass_verified": (trainer.state.epoch or 0.0) >= 1.0 - 1e-6,
        "this_invocation_full_pass_verified": (trainer.state.epoch or 0.0) - report.get("starting_epoch", 0.0) >= 1.0 - 1e-6,
        "metrics": result.metrics, "adapter": str(adapter),
        "precision": "bfloat16" if bf16 else "float16",
        "lora_update": {"changed_tensors": changed, "total_tensors": len(changes),
                        "maximum_absolute_delta": max(changes.values())},
        "final_lora_sha256": trainable_state_hash(trainable),
        "cuda_peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20, 2),
        "cuda_peak_reserved_mib": round(torch.cuda.max_memory_reserved() / 2**20, 2),
        "adapter_reload_verified": False,
        "interpretation": ("Infrastructure smoke-test only; loss/updates do not prove knowledge acquisition."
                           if args.smoke else "Training completed; use held-out and source-stratified evaluation before making quality claims."),
    })
    # Reload in a fresh process to avoid retaining two GPU models at the same time.
    report["reload_command"] = f'python finetuning/inference.py --adapter "{adapter}" --output outputs/reload.json'
    write_json(output / "effective_config.json", config)


def main():
    args = parse_args()
    output = lab_path(args.output_dir or ("outputs/smoke" if args.smoke else "outputs/train-" + time.strftime("%Y%m%d-%H%M%S")))
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise SystemExit(f"Output directory is not empty: {output}. Choose another --output-dir or --resume.")
    output.mkdir(parents=True, exist_ok=True)
    report = {"status": "running", "smoke": args.smoke, "started_unix": time.time(), "command": sys.argv}
    started = time.perf_counter()
    exit_code = 0
    with NvidiaMonitor(output / "nvidia_vram.csv") as monitor:
        try:
            run(args, output, report)
        except Exception as exc:
            report.update({"status": "failed", "error": str(exc), "traceback": traceback.format_exc()})
            if "out of memory" in str(exc).lower():
                report["oom_retry"] = "Use a NEW output directory with --max-seq-length 512, then 256. Batch size remains 1."
            traceback.print_exc()
            exit_code = 1
    report["nvidia"] = monitor.summary()
    report["elapsed_seconds"] = round(time.perf_counter() - started, 2)
    write_json(output / "training_report.json", report)
    print(json.dumps({key: report.get(key) for key in ("status", "losses", "lora_update", "nvidia", "error")}, ensure_ascii=False), flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
