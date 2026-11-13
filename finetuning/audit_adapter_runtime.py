"""GPU evidence for saved LoRA application and causal assistant-token loss."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time

from common import (NvidiaMonitor, encode_assistant_only, file_hash, lab_path,
                    load_model, read_config, read_jsonl, render_chat, write_json)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="datasets/abc_v1/configs/B.yaml")
    parser.add_argument("--adapter", default="outputs/abc_v1/B/adapter")
    parser.add_argument("--output", default="evaluation/reports/diagnostics_v1/adapter_runtime.json")
    args = parser.parse_args()
    output = lab_path(args.output)
    if output.exists():
        raise SystemExit(f"Preserving existing audit: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    config = read_config(args.config)
    result = {"status": "running", "created_unix": time.time(), "config": config,
              "adapter": str(lab_path(args.adapter).resolve()), "samples": []}
    with NvidiaMonitor(output.with_suffix(".vram.csv")) as monitor:
        model, tokenizer = load_model(config, adapter=args.adapter)
        import torch
        from torch.nn import functional as F
        from peft import get_peft_model_state_dict, PeftModel
        from safetensors.torch import load_file
        assert isinstance(model, PeftModel)
        model.eval()
        result["tokenizer_metadata"] = {"chat_template_sha256": hashlib.sha256(tokenizer.chat_template.encode("utf-8")).hexdigest(),
            "eos_token": tokenizer.eos_token, "eos_token_id": tokenizer.eos_token_id,
            "pad_token": tokenizer.pad_token, "pad_token_id": tokenizer.pad_token_id,
            "bos_token_id": tokenizer.bos_token_id}
        result["loaded_generation_config"] = model.generation_config.to_dict()
        result["active_adapters"] = list(model.active_adapters)
        saved_path = lab_path(args.adapter) / "adapter_model.safetensors"
        saved = load_file(str(saved_path), device="cpu")
        live = get_peft_model_state_dict(model)
        assert set(saved) == set(live), "Saved/live adapter parameter keys differ"
        differences, cast_differences = {}, {}
        for key, value in live.items():
            value = value.detach().cpu()
            differences[key] = float((value.float() - saved[key].float()).abs().max())
            cast_differences[key] = float((value.float() - saved[key].to(value.dtype).float()).abs().max())
        result["adapter_weight_evidence"] = {
            "file_sha256": file_hash(saved_path), "tensor_count": len(saved),
            "max_absolute_difference": max(differences.values()),
            "max_difference_after_loaded_dtype_cast": max(cast_differences.values()),
            "all_saved_tensors_matched_after_dtype_cast": all(v == 0 for v in cast_differences.values()),
            "loaded_dtypes": sorted({str(value.dtype) for value in live.values()})}
        del saved, live

        def forward(row, with_labels=False):
            encoded = encode_assistant_only(tokenizer, row, config["model"]["max_seq_length"])
            batch = {key: torch.tensor([value], device=model.device) for key, value in encoded.items()}
            with torch.inference_mode():
                answer = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                               labels=batch["labels"] if with_labels else None, use_cache=False)
            return answer, batch

        def target_loss(row):
            answer, batch = forward(row)
            logits = answer.logits
            if not isinstance(logits, torch.Tensor) or logits.ndim != 3:
                raise RuntimeError("Full logits unavailable; cannot independently audit loss")
            targets = batch["labels"][:, 1:]
            with torch.inference_mode():
                ce = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                                     targets.reshape(-1), ignore_index=-100)
            selected = logits[:, :-1][targets != -100].detach().float().cpu()
            del answer, logits, batch
            return float(ce), selected

        def raw_generation(row):
            question = {"messages": row["messages"][:-1]}
            rendered = render_chat(tokenizer, question, generation=True)
            batch = tokenizer(rendered, add_special_tokens=False, return_tensors="pt").to(model.device)
            with torch.inference_mode():
                generated = model.generate(**batch, do_sample=False, max_new_tokens=96, use_cache=True,
                    pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
            tokens = generated[0, batch["input_ids"].shape[1]:].tolist()
            return {"token_ids": tokens, "raw": tokenizer.decode(tokens, skip_special_tokens=False),
                    "ended_with_eos": bool(tokens and tokens[-1] == tokenizer.eos_token_id)}

        rows = read_jsonl(lab_path(config["data"]["train"]))
        indices = sorted({0, min(18, len(rows)-1), min(32, len(rows)-1)})
        for index in indices:
            row = rows[index]
            enabled_loss, enabled_logits = target_loss(row)
            answer, _ = forward(row, with_labels=True)
            reported = float(answer.loss)
            del answer
            with model.disable_adapter():
                disabled_loss, disabled_logits = target_loss(row)
            difference = (enabled_logits - disabled_logits).abs()
            restored_loss, restored_logits = target_loss(row)
            sample = {"id": row["id"], "enabled_manual_ce": enabled_loss,
                      "model_reported_loss": reported, "loss_absolute_difference": abs(enabled_loss-reported),
                      "disabled_manual_ce": disabled_loss,
                      "enabled_vs_disabled_max_logit_delta": float(difference.max()),
                      "enabled_vs_disabled_mean_logit_delta": float(difference.mean()),
                      "restored_max_logit_delta": float((enabled_logits-restored_logits).abs().max()),
                      "restored_manual_ce": restored_loss}
            if index == 0:
                sample["enabled_generation"] = raw_generation(row)
                with model.disable_adapter():
                    sample["disabled_generation"] = raw_generation(row)
                # Stronger independent intervention: zero only B matrices in RAM,
                # then restore them exactly. No saved weight is ever modified.
                b_matrices = {name: parameter for name, parameter in model.named_parameters() if ".lora_B." in name}
                originals = {name: parameter.detach().cpu().clone() for name, parameter in b_matrices.items()}
                try:
                    with torch.no_grad():
                        for parameter in b_matrices.values():
                            parameter.zero_()
                    zero_loss, zero_logits = target_loss(row)
                    sample["zero_B_manual_ce"] = zero_loss
                    sample["zero_B_vs_disabled_max_logit_delta"] = float((zero_logits-disabled_logits).abs().max())
                finally:
                    with torch.no_grad():
                        for name, parameter in b_matrices.items():
                            parameter.copy_(originals[name].to(parameter.device))
                after_loss, after_logits = target_loss(row)
                sample["after_zero_restore_max_logit_delta"] = float((after_logits-enabled_logits).abs().max())
                del originals, b_matrices
            result["samples"].append(sample)
            write_json(output, result)
            print(json.dumps({key: value for key,value in sample.items() if not key.endswith("generation")}), flush=True)
            del enabled_logits, disabled_logits, restored_logits, difference
        validation = []
        for row in read_jsonl(lab_path(config["data"]["validation"])):
            loss, logits = target_loss(row)
            validation.append({"id": row["id"], "assistant_token_ce": loss})
            del logits
        result["reloaded_validation"] = {"rows": validation, "mean_per_example_loss": sum(row["assistant_token_ce"] for row in validation)/len(validation)}
        original_state = lab_path(args.adapter).parent / "trainer_state.json"
        if original_state.exists():
            history = json.loads(original_state.read_text(encoding="utf-8"))["log_history"]
            previous = [row["eval_loss"] for row in history if "eval_loss" in row]
            result["original_last_validation_loss"] = previous[-1] if previous else None
        result["status"] = "complete"
    result["nvidia"] = monitor.summary()
    write_json(output, result)
    print(f"Runtime audit saved: {output}", flush=True)


if __name__ == "__main__":
    main()
