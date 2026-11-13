"""Raw decoding, exact-answer probes and teacher-forced NLL for local diagnostics."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import time
import unicodedata

from common import (LAB, NvidiaMonitor, encode_assistant_only, file_hash, lab_path,
                    load_model, read_config, read_jsonl, render_chat, write_json)


def token_hash(ids):
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode("ascii")).hexdigest()


def tokenizer_audit(tokenizer):
    template = tokenizer.chat_template
    serialized = template if isinstance(template, str) else json.dumps(template, ensure_ascii=False, sort_keys=True)
    return {"chat_template_sha256": hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
            "special_tokens_map": tokenizer.special_tokens_map,
            "special_token_ids": {key: getattr(tokenizer, key, None) for key in ("eos_token_id", "pad_token_id", "bos_token_id", "unk_token_id")},
            "vocab_size": len(tokenizer), "is_fast": tokenizer.is_fast, "name_or_path": tokenizer.name_or_path}


def normalize_answer(text, case_sensitive=False):
    text = unicodedata.normalize("NFKC", str(text))
    text = " ".join(text.split()).strip().rstrip(".!?").rstrip()
    return text if case_sensitive else text.casefold()


def lexical_scores(response, metadata):
    expected = metadata.get("expected_answer")
    if expected is None:
        return {"exact_normalized": None, "answer_inclusion": None,
                "note": "No frozen exact target; substantive/style correctness requires rubric review."}
    normalized = normalize_answer(response, metadata.get("case_sensitive", False))
    target = normalize_answer(expected, metadata.get("case_sensitive", False))
    if not target:
        raise ValueError("An empty exact target would give a meaningless inclusion score")
    # Boundaries prevent a longer fabricated identifier from passing inclusion.
    inclusion = bool(re.search(r"(?<![\w-])" + re.escape(target) + r"(?![\w-])", normalized))
    return {"exact_normalized": normalized == target, "answer_inclusion": inclusion,
            "normalized_response": normalized, "normalized_expected": target,
            "note": "Lexical diagnostics only: inclusion can coexist with contradictory or invented extra claims."}


def generation_row(row):
    messages = row["messages"]
    if not messages or messages[-1]["role"] != "user" or any(m["role"] not in ("system", "user") for m in messages):
        raise ValueError("Evaluation generation inputs must contain only system/user turns and end with the user")
    # References/rubrics/expected identifiers are deliberately not passed through.
    return {"messages": messages, "tools": row.get("tools") or []}


def raw_generate(model, tokenizer, row, max_new_tokens):
    import torch
    prompt = render_chat(tokenizer, generation_row(row), generation=True)
    inputs = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to(model.device)
    input_ids = inputs["input_ids"][0].detach().cpu().tolist()
    started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True,
                                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    ids = output[0, len(input_ids):].detach().cpu().tolist()
    raw = tokenizer.decode(ids, skip_special_tokens=False)
    decoded = tokenizer.decode(ids, skip_special_tokens=True)
    clean = re.sub(r"<think>.*?</think>", "", decoded, flags=re.DOTALL).strip()
    eos = bool(ids and ids[-1] == tokenizer.eos_token_id)
    return {"input_token_ids": input_ids, "input_token_ids_sha256": token_hash(input_ids),
            "rendered_prompt": prompt, "rendered_prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "generated_token_ids": ids, "raw_decoded": raw, "decoded_skip_special": decoded, "response": clean,
            "generated_tokens": len(ids), "ended_with_eos": eos, "hit_token_limit": len(ids) >= max_new_tokens and not eos,
            "finish_reason": "eos" if eos else ("length" if len(ids) >= max_new_tokens else "other"),
            "think_open_count": raw.count("<think>"), "think_close_count": raw.count("</think>"),
            "unclosed_think": raw.count("<think>") > raw.count("</think>"),
            "seconds": round(time.perf_counter()-started, 3)}


def teacher_forced_nll(model, tokenizer, row, max_length):
    import torch
    import torch.nn.functional as F
    reference = row.get("metadata", {}).get("reference")
    if not isinstance(reference, str) or not reference:
        raise ValueError(f"Missing metadata.reference for NLL: {row.get('id')}")
    input_row = generation_row(row)
    target_row = {"id": row["id"], "messages": input_row["messages"] + [{"role": "assistant", "content": reference}],
                  "tools": input_row["tools"]}
    encoded = encode_assistant_only(tokenizer, target_row, max_length)
    ids = torch.tensor([encoded["input_ids"]], dtype=torch.long, device=model.device)
    attention = torch.ones_like(ids)
    labels = torch.tensor(encoded["labels"][1:], dtype=torch.long, device=model.device)
    mask = labels != -100
    with torch.inference_mode():
        outputs = model(input_ids=ids, attention_mask=attention, use_cache=False, logits_to_keep=0)
        logits = outputs.logits
        if not isinstance(logits, torch.Tensor) or logits.shape[1] != ids.shape[1]:
            raise RuntimeError("Teacher-forced NLL requires full-sequence logits; refusing a partial/logits-disabled result")
        losses = F.cross_entropy(logits[0, :-1][mask].float(), labels[mask], reduction="none")
    values = losses.detach().cpu().tolist()
    target_ids = labels[mask].detach().cpu().tolist()
    if not values or not all(math.isfinite(value) for value in values):
        raise RuntimeError("Teacher-forced NLL is missing or nonfinite")
    return {"mean_nll": sum(values)/len(values), "total_nll": sum(values), "target_tokens": len(values),
            "target_token_ids": target_ids, "token_nll": values, "reference": reference,
            "teacher_input_token_ids_sha256": token_hash(encoded["input_ids"]),
            "target_definition": "Same causal assistant-only labels as training, including assistant template/empty-think and EOS targets; not semantic correctness."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--adapter")
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--output", required=True, help="New report path stem relative to finetuning")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()
    output = lab_path(args.output)
    report_path = output.with_suffix(".json")
    if report_path.exists() or output.with_suffix(".md").exists():
        raise FileExistsError(f"Preserving existing evaluation: {report_path}")
    benchmark = lab_path(args.benchmark)
    rows = read_jsonl(benchmark)
    for row in rows:
        generation_row(row)
    config = read_config(args.config)
    protocol_path = benchmark.parent / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8")) if protocol_path.exists() else {}
    model, tokenizer = load_model(config, adapter=args.adapter)
    active = []
    if args.adapter:
        from peft import PeftModel
        if not isinstance(model, PeftModel) or not model.active_adapters:
            raise RuntimeError("A freshly loaded active PEFT adapter is required")
        active = list(model.active_adapters)
    report = {"status": "running", "created_unix": time.time(), "base_model": config["model"],
              "adapter": str(lab_path(args.adapter).resolve()) if args.adapter else None,
              "adapter_reload_verified": bool(args.adapter), "active_adapters": active,
              "benchmark": str(benchmark.resolve()), "benchmark_sha256": file_hash(benchmark),
              "protocol_sha256": file_hash(protocol_path) if protocol_path.exists() else None,
              "evaluator_sha256": file_hash(Path(__file__)), "tokenizer": tokenizer_audit(tokenizer),
              "generation": {"do_sample": False, "max_new_tokens": args.max_new_tokens, "enable_thinking": False},
              "exact_normalization": "Unicode NFKC, collapse whitespace, remove terminal .!?, casefold unless metadata.case_sensitive; internal punctuation preserved",
              "results": []}
    if args.adapter:
        report["adapter_files_sha256"] = {path.name: file_hash(path) for path in lab_path(args.adapter).iterdir() if path.is_file()}
    write_json(report_path, report)
    with NvidiaMonitor(output.with_name(output.name + "_vram.csv")) as monitor:
        for row in rows:
            generated = raw_generate(model, tokenizer, row, args.max_new_tokens)
            metrics = lexical_scores(generated["response"], row.get("metadata", {}))
            nll = teacher_forced_nll(model, tokenizer, row, config["model"]["max_seq_length"])
            result = {"id": row["id"], "category": row.get("category"), "prompt": row.get("prompt"),
                      "metadata": row.get("metadata"), **generated, "lexical": metrics, "teacher_forced": nll}
            report["results"].append(result)
            write_json(report_path, report)
            print(json.dumps({"id": row["id"], "response": generated["response"], "exact": metrics["exact_normalized"], "nll": nll["mean_nll"]}, ensure_ascii=False), flush=True)
    gate_spec = protocol.get("one_fact_gate", {})
    gate_id = gate_spec.get("primary_exact_id", "diag_one_exact_01")
    gate_row = next((row for row in report["results"] if row["id"] == gate_id), None)
    report["one_fact_gate"] = {"item_id": gate_id, "applicable": gate_row is not None,
                               "passed": bool(gate_row and gate_row["lexical"]["exact_normalized"] is True and gate_row["ended_with_eos"] and not gate_row["unclosed_think"]),
                               "rule": "Designated exact training-question probe must emit only the complete normalized identifier and finish with EOS; unknown/style items are reported separately."}
    scored = [row for row in report["results"] if row["lexical"]["exact_normalized"] is not None]
    report.update(status="complete", nvidia=monitor.summary(),
                  lexical_summary={"n_with_frozen_target": len(scored),
                                   "exact_normalized_count": sum(row["lexical"]["exact_normalized"] for row in scored),
                                   "answer_inclusion_count": sum(row["lexical"]["answer_inclusion"] for row in scored)},
                  mean_teacher_forced_nll=sum(row["teacher_forced"]["total_nll"] for row in report["results"])/sum(row["teacher_forced"]["target_tokens"] for row in report["results"]))
    write_json(report_path, report)
    lines = ["# Local diagnostic evaluation", "", "Exact/inclusion diagnostics and teacher-forced NLL are distinct from semantic/style quality.",
             "", "```json", json.dumps({key: report[key] for key in ("generation", "lexical_summary", "one_fact_gate", "tokenizer")}, ensure_ascii=False, indent=2), "```", ""]
    for row in report["results"]:
        lines.extend([f"## {row['id']}", "", str(row["prompt"]), "", row["response"], "", "```json",
                      json.dumps({"lexical": row["lexical"], "NLL": row["teacher_forced"]["mean_nll"], "raw_decoded": row["raw_decoded"],
                                  "finish_reason": row["finish_reason"]}, ensure_ascii=False, indent=2), "```", ""])
    output.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
