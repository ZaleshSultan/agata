"""Shared lab utilities. Importing this module never initializes AGATA or CUDA."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import threading
import time

LAB = Path(__file__).resolve().parent


def lab_path(value):
    path = Path(value)
    return path if path.is_absolute() else LAB / path


def read_jsonl(path):
    with Path(path).open(encoding="utf-8-sig") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def read_config(path=None):
    import yaml
    return yaml.safe_load(lab_path(path or "config.yaml").read_text(encoding="utf-8"))


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def runtime_environment():
    """Settings affect this process only, never the user's global environment."""
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("WANDB_DISABLED", "true")
    os.environ.setdefault("UNSLOTH_DISABLE_STATISTICS", "1")
    cache = Path.home() / ".cache" / "agata-ft"
    for variable, subdir in (("UNSLOTH_COMPILE_LOCATION", "unsloth"),
                             ("TRITON_CACHE_DIR", "triton"),
                             ("TORCHINDUCTOR_CACHE_DIR", "torchinductor")):
        location = cache / subdir
        location.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault(variable, str(location))


def load_model(config, adapter=None, training=False):
    runtime_environment()
    # Unsloth must patch Transformers/TRL before either is imported.
    from unsloth import FastLanguageModel
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Run the lab environment check; CPU training is not used.")
    kwargs = dict(
        model_name=str(lab_path(adapter)) if adapter else config["model"]["name"],
        max_seq_length=config["model"]["max_seq_length"],
        dtype=None, load_in_4bit=True, device_map={"": 0}, trust_remote_code=False,
    )
    kwargs["revision"] = config["model"].get("revision", "main")
    if adapter:
        saved = json.loads((lab_path(adapter) / "adapter_config.json").read_text(encoding="utf-8"))
        if saved.get("base_model_name_or_path") != config["model"]["name"]:
            raise ValueError("Adapter and configured base model differ")
        if saved.get("revision") not in (None, config["model"]["revision"]):
            raise ValueError("Adapter and configured base revision differ")
    model, tokenizer = FastLanguageModel.from_pretrained(**kwargs)
    if not tokenizer.is_fast:
        raise RuntimeError("Assistant masking requires the Qwen fast tokenizer with offset mappings.")
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if not training:
        FastLanguageModel.for_inference(model)
    return model, tokenizer


def render_chat(tokenizer, row, generation=False):
    return tokenizer.apply_chat_template(
        row["messages"], tools=row.get("tools") or None, tokenize=False,
        add_generation_prompt=generation, enable_thinking=False,
    )


def encode_assistant_only(tokenizer, row, max_length):
    """Mask all user/system/tool tokens, including tool results between assistants.

    Native Qwen3 Jinja has no generation blocks in some revisions. Offset mappings
    let us keep that exact template and label only each assistant payload + EOS.
    Reject inserted special delimiters so source text cannot impersonate a role.
    Never silently truncate a conversation/tool argument or drop its answer.
    """
    source = json.dumps({"messages": row["messages"], "tools": row.get("tools")}, ensure_ascii=False)
    if "<|im_start|>" in source or "<|im_end|>" in source:
        raise ValueError("Reserved Qwen role delimiter in source messages")
    rendered = render_chat(tokenizer, row)
    spans = [(m.start(1), m.end()) for m in re.finditer(
        r"<\|im_start\|>assistant\n(.*?)<\|im_end\|>", rendered, re.DOTALL
    )]
    expected = sum(m["role"] == "assistant" for m in row["messages"])
    if len(spans) != expected or not spans:
        raise ValueError("Chat template assistant boundaries differ from expected Qwen3 format")
    encoded = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    ids = encoded["input_ids"]
    if len(ids) > max_length:
        raise OverflowError(f"{row.get('id', '?')}: {len(ids)} tokens exceeds {max_length}")
    labels = [token if end > start and any(start >= lo and end <= hi for lo, hi in spans)
              else -100 for token, (start, end) in zip(ids, encoded["offset_mapping"])]
    if not any(label != -100 for label in labels):
        raise ValueError("No supervised assistant tokens")
    return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels}


class LabelPreservingCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, features):
        import torch
        width = max(len(row["input_ids"]) for row in features)
        result = {"input_ids": [], "attention_mask": [], "labels": []}
        for row in features:
            padding = width - len(row["input_ids"])
            result["input_ids"].append(row["input_ids"] + [self.pad_token_id] * padding)
            result["attention_mask"].append(row["attention_mask"] + [0] * padding)
            result["labels"].append(row["labels"] + [-100] * padding)
        return {key: torch.tensor(value, dtype=torch.long) for key, value in result.items()}


def generate(model, tokenizer, row, max_new_tokens=128):
    import torch
    rendered = render_chat(tokenizer, row, generation=True)
    inputs = tokenizer(rendered, add_special_tokens=False, return_tensors="pt").to(model.device)
    start = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                use_cache=True, pad_token_id=tokenizer.pad_token_id,
                                eos_token_id=tokenizer.eos_token_id)
    tokens = output[0, inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(tokens, skip_special_tokens=True)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    # Diagnostic only: generation settings and returned text are unchanged.
    ended_with_eos = bool(len(tokens) and int(tokens[-1]) == tokenizer.eos_token_id)
    return {"response": text, "generated_tokens": len(tokens),
            "ended_with_eos": ended_with_eos,
            "hit_token_limit": len(tokens) >= max_new_tokens and not ended_with_eos,
            "finish_reason": "eos" if ended_with_eos else ("length" if len(tokens) >= max_new_tokens else "other"),
            "seconds": round(time.perf_counter() - start, 3)}


class NvidiaMonitor:
    """Sample NVIDIA driver VRAM while work runs; report unavailability explicitly."""
    def __init__(self, path):
        self.path = Path(path)
        self.samples = []
        self.error = None
        self.stop = threading.Event()

    def _sample(self):
        try:
            flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=timestamp,memory.used,memory.total,utilization.gpu,temperature.gpu",
                 "--format=csv,noheader,nounits"], capture_output=True, text=True,
                timeout=10, check=True, creationflags=flags,
            )
            self.samples.append(result.stdout.strip())
        except (OSError, subprocess.SubprocessError) as exc:
            self.error = str(exc)

    def _loop(self):
        while not self.stop.is_set():
            self._sample()
            self.stop.wait(1)

    def __enter__(self):
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join(timeout=12)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("timestamp,memory_used_mib,memory_total_mib,gpu_util_percent,temperature_c\n"
                             + "\n".join(self.samples) + "\n", encoding="utf-8")

    def summary(self):
        memory = []
        for row in self.samples:
            try:
                memory.append(float(row.split(",")[1]))
            except (ValueError, IndexError):
                pass
        return {"samples": len(self.samples), "peak_used_mib": max(memory) if memory else None,
                "error": self.error, "csv": str(self.path)}
