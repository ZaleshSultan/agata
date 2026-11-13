"""Separate, explicit export steps. Never overwrite the running AGATA model."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from urllib.request import Request, urlopen

from common import lab_path, load_model, read_config, write_json


def api(route, payload=None, base="http://localhost:11434"):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    with urlopen(Request(base + route, data=data, headers={"Content-Type": "application/json"}), timeout=180) as response:
        return json.load(response)


def validate_test_name(name):
    if not re.fullmatch(r"agata-finetuned-test(?:-[a-z0-9-]+)?", name):
        raise ValueError("Only agata-finetuned-test or agata-finetuned-test-<suffix> is allowed")
    return name


def new_directory(value):
    path = lab_path(value).resolve()
    if path.exists() and any(path.iterdir()):
        raise ValueError(f"Output is not empty: {path}. Preserve it and choose a new directory.")
    path.mkdir(parents=True, exist_ok=True)
    return path


def merge(args):
    config = read_config(args.config)
    output = new_directory(args.output_dir)
    if shutil.disk_usage(output).free < 10 * 2**30:
        raise RuntimeError("Export needs at least 10 GiB free disk for merged/F16/quantized intermediates")
    model, tokenizer = load_model(config, adapter=args.adapter, training=True)
    # Supported Unsloth merge API; the separate conversion step uses existing tools.
    model.save_pretrained_merged(str(output), tokenizer, save_method="merged_16bit", maximum_memory_usage=0.5)
    write_json(output / "agata_export_manifest.json", {"base_model": config["model"], "adapter": args.adapter,
                                                       "chat_template": tokenizer.chat_template,
                                                       "status": "merged_only_not_ollama_validated"})


def convert(args):
    merged = lab_path(args.merged_dir).resolve()
    converter = Path(args.llama_cpp_dir).resolve() / "convert_hf_to_gguf.py"
    quantizer = Path(args.quantizer).resolve()
    if not converter.is_file() or not quantizer.is_file():
        raise ValueError("Provide an existing llama.cpp converter and llama-quantize executable; no system installation is attempted")
    output = new_directory(args.output_dir)
    f16 = output / "agata-f16.gguf"
    q4 = output / "agata-q4_k_m.gguf"
    converter_python_path = Path(args.converter_python) if args.converter_python else Path.home() / ".cache/agata-ft/converter-env/Scripts/python.exe"
    if not converter_python_path.is_file():
        raise ValueError("Run setup_export.py --install-converter or provide --converter-python from a compatible isolated environment")
    converter_python = str(converter_python_path.resolve())
    subprocess.run([converter_python, str(converter), str(merged), "--outfile", str(f16), "--outtype", "f16", "--use-temp-file"], check=True)
    subprocess.run([str(quantizer), str(f16), str(q4), "Q4_K_M"], check=True)
    if not q4.is_file() or q4.stat().st_size < 1_000_000:
        raise RuntimeError("Quantization did not produce a plausible GGUF")
    print(q4)


def register(args):
    name = validate_test_name(args.name)
    gguf = lab_path(args.gguf).resolve()
    if not gguf.is_file() or gguf.suffix.lower() != ".gguf":
        raise ValueError("An existing GGUF is required")
    existing = {row["name"].split(":")[0] for row in api("/api/tags")["models"]}
    if name in existing:
        raise ValueError(f"Ollama model {name} already exists. Use a new test suffix.")
    output = new_directory(args.output_dir)
    modelfile = output / "Modelfile.test"
    # Ollama detects Qwen3 and its tool template from GGUF metadata. Verification
    # below is mandatory before considering this an AGATA-compatible artifact.
    modelfile.write_text(f'FROM "{gguf.as_posix()}"\nPARAMETER num_ctx 1024\nPARAMETER temperature 0\n', encoding="utf-8")
    subprocess.run(["ollama", "create", name, "-f", str(modelfile)], check=True)
    write_json(output / "registration.json", {"name": name, "gguf": str(gguf),
                                              "status": "created_not_behaviorally_verified"})
    print(f"Created {name}. Run the verify subcommand; the working agata model is unchanged.")


def verify(args):
    name = validate_test_name(args.name)
    show = api("/api/show", {"model": name})
    tool = {"type": "function", "function": {"name": "get_current_datetime",
            "description": "Получить текущую дату и время.", "parameters": {"type": "object", "properties": {}}}}
    cases = [
        {"id": "russian", "messages": [{"role": "user", "content": "Ответь по-русски: как тебя зовут?"}]},
        {"id": "tool_call", "messages": [{"role": "user", "content": "Который сейчас час? Вызови get_current_datetime."}], "tools": [tool]},
        {"id": "failed_action", "messages": [
            {"role": "user", "content": "Узнай текущее время. При ошибке сообщи об ошибке без повторной попытки."},
            {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": "get_current_datetime", "arguments": {}}}]},
            {"role": "tool", "content": '{"success": false, "error": "Источник времени недоступен"}'}], "tools": [tool]},
    ]
    results = []
    system = {"role": "system", "content": "Ты — Агата, русскоязычный голосовой ассистент. Отвечай кратко, без Markdown. Никогда не утверждай успех при ошибке инструмента."}
    for case in cases:
        payload = {"model": name, "stream": False, "think": False, "keep_alive": 0,
                   "messages": [system] + case["messages"], "options": {"temperature": 0, "num_predict": 128}}
        if case.get("tools"):
            payload["tools"] = case["tools"]
        result = api("/api/chat", payload)
        message = result.get("message", {})
        content = message.get("content", "")
        if case["id"] == "tool_call":
            passed = any(item.get("function", {}).get("name") == "get_current_datetime" for item in message.get("tool_calls", []))
        elif case["id"] == "russian":
            passed = bool(re.search(r"[А-Яа-яЁё]", content))
        else:
            passed = bool(re.search(r"не (?:удалось|могу)|недоступ|ошибк", content, re.I))
        results.append({"id": case["id"], "message": message, "mechanical_check_passed": passed})
    report = {"name": name, "status": "passed_mechanical_checks" if all(r["mechanical_check_passed"] for r in results) else "failed",
              "manual_review_required": True, "template": show.get("template"), "results": results,
              "note": "No proposed tools were executed. Runtime compatibility still requires reviewing all responses."}
    write_json(lab_path(args.output), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["status"] == "failed":
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="operation", required=True)
    p = sub.add_parser("merge", help="Merge adapter into FP16; separate RAM/disk intensive operation")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--adapter", required=True)
    p.add_argument("--output-dir", default="outputs/export-merged")
    p.set_defaults(run=merge)
    p = sub.add_parser("convert", help="Convert/quantize with an existing local llama.cpp installation")
    p.add_argument("--merged-dir", required=True)
    p.add_argument("--llama-cpp-dir", required=True)
    p.add_argument("--quantizer", required=True)
    p.add_argument("--converter-python", help="Python from the isolated CPU conversion venv")
    p.add_argument("--output-dir", default="outputs/export-gguf")
    p.set_defaults(run=convert)
    p = sub.add_parser("register", help="Create a NEW test-only Ollama model")
    p.add_argument("--gguf", required=True)
    p.add_argument("--name", default="agata-finetuned-test")
    p.add_argument("--output-dir", default="outputs/ollama-test")
    p.set_defaults(run=register)
    p = sub.add_parser("verify", help="Test Russian answers and tool parsing without executing actions")
    p.add_argument("--name", default="agata-finetuned-test")
    p.add_argument("--output", default="evaluation/reports/ollama-test.json")
    p.set_defaults(run=verify)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
