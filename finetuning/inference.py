"""Generate Russian text with the base model or a saved LoRA in a fresh process."""
import argparse
import json
import re

from common import generate, lab_path, load_model, read_config, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--adapter")
    parser.add_argument("--prompt", default="Представься по-русски в двух коротких предложениях.")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--output", default="outputs/inference.json")
    args = parser.parse_args()
    if lab_path(args.output).exists():
        raise SystemExit(f"Report exists: {lab_path(args.output)}. Choose a new --output path.")
    config = read_config(args.config)
    model, tokenizer = load_model(config, adapter=args.adapter)
    probe = "Агата, здравствуйте! Ёжик и съёмка — русский текст."
    roundtrip = tokenizer.decode(tokenizer.encode(probe, add_special_tokens=False))
    if roundtrip != probe:
        raise RuntimeError("Russian tokenizer roundtrip failed")
    row = {"messages": [{"role": "system", "content": "Ты — Агата, русскоязычный голосовой ассистент. Отвечай кратко, естественно и честно."},
                        {"role": "user", "content": args.prompt}]}
    result = generate(model, tokenizer, row, args.max_new_tokens)
    result.update({"base_model": config["model"], "adapter": args.adapter,
                   "russian_tokenizer_roundtrip": True,
                   "contains_cyrillic": bool(re.search(r"[А-Яа-яЁё]", result["response"])),
                   "device": str(model.device), "prompt": args.prompt,
                   "adapter_reload_verified": bool(args.adapter)})
    if args.adapter:
        from peft import PeftModel
        if not isinstance(model, PeftModel):
            raise RuntimeError("Expected a PEFT model when reloading the adapter")
        result["active_adapters"] = model.active_adapters
    write_json(lab_path(args.output), result)
    if args.adapter:
        training_report = lab_path(args.adapter).resolve().parent / "training_report.json"
        if training_report.is_file():
            saved = json.loads(training_report.read_text(encoding="utf-8"))
            if saved.get("status") == "passed" and lab_path(saved["adapter"]).resolve() == lab_path(args.adapter).resolve():
                saved["adapter_reload_verified"] = True
                saved["adapter_reload_evidence"] = str(lab_path(args.output).resolve())
                write_json(training_report, saved)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
