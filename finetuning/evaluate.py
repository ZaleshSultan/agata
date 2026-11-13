"""A-F benchmark: retain raw answers; lexical indicators are not a truth oracle."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import re
import time

from common import file_hash, generate, lab_path, load_model, read_config, read_jsonl, write_json

CATEGORIES = {
    "A": "Личность AGATA", "B": "Запоминание обучающих фактов",
    "C": "Перефразированные вопросы", "D": "Исключённые главы",
    "E": "Неизвестные факты и неопределённость", "F": "Русский язык и общение",
}
ABSTENTION = re.compile(r"не знаю|неизвестн|нет (?:достаточн\w* )?(?:данных|информации|сведений)|"
                        r"не могу (?:подтвердить|знать|определить|достоверно)|не (?:указан|сообщён|сообщен)|"
                        r"недостаточно (?:данных|информации)|не уверен", re.I)


def assess(row, response):
    words = row.get("expected_keywords") or []
    normalized = response.casefold().replace("ё", "е")
    hits = [word for word in words if str(word).casefold().replace("ё", "е") in normalized]
    abstains = bool(ABSTENTION.search(response))
    return {
        "keyword_recall": len(hits) / len(words) if words else None,
        "keyword_hits": hits, "abstention_phrase": abstains,
        "unsupported_answer_candidate": row["category"] == "E" and not abstains,
        "characters": len(response), "contains_cyrillic": bool(re.search(r"[А-Яа-яЁё]", response)),
        "markdown_candidate": bool(re.search(r"\*\*|```|^#+\s", response, re.M)),
        "manual_review_required": True,
    }


def summarize(results):
    groups = defaultdict(list)
    for row in results:
        groups[row["category"]].append(row["assessment"])
    summary = {}
    for key in CATEGORIES:
        rows = groups[key]
        recalls = [r["keyword_recall"] for r in rows if r["keyword_recall"] is not None]
        summary[key] = {"name": CATEGORIES[key], "n": len(rows),
                        "mean_keyword_recall": sum(recalls) / len(recalls) if recalls else None,
                        "abstention_phrase_count": sum(r["abstention_phrase"] for r in rows),
                        "unsupported_answer_candidates": sum(r["unsupported_answer_candidate"] for r in rows)}
    return summary


def source_kind(row):
    metadata = row.get("metadata") or {}
    if metadata.get("real_book_evidence"):
        return "real_book"
    if metadata.get("fictional"):
        return "fictional"
    return "general"


def exposure(row, training_report):
    fact = row.get("metadata", {}).get("training_fact_id")
    if not fact:
        return "not_a_training_fact_probe"
    if training_report is None:
        return "base_model_no_adapter"
    accepted = training_report.get("dataset", {}).get("accepted_examples", [])
    ids = {item.get("id") for item in accepted} | {item.get("fact_id") for item in accepted}
    if not accepted:
        return "unknown_missing_training_manifest"
    if fact not in ids:
        return "not_in_adapter_training_dataset"
    if training_report.get("dataset_full_pass_verified"):
        return "confirmed_full_training_pass"
    return "eligible_but_actual_exposure_unverified"


def markdown(report):
    lines = ["# AGATA: экспериментальная проверка", "", f"Модель: `{report['base_model']['name']}`.",
             f"Адаптер: `{report.get('adapter') or 'нет (базовая модель)'}`.", "",
             "Результаты — диагностические индикаторы. Совпадение ключевых слов не проверяет истинность, отрицания или смысл. "
             "Все ответы требуют содержательной оценки; короткий smoke-test не доказывает приобретение знаний.", "",
             "| Категория | N | Доля найденных ключей | Признаки неопределённости | Кандидаты неподтверждённых ответов |",
             "|---|---:|---:|---:|---:|"]
    for category, item in report["summary"].items():
        score = item["mean_keyword_recall"]
        lines.append(f"| {category}: {item['name']} | {item['n']} | {score if score is not None else 'ручная оценка'} | "
                     f"{item['abstention_phrase_count']} | {item['unsupported_answer_candidates']} |")
    lines += ["", "B измеряет воспроизведение известных обучению фактов; C — перенос на другие формулировки. "
              "D полностью исключён из обучения; E проверяет отказ от выдумывания. "
              "Синтетические факты явно вымышлены: их оценки относятся только к искусственному эксперименту. "
              "Усвоение реальной книги оценивается отдельно по её собственным проверенным вопросам и источникам.", ""]
    lines += ["Принадлежность B/C к обучающему набору сама по себе не доказывает, что короткий запуск видел факт. "
              "Поле actual_training_exposure подтверждает полный проход именно по фактам сохранённого адаптера.", ""]
    for kind, summary in report.get("summary_by_source", {}).items():
        lines += [f"### Отдельная группа: {kind}", "", "```json", json.dumps(summary, ensure_ascii=False, indent=2), "```", ""]
    for row in report["results"]:
        lines += [f"## {row['id']} · {row['category']}", "", row["prompt"], "",
                  "Ответ: " + row["response"], "", "Эталон: " + str(row.get("reference", "")), "",
                  "Воздействие обучения: " + row.get("actual_training_exposure", "unknown"), "",
                  "Метаданные: `" + json.dumps(row.get("metadata", {}), ensure_ascii=False) + "`", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--adapter")
    parser.add_argument("--benchmark")
    parser.add_argument("--output", required=True, help="Report path stem relative to finetuning")
    parser.add_argument("--limit", type=int, help="Explicit diagnostic subset; report records the limit")
    parser.add_argument("--max-new-tokens", type=int)
    args = parser.parse_args()
    config = read_config(args.config)
    benchmark = lab_path(args.benchmark or config["evaluation"]["benchmark"])
    rows = read_jsonl(benchmark)
    if args.limit:
        rows = rows[:args.limit]
    output = lab_path(args.output)
    json_path = output.with_suffix(".json")
    if json_path.exists() or output.with_suffix(".md").exists():
        raise SystemExit(f"Report exists: {json_path}. Choose a new output stem.")
    model, tokenizer = load_model(config, adapter=args.adapter)
    training_report = None
    if args.adapter:
        training_path = lab_path(args.adapter).resolve().parent / "training_report.json"
        if training_path.is_file():
            training_report = json.loads(training_path.read_text(encoding="utf-8"))
        else:
            training_report = {}
    max_new = args.max_new_tokens or config["evaluation"]["max_new_tokens"]
    report = {"status": "running", "base_model": config["model"], "adapter": args.adapter,
              "benchmark_sha256": file_hash(benchmark), "benchmark": str(benchmark),
              "generation": {"do_sample": False, "max_new_tokens": max_new, "enable_thinking": False},
              "limit": args.limit, "created_unix": time.time(), "results": []}
    for index, row in enumerate(rows, 1):
        generated = generate(model, tokenizer, row, max_new)
        result = {key: row.get(key) for key in ("id", "category", "prompt", "reference", "metadata")}
        result.update(generated)
        result["source_kind"] = source_kind(row)
        result["actual_training_exposure"] = exposure(row, training_report)
        result["assessment"] = assess(row, generated["response"])
        report["results"].append(result)
        write_json(json_path, report)
        print(f"{index}/{len(rows)} {row['id']}: {generated['response'][:120]}", flush=True)
    report["status"] = "complete"
    report["summary"] = summarize(report["results"])
    report["summary_by_source"] = {kind: summarize([row for row in report["results"] if row["source_kind"] == kind])
                                    for kind in ("real_book", "fictional", "general")}
    report["summary_by_training_exposure"] = {
        kind: summarize([row for row in report["results"] if row["actual_training_exposure"] == kind])
        for kind in sorted({row["actual_training_exposure"] for row in report["results"]})}
    write_json(json_path, report)
    output.with_suffix(".md").write_text(markdown(report), encoding="utf-8")


if __name__ == "__main__":
    main()
