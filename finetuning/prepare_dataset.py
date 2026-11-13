"""Prepare auditable seed data without reading AGATA memory or running runtime tools.

Usage: python finetuning/prepare_dataset.py [--book reviewed_book.jsonl]
Run directories are preserved unless --overwrite is explicitly supplied. Generated data
is private/ignored; seed_data.py is the reproducible, non-private source.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import re
import os
import subprocess

from seed_data import (BENCHMARK_CONVERSATION, GENERAL_CHAT, PERSONALITY,
                       SYNTHETIC_FACTS, SYSTEM_PROMPT, tool_examples)

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
BOOK_LAB_MODE = (
    "Лабораторный режим проверки локального источника: в вопросах об этой книге не вызывай внешний поиск. "
    "Воспроизведи известное содержание с указанием источника и года, как позицию автора, а не современную норму. "
    "Если содержание не известно, прямо признай нехватку данных и не придумывай цитату или страницу."
)


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def normalize(text: str) -> str:
    return " ".join(re.findall(r"\w+", text.casefold()))


def runtime_tool_schemas() -> dict[str, dict]:
    """Read literal JSON schemas with AST; never import tools/__init__.py or SQLite."""
    schemas = {}
    for file in sorted((REPO / "tools").glob("*.py")):
        tree = ast.parse(file.read_text(encoding="utf-8-sig"))
        for node in tree.body:
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            try:
                value = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                continue
            if isinstance(value, dict) and isinstance(value.get("function"), dict):
                function = value["function"]
                if {"name", "parameters"} <= function.keys():
                    schemas[function["name"]] = {"type": "function", "function": function}
    return schemas


def record(key, category, messages, schemas, *, group_id=None, metadata=None):
    names = sorted({call["function"]["name"] for message in messages
                    for call in message.get("tool_calls", [])})
    row = {"id": key, "group_id": group_id or key, "category": category,
           "messages": [{"role": "system", "content": SYSTEM_PROMPT}] + messages,
           "tools": [schemas[name] for name in names],
           "metadata": {"source": "hand_authored_from_prompts.py", "private_data": False,
                        "tool_results_simulated": bool(names), **(metadata or {})}}
    return row


def dialogue(user, assistant):
    return [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]


def validate_record(row):
    if not row.get("id") or not row.get("group_id"):
        raise ValueError("Every row needs id and group_id")
    messages = row.get("messages", [])
    if len(messages) < 2 or messages[-1].get("role") != "assistant":
        raise ValueError(f"{row['id']}: transcript must end with assistant")
    schemas = {tool["function"]["name"]: tool["function"]["parameters"] for tool in row.get("tools", [])}
    pending = []
    for message in messages:
        role = message.get("role")
        if role not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"{row['id']}: invalid role {role}")
        content = message.get("content", "")
        if not isinstance(content, str) or any(tag in content for tag in ("<|im_start|>", "<|im_end|>")):
            raise ValueError(f"{row['id']}: content must be plain text without chat delimiters")
        calls = message.get("tool_calls", [])
        if calls and role != "assistant":
            raise ValueError("Only assistant may call tools")
        if role == "tool":
            if not pending:
                raise ValueError(f"{row['id']}: tool result without a call")
            expected = pending.pop(0)
            if message.get("name", expected) != expected:
                raise ValueError(f"{row['id']}: tool result name mismatch")
        elif pending:
            raise ValueError(f"{row['id']}: missing tool result before next message")
        for call in calls:
            function = call["function"]
            name, arguments = function["name"], function["arguments"]
            if name not in schemas or not isinstance(arguments, dict):
                raise ValueError(f"{row['id']}: missing schema or invalid arguments")
            schema = schemas[name]
            if not set(schema.get("required", [])).issubset(arguments):
                raise ValueError(f"{row['id']}: missing required arguments for {name}")
            if not set(arguments).issubset(schema.get("properties", {})):
                raise ValueError(f"{row['id']}: unexpected argument for {name}")
            for argument, value in arguments.items():
                prop = schema["properties"][argument]
                types = {"string": str, "integer": int, "number": (int, float), "boolean": bool, "object": dict, "array": list}
                if prop.get("type") in types and not isinstance(value, types[prop["type"]]):
                    raise ValueError(f"{row['id']}: wrong type for {argument}")
                if "enum" in prop and value not in prop["enum"]:
                    raise ValueError(f"{row['id']}: invalid value for {argument}")
            pending.append(name)
    if pending:
        raise ValueError(f"{row['id']}: unanswered tool call")


def chapter_group(row):
    metadata = row.get("metadata", {})
    chapter = metadata.get("chapter_id")
    source = metadata.get("source_id")
    if not chapter or not source:
        raise ValueError("Book rows require metadata.source_id and metadata.chapter_id")
    return f"book:{source}:{chapter}"


def book_records(path: Path | None):
    if path is None:
        return []
    from book_pipeline import complete_source_quote, normal, risk_categories
    result = []
    for index, raw in enumerate(read_jsonl(path)):
        row = dict(raw)
        meta = row.get("metadata", {})
        if meta.get("review_status") != "approved":
            raise ValueError(f"Book row {index}: review_status must be approved")
        if not meta.get("evidence"):
            raise ValueError(f"Book row {index}: source evidence is required")
        required = ("title", "chapter", "locator", "reviewer", "review_note", "training_mode")
        if any(not meta.get(field) for field in required) or not meta.get("title_verified"):
            raise ValueError(f"Book row {index}: verified bibliography and complete review provenance required")
        if meta["training_mode"] not in {"source_attributed", "historical"}:
            raise ValueError(f"Book row {index}: unknown training mode")
        final_answer = row["messages"][-1].get("content", "")
        date = str(meta.get("year") or "год не установлен")
        answer_quote = meta.get("answer_quote", meta["evidence"])
        if (meta["title"] not in final_answer or date not in final_answer
                or not complete_source_quote(answer_quote, meta["evidence"])
                or normal(answer_quote) not in normal(final_answer)):
            raise ValueError(f"Book row {index}: answer must retain title, date and exact reviewed evidence")
        checked_risks = sorted(set(meta.get("risk_categories", []) + risk_categories(
            " ".join(message.get("content", "") for message in row["messages"]) + " " + meta["chapter"])))
        meta["risk_categories"] = checked_risks
        if checked_risks or meta["training_mode"] == "historical":
            if (meta["training_mode"] != "historical" or not meta.get("year")
                    or "историческое утверждение источника" not in final_answer
                    or "не современная" not in final_answer):
                raise ValueError(f"Book row {index}: risky material requires dated historical attribution")
        row["id"] = row.get("id", f"book_{index:05d}")
        row["group_id"] = chapter_group(row)
        row["category"] = "book_knowledge"
        row.setdefault("tools", [])
        row["metadata"] = {**meta, "private_data": True}
        if row["messages"][0]["role"] != "system":
            row["messages"] = [{"role": "system", "content": SYSTEM_PROMPT}] + row["messages"]
        if BOOK_LAB_MODE not in row["messages"][0]["content"]:
            row["messages"][0]["content"] += "\n\n" + BOOK_LAB_MODE
        row["metadata"]["lab_source_recall_mode"] = True
        result.append(row)
    return result


def split_records(rows, seed=3407):
    """Whole groups, stratified by category; book chapters can request a partition."""
    groups = defaultdict(list)
    for row in rows:
        groups[row["group_id"]].append(row)
    assignments, strata = {}, defaultdict(list)
    for group, members in groups.items():
        requested = {row.get("metadata", {}).get("partition") for row in members} - {None}
        if len(requested) > 1:
            raise ValueError(f"Conflicting partitions within chapter/group {group}")
        if requested:
            partition = requested.pop()
            if partition not in {"train", "validation", "test"}:
                raise ValueError(f"Invalid partition {partition}")
            assignments[group] = partition
        else:
            categories = {row["category"] for row in members}
            if len(categories) != 1:
                raise ValueError(f"Group {group} spans categories")
            strata[categories.pop()].append(group)
    for category, category_groups in sorted(strata.items()):
        ordered = sorted(category_groups)
        random.Random(f"{seed}:{category}").shuffle(ordered)
        holdout = max(1, round(len(ordered) * .1)) if len(ordered) >= 3 else 0
        for index, group in enumerate(ordered):
            assignments[group] = "validation" if index < holdout else "test" if index < 2 * holdout else "train"
    splits = {partition: [] for partition in ("train", "validation", "test")}
    for row in rows:
        splits[assignments[row["group_id"]]].append(row)
    for partition, members in splits.items():
        random.Random(f"{seed}:{partition}").shuffle(members)
    assert_no_leakage(splits)
    return splits


def assert_no_leakage(splits):
    seen_groups, seen_prompts, seen_dialogues = {}, {}, {}
    ids = set()
    for partition, rows in splits.items():
        for row in rows:
            validate_record(row)
            if row["id"] in ids:
                raise ValueError(f"Duplicate id {row['id']}")
            ids.add(row["id"])
            group = row["group_id"]
            if group in seen_groups and seen_groups[group] != partition:
                raise ValueError(f"Group leakage: {group}")
            seen_groups[group] = partition
            fingerprint = json.dumps(row["messages"], ensure_ascii=False, sort_keys=True)
            if fingerprint in seen_dialogues:
                raise ValueError("Duplicate dialogue")
            seen_dialogues[fingerprint] = partition
            prompt = normalize(next(message["content"] for message in row["messages"] if message["role"] == "user"))
            if prompt in seen_prompts and seen_prompts[prompt] != partition:
                raise ValueError(f"Prompt leakage: {row['id']}")
            seen_prompts[prompt] = partition


def build_seed(schemas):
    datasets = {}
    for name, examples in (("personality", PERSONALITY), ("general_chat", GENERAL_CHAT)):
        datasets[name] = [record(key, name, dialogue(user, answer), schemas) for key, user, answer in examples]
    datasets["tool_calling"] = [record(key, "tool_calling", messages, schemas) for key, messages in tool_examples()]
    datasets["synthetic_knowledge"] = []
    benchmark = []
    for key, title, chapter, question, answer, paraphrase, keywords in SYNTHETIC_FACTS:
        metadata = {"fictional": True, "source_title": title, "chapter": chapter,
                    "partition": "train", "fact_id": key, "source": "authored_fictional_fixture"}
        datasets["synthetic_knowledge"].append(record(key, "synthetic_knowledge", dialogue(question, answer), schemas,
                                                     group_id=f"synthetic:{chapter}", metadata=metadata))
        for category, prompt in (("B", question), ("C", paraphrase)):
            benchmark.append(benchmark_row(f"{category}_{key}", category, prompt, answer, keywords,
                metadata={**metadata, "seen_question": category == "B", "training_fact_id": key,
                          "interpretation": "memorization" if category == "B" else "paraphrase_generalization"}))
    # An entire synthetic chapter is withheld; never infer that D measures acquired knowledge.
    heldout = [
        ("syn_holdout_bell", "Как называется колокол на башне Ширвель в вымышленной Нэлавии?", "Колокол называется «Лаор-86». Это факт из полностью исключённой из обучения главы.", ["лаор-86"]),
        ("syn_holdout_flag", "Какой знак на флаге острова Мерту в вымышленной Нэлавии?", "На флаге острова Мерту изображена янтарная спираль. Это вымышленный факт из исключённой главы.", ["янтарн", "спирал"]),
    ]
    datasets["synthetic_holdout"] = []
    for key, prompt, answer, keywords in heldout:
        metadata = {"fictional": True, "source_title": "Путеводитель вымышленной Нэлавии",
                    "chapter": "Запретный остров", "partition": "test", "excluded_entire_chapter": True}
        datasets["synthetic_holdout"].append(record(key, "synthetic_holdout", dialogue(prompt, answer), schemas,
                                                   group_id="synthetic:Запретный остров", metadata=metadata))
        benchmark.append(benchmark_row("D_" + key, "D", prompt, answer, keywords,
                                       metadata={**metadata, "interpretation": "unseen_chapter_control"}))
    for key, category, prompt, reference, keywords in BENCHMARK_CONVERSATION:
        benchmark.append(benchmark_row(key, category, prompt, reference, keywords,
                                       metadata={"manual_review_required": True}))
    return datasets, benchmark


def benchmark_row(key, category, prompt, reference, keywords, *, metadata=None):
    return {"id": key, "category": category, "prompt": prompt,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
            "tools": [], "reference": reference, "expected_keywords": keywords,
            "metadata": {"real_book_evidence": False, **(metadata or {})}}


def ensure_private_output_paths(paths):
    """Source-derived data inside this Git repo must be ignored and untracked."""
    for path in paths:
        resolved = Path(path).resolve()
        try:
            relative = resolved.relative_to(REPO.resolve()).as_posix()
        except ValueError:
            continue  # An explicitly selected external destination is outside this repository.
        options = {"capture_output": True, "text": True,
                   "creationflags": subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0}
        tracked = subprocess.run(["git", "-C", str(REPO), "ls-files", "--error-unmatch", "--", relative], **options)
        ignored = subprocess.run(["git", "-C", str(REPO), "check-ignore", "--quiet", "--no-index", "--", relative], **options)
        if tracked.returncode == 0 or ignored.returncode != 0:
            raise ValueError(f"Private output must be Git-ignored and untracked: {relative}")


def validate_benchmark(benchmark, splits):
    train = {normalize(next(m["content"] for m in row["messages"] if m["role"] == "user")): row for row in splits["train"]}
    if len({row["id"] for row in benchmark}) != len(benchmark):
        raise ValueError("Benchmark IDs must be unique")
    training_fact_ids = {row["id"] for row in splits["train"]} | {
        row.get("metadata", {}).get("fact_id") for row in splits["train"]}
    training_chapters = {
        (meta.get("source_id") or meta.get("source_title"), meta.get("chapter_id") or meta.get("chapter"))
        for meta in (row.get("metadata", {}) for row in splits["train"])
        if meta.get("chapter_id") or meta.get("chapter")}
    for row in benchmark:
        overlap = normalize(row["prompt"]) in train
        if overlap and not (row["category"] == "B" and row["metadata"].get("seen_question")):
            raise ValueError(f"Unintended benchmark leakage: {row['id']}")
        if row["category"] == "B" and not overlap:
            raise ValueError(f"Memorization item absent from training: {row['id']}")
        if row["category"] in {"B", "C"} and row["metadata"].get("training_fact_id") not in training_fact_ids - {None}:
            raise ValueError(f"Known-fact benchmark points to absent training fact: {row['id']}")
        if row["category"] == "D":
            meta = row["metadata"]
            chapter = (meta.get("source_id") or meta.get("source_title"), meta.get("chapter_id") or meta.get("chapter"))
            if not all(chapter) or chapter in training_chapters:
                raise ValueError(f"Excluded benchmark chapter is missing or leaks into train: {row['id']}")
    if {row["category"] for row in benchmark} != set("ABCDEF"):
        raise ValueError("Benchmark must cover A-F")


def prepare(output_dir=ROOT / "datasets", benchmark_path=ROOT / "evaluation" / "benchmark.jsonl",
            book=None, seed=3407, overwrite=False, extra_benchmark=None):
    output_dir, benchmark_path = Path(output_dir), Path(benchmark_path)
    schemas = runtime_tool_schemas()
    datasets, benchmark = build_seed(schemas)
    datasets["book_knowledge"] = book_records(Path(book) if book else None)
    if extra_benchmark:
        extra_rows = read_jsonl(Path(extra_benchmark))
        for row in extra_rows:
            if row.get("metadata", {}).get("real_book_evidence"):
                messages = row.setdefault("messages", [{"role": "user", "content": row["prompt"]}])
                if messages[0]["role"] != "system":
                    messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
                if BOOK_LAB_MODE not in messages[0]["content"]:
                    messages[0]["content"] += "\n\n" + BOOK_LAB_MODE
                row["metadata"]["lab_source_recall_mode"] = True
        benchmark.extend(extra_rows)
    rows = [row for members in datasets.values() for row in members]
    splits = split_records(rows, seed)
    validate_benchmark(benchmark, splits)
    # Representative tiny subset; all smoke items also belong to train, never validation/test.
    smoke = []
    for category in ("personality", "general_chat", "tool_calling", "synthetic_knowledge", "book_knowledge"):
        available = [row for row in splits["train"] if row["category"] == category]
        cap = 6 if category == "synthetic_knowledge" else 8 if category == "book_knowledge" else 2
        smoke.extend(available[:cap])
    all_sets = {**datasets, **splits, "smoke": smoke}
    targets = [output_dir / f"{name}.jsonl" for name in all_sets] + [output_dir / "manifest.json", benchmark_path]
    if book or extra_benchmark:
        ensure_private_output_paths(targets)
    if not overwrite:
        existing = [str(path) for path in targets if path.exists()]
        if existing:
            raise FileExistsError("Refusing to replace existing data. Choose --output-dir/--benchmark or --overwrite: " + ", ".join(existing))
    for name, members in all_sets.items():
        write_jsonl(output_dir / f"{name}.jsonl", members)
    write_jsonl(benchmark_path, benchmark)
    manifest = {
        "seed": seed, "counts": {name: len(members) for name, members in all_sets.items()},
        "benchmark_counts": dict(sorted(Counter(row["category"] for row in benchmark).items())),
        "prompts_sha256": hashlib.sha256((REPO / "prompts.py").read_bytes()).hexdigest(),
        "seed_sha256": hashlib.sha256((ROOT / "seed_data.py").read_bytes()).hexdigest(),
        "system_prompt": "Condensed from prompts.py for 1024-token budget; full runtime prompt unchanged",
        "tool_schemas_source": "Literal AST extraction from tools/*.py; tools never executed",
        "book_source": str(book) if book else None,
        "book_sha256": hashlib.sha256(Path(book).read_bytes()).hexdigest() if book else None,
        "extra_benchmark_source": str(extra_benchmark) if extra_benchmark else None,
        "extra_benchmark_sha256": hashlib.sha256(Path(extra_benchmark).read_bytes()).hexdigest() if extra_benchmark else None,
        "split_strategy": "category-stratified deterministic whole groups; books grouped by source+chapter",
        "known_exposure": "B repeats full-train questions intentionally; C questions unseen, facts in full train; D entire chapters excluded; actual adapter exposure recorded by trainer",
        "limitations": ["Small curated bootstrap corpus; insufficient for production-quality personality.",
                       "Synthetic facts are invented; no guarantee the base model cannot guess them.",
                       "Benchmark is a controlled pilot, not proof of reliable knowledge acquisition.",
                       "Real book claims require reviewed source-backed book data and a separate book benchmark.",
                       "Automatic overlap checks detect exact normalized prompts and groups, not all semantic paraphrases."],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "datasets")
    parser.add_argument("--benchmark", type=Path, default=ROOT / "evaluation" / "benchmark.jsonl")
    parser.add_argument("--book", type=Path, help="Reviewed private book_knowledge.jsonl, metadata.review_status=approved")
    parser.add_argument("--extra-benchmark", type=Path, help="Reviewed source-backed benchmark to combine with synthetic pilot")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--overwrite", action="store_true", help="Explicitly replace these generated files only")
    args = parser.parse_args()
    print(json.dumps(prepare(args.output_dir, args.benchmark, args.book, args.seed, args.overwrite,
                             extra_benchmark=args.extra_benchmark), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
