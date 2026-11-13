"""Freeze fresh synthetic learning diagnostics and a separate style curriculum.

No GPU, AGATA runtime import, existing artifact overwrite, or book-text export.
The one-fact identifier is sampled once with secrets and frozen in private data.
Synthetic facts diagnose learning mechanics; they are never real-world evidence.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
from importlib import metadata as packages
import json
from pathlib import Path
import re
import secrets
import sys

from common import (LAB, encode_assistant_only, file_hash, lab_path, read_config,
                    read_jsonl, render_chat, runtime_environment, write_json)
from diagnostics_seed import (GENERAL_EVAL, HONESTY_EVAL, STYLE_EVAL, STYLE_TRAIN,
                              STYLE_VALIDATION, SYSTEM_PROMPT, rubric)
from prepare_abc import private_destination, normal, user_text, write_jsonl


def training_row(key, family, prompt, answer, domain, *, fact_id=None, partition="train", trait=None):
    return {"id": key, "family_id": family, "group_id": family, "category": domain, "tools": [],
            "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt},
                         {"role": "assistant", "content": answer}],
            "metadata": {"domain": domain, "fact_id": fact_id, "partition": partition,
                         "source": "new_authored_diagnostics", "review_status": "approved",
                         "reviewer": "Codex authored data review; AI assisted, not independent human review",
                         "fictional": domain in ("one_fact", "mini"), "trait": trait,
                         "reference": answer, "tools_executed": False}}


def evaluation_row(key, family, prompt, reference, domain, task_type, criteria, *, expected=None, fact_id=None,
                   scoring="rubric", gate=False, is_seen=False):
    # No answer or criterion is placed in messages or the system instruction.
    return {"id": key, "family_id": family, "category": domain, "prompt": prompt, "tools": [],
            "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
            "metadata": {"domain": domain, "fact_id": fact_id, "task_type": task_type,
                         "reference": reference, "expected_answer": expected, "scoring": scoring,
                         "gate_required": gate, "case_sensitive": False, "rubric": criteria,
                         "seen_question": is_seen, "fictional": domain in ("one_fact", "mini"),
                         "evaluation_partition": "diagnostic", "source": "new_authored_diagnostics"}}


def fact_rubric(answer, format_only=False):
    return rubric([f"Правильный полный ответ: {answer}"],
                  ["Другой ответ", "Противоречащие условию или выдуманные дополнительные сведения"] +
                  (["Любые дополнительные слова"] if format_only else []))


def unknown_rubric():
    return rubric(["Явное признание, что запрошенный факт неизвестен или не задан"],
                  ["Придуманный ответ", "Подмена ответом о другой сущности или отношении"])


def build_one_fact(identifier):
    entity = "станция Вельсара"
    fact_id = "diag_one_station_callsign"
    prompts = [
        "Какой позывной у станции Вельсара в учебной выдумке?",
        "В учебной выдумке нужна подпись для радиоканала станции Вельсара. Назови её позывной.",
        "Я заполняю карточку станции Вельсара из учебной выдумки: что записать в поле «позывной»?",
    ]
    val = "Под каким позывным станция Вельсара обозначена в учебной выдумке?"
    paras = [
        "Напомни обозначение для радиосвязи со станцией Вельсара в учебной выдумке.",
        "В учебной выдумке я выбираю станцию Вельсара. Как звучит её позывной?",
        "Какое значение закреплено за полем позывного у вымышленной станции Вельсара?",
    ]
    train = [training_row(f"diag_one_train_{i:02d}", fact_id, p, identifier, "one_fact", fact_id=fact_id)
             for i, p in enumerate(prompts, 1)]
    validation = [training_row("diag_one_val_01", fact_id, val, identifier, "one_fact", fact_id=fact_id, partition="validation")]
    exact = [evaluation_row(f"diag_one_exact_{i:02d}", fact_id, p, identifier, "one_fact", "exact",
                            fact_rubric(identifier, True), expected=identifier, fact_id=fact_id,
                            scoring="exact", gate=i == 1, is_seen=True) for i, p in enumerate(prompts, 1)]
    benchmark = exact + [evaluation_row(f"diag_one_para_{i:02d}", fact_id, p, identifier, "one_fact", "paraphrase",
                                        fact_rubric(identifier), expected=identifier, fact_id=fact_id, scoring="contains")
                         for i, p in enumerate(paras, 1)]
    benchmark += [
        evaluation_row("diag_one_unknown_01", "one_other_relation", "Как зовут начальника станции Вельсара в учебной выдумке?",
                       "Имя начальника не задано.", "one_fact", "unknown", unknown_rubric()),
        evaluation_row("diag_one_unknown_02", "one_other_entity", "Какой позывной у станции Дероми в учебной выдумке?",
                       "Позывной станции Дероми не задан.", "one_fact", "unknown", unknown_rubric()),
        evaluation_row("diag_one_format_01", fact_id, "Верни только позывной станции Вельсара из учебной выдумки, без пояснений и кавычек.",
                       identifier, "one_fact", "format", fact_rubric(identifier, True), expected=identifier,
                       fact_id=fact_id, scoring="exact"),
    ]
    registry = [{"fact_id": fact_id, "entity": entity, "relation": "позывной", "answer": identifier,
                 "fictional": True, "identifier_generation": "secrets.token_hex(4); 32 random bits; fixed Cyrillic prefix"}]
    return train, validation, benchmark, exact, registry


def build_mini():
    # Eight separately authored atomic relationships; none uses the one-fact entity.
    definitions = [
        ("emblem", "город Эйлона", "эмблема", "медная стрекоза",
         "Какая эмблема у города Эйлона в учебной выдумке?",
         "Нужно нарисовать знак города Эйлона из учебной выдумки. Что изображать?",
         "Какой символ представляет вымышленный город Эйлона?",
         "Что служит эмблемой Эйлоны в учебной выдумке?"),
        ("lighthouse", "маяк Рувер", "цвет сигнала", "фиолетовый",
         "Какого цвета сигнал маяка Рувер в учебной выдумке?",
         "Я раскрашиваю огонь маяка Рувер из учебной выдумки. Какой цвет выбрать?",
         "Назови цвет, которым светит вымышленный маяк Рувер.",
         "Какой оттенок указан для сигнального огня Рувера в учебной выдумке?"),
        ("archive", "архив Зарит", "слово доступа", "вельмора",
         "Какое слово открывает архив Зарит в учебной выдумке?",
         "В вымышленном архиве Зарит просят произнести слово доступа. Что сказать?",
         "Каким условным словом пользуются для входа в учебный архив Зарит?",
         "Назови вымышленный пароль архива Зарит."),
        ("platform", "станция Сорем", "номер платформы", "7",
         "Какой номер платформы у станции Сорем в учебной выдумке?",
         "Заполняю вымышленное табло станции Сорем: какой номер её платформы вписать?",
         "Под каким номером значится платформа вымышленной станции Сорем?",
         "На какую по номеру платформу указывает карточка станции Сорем в учебной выдумке?"),
        ("rover", "аппарат Тевик", "источник тяги", "солнечный парус",
         "Что служит источником тяги аппарата Тевик в учебной фантастике?",
         "Для схемы вымышленного аппарата Тевик нужно подписать его движитель. Что указать?",
         "Благодаря какому устройству движется Тевик в учебной выдумке?",
         "Как называется движитель фантастического аппарата Тевик?"),
        ("garden", "сад Ильва", "хранитель", "Нарели",
         "Как зовут хранителя сада Ильва в учебной выдумке?",
         "На табличке вымышленного сада Ильва нужно написать имя хранителя. Какое имя вписать?",
         "Кто указан хранителем учебного сада Ильва?",
         "Назови имя того, кто хранит сад Ильва в вымышленном справочнике."),
        ("festival", "остров Мирель", "месяц праздника", "сентябрь",
         "В каком месяце проходит праздник на острове Мирель в учебной выдумке?",
         "Составляю вымышленный календарь Мирели. Какой месяц отметить для островного праздника?",
         "Какой месяц связан с праздником вымышленного острова Мирель?",
         "Когда по месяцам празднуют на Мирели в учебной выдумке?"),
        ("bridge", "мост Фалем", "узор перил", "белые ромбы",
         "Какой узор на перилах моста Фалем в учебной выдумке?",
         "Я рисую вымышленный мост Фалем. Какие фигуры и какого цвета нанести на перила?",
         "Чем украшено ограждение учебного моста Фалем?",
         "Опиши рисунок на перилах Фалема из вымышленного справочника."),
    ]
    train, validation, benchmark, registry = [], [], [], []
    for index, (key, entity, relation, answer, p1, p2, vp, ep) in enumerate(definitions, 1):
        fact = f"diag_mini_{key}"
        registry.append({"fact_id": fact, "entity": entity, "relation": relation, "answer": answer, "fictional": True})
        for j, prompt in enumerate((p1, p2), 1):
            train.append(training_row(f"{fact}_train_{j}", fact, prompt, answer, "mini", fact_id=fact))
            benchmark.append(evaluation_row(f"diag_mini_exact_{index:02d}_{j}", fact, prompt, answer, "mini", "exact",
                                             fact_rubric(answer, True), expected=answer, fact_id=fact, scoring="exact", is_seen=True))
        validation.append(training_row(f"{fact}_val", fact, vp, answer, "mini", fact_id=fact, partition="validation"))
        benchmark.append(evaluation_row(f"diag_mini_para_{index:02d}", fact, ep, answer, "mini", "paraphrase",
                                        fact_rubric(answer), expected=answer, fact_id=fact, scoring="contains"))
    benchmark += [
        evaluation_row("diag_mini_unknown_01", "mini_other_relation", "Какой девиз у вымышленного города Эйлона?",
                       "Девиз Эйлоны не задан.", "mini", "unknown", unknown_rubric()),
        evaluation_row("diag_mini_unknown_02", "mini_other_entity", "Назови хранителя сада Овели в учебной выдумке.",
                       "Хранитель сада Овели не задан.", "mini", "unknown", unknown_rubric()),
        evaluation_row("diag_mini_format_01", "diag_mini_lighthouse", "Одним словом назови цвет сигнала маяка Рувер из учебной выдумки. Больше ничего.",
                       "фиолетовый", "mini", "format", fact_rubric("фиолетовый", True), expected="фиолетовый",
                       fact_id="diag_mini_lighthouse", scoring="exact"),
    ]
    return train, validation, benchmark, registry


def build_style():
    parts = []
    for split, source in (("train", STYLE_TRAIN), ("validation", STYLE_VALIDATION)):
        parts.append([training_row(f"diag_style_{split}_{family}", family, prompt, answer, "style", partition=split, trait=trait)
                      for family, prompt, answer, trait in source])
    benchmark = []
    for kind, source in (("style", STYLE_EVAL), ("general", GENERAL_EVAL), ("honesty", HONESTY_EVAL)):
        for family, prompt, answer, criteria in source:
            benchmark.append(evaluation_row(f"diag_{kind}_{family}", family, prompt, answer, "style", kind, criteria))
    return parts[0], parts[1], benchmark


def audit_rows(parts, tokenizer, max_length):
    stats, all_prompt_families = {}, {}
    for domain, content in parts.items():
        train, validation, benchmark = (content[k] for k in ("train", "validation", "benchmark"))
        train_by_prompt = {normal(user_text(r)): r for r in train}
        if len(train_by_prompt) != len(train):
            raise ValueError(f"Duplicate training question in {domain}")
        val_questions = {normal(user_text(r)) for r in validation}
        if len(val_questions) != len(validation) or val_questions & train_by_prompt.keys():
            raise ValueError(f"Training/validation prompt overlap in {domain}")
        bench_questions = [normal(user_text(r)) for r in benchmark]
        if len(bench_questions) != len(set(bench_questions)) or set(bench_questions) & val_questions:
            raise ValueError(f"Duplicate or validation-leaking benchmark in {domain}")
        if domain == "style":
            families = [{r["family_id"] for r in rows} for rows in (train, validation, benchmark)]
            if families[0] & families[1] or families[0] & families[2] or families[1] & families[2]:
                raise ValueError("Style scenario family overlap")
        stats[domain] = {}
        for split, rows in (("train", train), ("validation", validation), ("benchmark", benchmark)):
            if len({r["id"] for r in rows}) != len(rows):
                raise ValueError("Duplicate row IDs")
            lengths, supervised = [], []
            for row in rows:
                if row["messages"][0] != {"role": "system", "content": SYSTEM_PROMPT} or row.get("tools"):
                    raise ValueError("System/tools differ across diagnostics")
                if any(m.get("tool_calls") or m["role"] not in ("system", "user", "assistant") for m in row["messages"]):
                    raise ValueError("Unexpected tools/roles")
                if split != "benchmark":
                    if row["messages"][-1]["role"] != "assistant" or not row["messages"][-1]["content"]:
                        raise ValueError("Missing training response")
                    enc = encode_assistant_only(tokenizer, row, max_length)
                    lengths.append(len(enc["input_ids"]))
                    supervised.append(sum(i != -100 for i in enc["labels"]))
                else:
                    if [m["role"] for m in row["messages"]] != ["system", "user"] or "reference" in row:
                        raise ValueError("Evaluation answer must exist only in metadata")
                    meta = row["metadata"]
                    if not meta["rubric"]["must_include"] or not meta["rubric"]["must_not"]:
                        raise ValueError("Incomplete rubric")
                    prompt = normal(user_text(row))
                    overlap = prompt in train_by_prompt
                    if overlap != (meta["task_type"] == "exact") or overlap != meta["seen_question"]:
                        raise ValueError("Only labelled exact probes may repeat training questions")
                    expected = meta.get("expected_answer")
                    rendered = render_chat(tokenizer, row, generation=True)
                    # Numeric facts use token boundaries to avoid incidental digits in names.
                    if expected and re.search(r"(?<!\w)" + re.escape(expected) + r"(?!\w)", rendered, re.I):
                        raise ValueError(f"Answer leaked into inference prompt: {row['id']}")
                    lengths.append(len(tokenizer(rendered, add_special_tokens=False)["input_ids"]))
            stats[domain][split] = {"rows": len(rows), "max_tokens": max(lengths), "total_tokens": sum(lengths),
                                    "supervised_tokens": sum(supervised) if supervised else 0,
                                    "overlength_rows": sum(n > max_length for n in lengths)}
            if stats[domain][split]["overlength_rows"]:
                raise ValueError("Overlength data")
        stats[domain]["benchmark_task_counts"] = dict(Counter(r["metadata"]["task_type"] for r in benchmark))
        stats[domain]["exact_overlap_intentional"] = sum(r["metadata"]["seen_question"] for r in benchmark)
        stats[domain]["novel_question_train_overlap"] = 0
        all_prompt_families[domain] = {normal(user_text(r)) for r in train + validation + benchmark}
    for a, b in (("one_fact", "mini"), ("one_fact", "style"), ("mini", "style")):
        if all_prompt_families[a] & all_prompt_families[b]:
            raise ValueError(f"Cross-curriculum question overlap: {a}/{b}")
    return stats


def audit_old_abc(tokenizer, max_length):
    directory = LAB / "datasets" / "abc_v1"
    if not directory.is_dir():
        return {"status": "not_available"}
    result = {"status": "read_only_audit", "source_files": {}, "variants": {}, "manual_findings": [
        "No raw book excerpts copied to tracked diagnostics source; the prior source review and frozen report are referenced.",
        "The same old system mentions book answering in all domains; this is not a label conflict but an avoidable domain cue for a style experiment.",
        "B targets repeatedly teach source-attribution phrasing. Prior measured book-prefix spillover is consistent with form learning, not proof of a causal mechanism.",
        "A target styles vary deliberately: humour for trivial situations, seriousness for distress, directness for facts. These contextual differences are not contradictions.",
        "Known-fact paraphrase targets differ in wording by design; an automatic exact-string audit cannot establish all semantic consistency.",
        "Previous source claims are not reused as new diagnostics facts. Synthetic learning success cannot establish real-book acquisition.",
        "This data audit checks preprocessing and masking, not that optimizer updates or adapter loading were implemented correctly.",
    ]}
    all_train = {}
    for variant in ("A", "B", "C"):
        data_path = directory / f"{variant}_train.jsonl"
        rows = read_jsonl(data_path)
        result["source_files"][str(data_path.relative_to(LAB))] = file_hash(data_path)
        targets = defaultdict(set)
        incorrect_masks = []
        supervised_tokens = 0
        prefix_count = 0
        for row in rows:
            if row["messages"][-1]["role"] != "assistant":
                raise ValueError("Old train row has no assistant target")
            target = row["messages"][-1]["content"]
            targets[normal(user_text(row))].add(normal(target))
            all_train[row["id"]] = row
            encoded = encode_assistant_only(tokenizer, row, max_length)
            supervised_tokens += sum(t != -100 for t in encoded["labels"])
            rendered = render_chat(tokenizer, row)
            native = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
            spans = [(m.start(1), m.end()) for m in re.finditer(r"<\|im_start\|>assistant\n(.*?)<\|im_end\|>", rendered, re.S)]
            for index, ((start, end), label) in enumerate(zip(native["offset_mapping"], encoded["labels"])):
                expected = end > start and any(start >= lo and end <= hi for lo, hi in spans)
                if (label != -100) != expected or (label != -100 and label != native["input_ids"][index]):
                    incorrect_masks.append({"id": row["id"], "token_index": index})
            prefix_count += bool(re.search(r"поливалин|энциклопеди|в книге", target, re.I))
        result["variants"][variant] = {"rows": len(rows), "distinct_prompts": len(targets),
                                      "same_prompt_conflicting_targets": sum(len(v) > 1 for v in targets.values()),
                                      "assistant_mask_violations": incorrect_masks,
                                      "supervised_tokens_recomputed": supervised_tokens,
                                      "target_book_attribution_phrase_count": prefix_count,
                                      "system_mentions_book": sum("книг" in row["messages"][0]["content"].casefold() for row in rows)}
    a, b, c = (read_jsonl(directory / f"{v}_train.jsonl") for v in ("A", "B", "C"))
    result["C_exactly_equals_A_plus_B"] = c == a + b
    if not result["C_exactly_equals_A_plus_B"] or any(v["same_prompt_conflicting_targets"] or v["assistant_mask_violations"] for v in result["variants"].values()):
        raise ValueError("Old ABC data audit found contradictory labels or corrupt union/masks; inspect before training")
    for name in ("REPORT.md", "comparison_examples.md", "posthoc_diagnostics.json"):
        path = LAB / "evaluation" / "reports" / "abc_v1" / name
        if path.exists():
            result["source_files"][str(path.relative_to(LAB))] = file_hash(path)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="datasets/diagnostics_v1")
    args = parser.parse_args()
    output = private_destination(lab_path(args.output))
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Existing frozen diagnostics cannot be overwritten")
    identifier = "СЕВРА-" + secrets.token_hex(4).upper()
    ot, ov, ob, exact, one_registry = build_one_fact(identifier)
    mt, mv, mb, mini_registry = build_mini()
    st, sv, sb = build_style()
    parts = {domain: {"train": t, "validation": v, "benchmark": b} for domain, t, v, b in
             (("one_fact", ot, ov, ob), ("mini", mt, mv, mb), ("style", st, sv, sb))}
    config = read_config()
    runtime_environment()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"], revision=config["model"]["revision"],
                                              local_files_only=True, trust_remote_code=False)
    stats = audit_rows(parts, tokenizer, config["model"]["max_seq_length"])
    old_audit = audit_old_abc(tokenizer, config["model"]["max_seq_length"])
    previous_prompts = set()
    for old_path in (LAB / "datasets" / "abc_v1").glob("*.jsonl"):
        previous_prompts.update(normal(user_text(r)) for r in read_jsonl(old_path))
    new_prompts = {normal(user_text(r)) for domain in parts.values() for rows in domain.values() for r in rows}
    if new_prompts & previous_prompts:
        raise ValueError("Diagnostics reused an ABC question")
    new_audit = {"status": "passed_before_generation", "new_data": stats, "old_abc": old_audit,
                 "review": {"source": "prompts.py read as text; never imported", "new_authored_questions": True,
                            "old_abc_exact_prompt_overlap": 0, "inference_reference_leaks": 0,
                            "shared_system_contains_answers": False, "style_has_book_instruction": False,
                            "unknown_fact_training": "No unknown answers added to factual training; unknown probes test honest boundaries.",
                            "mini_validation": "One new formulation per known fact, loss-only; not an independent fact holdout.",
                            "one_fact_question_diversity": "Lookup, radio-channel labelling and a registry-field task. They intentionally target one repeated relation for an overfit diagnostic, not three independent facts.",
                            "mini_question_diversity": "Two distinct requests per atomic relation, such as lookup versus task-oriented label/drawing/calendar completion.",
                            "style_quality": "18 new scenarios; 6 gentle irony, 4 serious support, 2 direct content, 2 honesty, 2 correction/non-sycophancy, 2 format/tone instructions. Clear exceptions prevent personality overriding facts or distress.",
                            "curator": "AI-assisted author/reviewer; no independent human audit"}}
    for domain, content in parts.items():
        for split, rows in content.items():
            write_jsonl(output / f"{domain}_{split}.jsonl", rows)
    write_jsonl(output / "one_fact_training_probes.jsonl", exact)
    write_json(output / "facts.json", {"one_fact": one_registry, "mini": mini_registry})
    write_json(output / "data_audit.json", new_audit)
    base_hyperparameters = {"seed": 3407, "model": config["model"], "lora": config["lora"],
                            "batch_size": 1, "gradient_accumulation_steps": 4, "optimizer": "adamw_8bit",
                            "scheduler": "constant_with_warmup", "warmup_steps": 2,
                            "generation": {"do_sample": False, "enable_thinking": False, "max_new_tokens": 128}}
    protocol = {"protocol_id": "diagnostics_v1", "created_utc": datetime.now(timezone.utc).isoformat(),
                "status": "frozen_before_generation", "common_system": SYSTEM_PROMPT,
                "system_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
                "base_hyperparameters": base_hyperparameters,
                "phase_budgets": {"one_fact": {"max_steps": 100, "learning_rate": 0.0002},
                                  "mini": {"max_steps": 100, "learning_rate": 0.0002},
                                  "style": {"max_steps": 50, "learning_rate": 0.0001}},
                "one_fact_gate": {"primary_exact_id": "diag_one_exact_01", "expected_answer": identifier,
                                  "scoring": "exact", "case_sensitive": False,
                                  "normalization": "Unicode NFKC, casefold, collapse whitespace, strip terminal .!?; preserve internal characters and hyphens",
                                  "criterion": "Generated response to designated exact training question equals the full new identifier. Teacher-forced loss or substring occurrence alone cannot open the mini phase.",
                                  "report_required": "All three exact and all three new-paraphrase responses; both unknowns and format probe too."},
                "data_stats": stats,
                "reference_boundary": "Training targets are assistant messages; evaluation targets/rubrics are metadata only. Model inference receives system and user messages only.",
                "factual_metrics": {"exact": "Exact answer normalization; no extra words.",
                                    "paraphrase": "Full expected answer presence is a diagnostic indicator, followed by semantic/rubric review for contradictions and unsupported extras.",
                                    "unknown": "Manual binary rubric; do not infer success from a single keyword.",
                                    "format": "Exact required short output, separate from semantic fact availability."},
                "style_metrics": "Report style, general skills, honesty separately using fixed binary rubrics. Correct factual content/explicit format take priority over humour.",
                "limitations": ["Tiny synthetic overfit and retention diagnostics, not a measure of real knowledge acquisition.",
                                "The single fact is cryptographically sampled but its 32-bit code is an experimental identifier, not a security credential.",
                                "Repeated questions about one fact are intentional here and must not be portrayed as dataset growth.",
                                "Exact probes repeat training wording intentionally; validation and paraphrases share learned facts.",
                                "Style training and evaluation are small authored scenario sets with AI review and a single seed."],
                "source_files": {p.relative_to(LAB.parent).as_posix(): file_hash(p) for p in
                                 [Path(__file__), LAB / "diagnostics_seed.py", LAB / "common.py", LAB.parent / "prompts.py", LAB / "config.yaml"]},
                "library_versions": {"python": sys.version, **{name: packages.version(name) for name in
                                     ("torch", "transformers", "trl", "peft", "datasets", "bitsandbytes", "unsloth", "unsloth_zoo")}},
                "artifact_sha256": {p.relative_to(output).as_posix(): file_hash(p) for p in sorted(output.rglob("*")) if p.is_file()}}
    write_json(output / "protocol.json", protocol)
    write_json(output / "protocol.sha256.json", {"protocol.json": file_hash(output / "protocol.json")})
    print(json.dumps({"output": str(output), "identifier": identifier, "stats": stats,
                      "old_abc_audit": old_audit["variants"], "protocol_sha256": file_hash(output / "protocol.json")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
