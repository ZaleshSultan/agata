"""Plot real training traces, prepare blinded review, and aggregate semantic A/B/C ratings.

No lexical truth scoring is used. Human/AI semantic ratings must cite the frozen
item rubric and are kept separate from measured loss, memory, and generation.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import random
import statistics
import time

from common import file_hash, lab_path, read_jsonl, write_json

MODELS = ("baseline", "A", "B", "C")
PRIMARY_GROUPS = ("persona", "known_paraphrase", "uncertainty", "general_skills")


def ensure_new(*paths):
    for path in paths:
        if path.exists():
            raise FileExistsError(f"Preserving existing artifact: {path}; choose a new output")


def load_reports(directory, suffix=""):
    reports = {name: json.loads((directory / f"{name}{suffix}.json").read_text(encoding="utf-8")) for name in MODELS}
    reference = reports["baseline"]
    ids = [row["id"] for row in reference["results"]]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate benchmark item IDs")
    for name, report in reports.items():
        if report.get("status") != "complete" or report.get("limit") is not None:
            raise ValueError(f"Incomplete or subset report: {name}{suffix}")
        for key in ("benchmark_sha256", "base_model", "generation", "limit"):
            if report.get(key) != reference.get(key):
                raise ValueError(f"Incomparable report {name}{suffix}: {key}")
        if [row["id"] for row in report["results"]] != ids:
            raise ValueError("Report item IDs/order differ")
        for base_row, row in zip(reference["results"], report["results"]):
            if any(row.get(key) != base_row.get(key) for key in ("prompt", "reference", "metadata")):
                raise ValueError("Report prompts, references, or rubrics differ")
    if reference.get("adapter"):
        raise ValueError("The baseline must have no adapter")
    adapters = [reports[name].get("adapter") for name in MODELS[1:]]
    if not all(adapters) or len(set(adapters)) != 3:
        raise ValueError("A/B/C must be three distinct saved adapters")
    return reports


def plots(runs_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for name in MODELS[1:]:
        run = runs_dir / name
        report = json.loads((run / "training_report.json").read_text(encoding="utf-8"))
        if report.get("status") != "passed":
            raise ValueError(f"Cannot plot incomplete run {name}")
        history = json.loads((run / "trainer_state.json").read_text(encoding="utf-8"))["log_history"]
        figure_path = run / "loss.png"
        if not figure_path.exists():
            fig, axis = plt.subplots(figsize=(8, 4.4), constrained_layout=True)
            for field, label, marker in (("loss", "Training loss", "."), ("eval_loss", "Validation loss", "o")):
                points = [(row["step"], row[field]) for row in history if field in row]
                if points:
                    axis.plot(*zip(*points), marker=marker, linewidth=1.4, label=label)
            axis.set(xlabel="Optimizer step", ylabel="Assistant-token cross-entropy",
                     title=f"AGATA experiment {name}: observed training trace")
            axis.grid(alpha=.2)
            axis.legend()
            fig.text(.5, -.025, "Different A/B/C datasets: loss magnitudes are not equivalent quality scores.",
                     ha="center", fontsize=8)
            fig.savefig(figure_path, dpi=170, bbox_inches="tight")
            plt.close(fig)
        vram_path = run / "vram.png"
        if not vram_path.exists():
            with (run / "nvidia_vram.csv").open(encoding="utf-8") as stream:
                records = list(csv.DictReader(stream))
            memory = [float(row["memory_used_mib"]) for row in records if row.get("memory_used_mib")]
            fig, axis = plt.subplots(figsize=(8, 3.7), constrained_layout=True)
            axis.plot(range(len(memory)), memory, color="#3859a6")
            axis.set(xlabel="Driver sample (approximately one per second)", ylabel="GPU memory used (MiB)",
                     title=f"AGATA experiment {name}: full-process NVIDIA memory")
            axis.grid(alpha=.2)
            fig.savefig(vram_path, dpi=170)
            plt.close(fig)
        print(f"Saved plots: {figure_path}, {vram_path}")


def packet(reports_dir, output, seed=3410):
    key_path = output.with_name(output.stem + "_key.json")
    template_path = output.with_name(output.stem + "_ratings_template.json")
    ensure_new(output, key_path, template_path)
    rng = random.Random(seed)
    items, mapping, sources, ratings = [], {}, {}, []
    counter = 0
    for suite, suffix in (("benchmark", ""), ("memorization", "_memorization")):
        reports = load_reports(reports_dir, suffix)
        for name in MODELS:
            path = reports_dir / f"{name}{suffix}.json"
            sources[str(path.resolve())] = file_hash(path)
        for index, row in enumerate(reports["baseline"]["results"]):
            metadata = row.get("metadata") or {}
            if not metadata.get("rubric"):
                raise ValueError(f"Missing preregistered semantic rubric: {row['id']}")
            item = {"suite": suite, "id": row["id"], "prompt": row["prompt"],
                    "reference": row.get("reference"), "metadata": metadata, "responses": []}
            order = list(MODELS)
            rng.shuffle(order)
            for name in order:
                answer = reports[name]["results"][index]
                counter += 1
                response_id = f"R{counter:04d}"
                item["responses"].append({"response_id": response_id, "response": answer["response"],
                                          "finish_reason": answer.get("finish_reason", "unrecorded"),
                                          "generated_tokens": answer.get("generated_tokens")})
                mapping[response_id] = {"model": name, "suite": suite, "item_id": row["id"]}
                ratings.append({"response_id": response_id, "score": None, "reason": "",
                                "unsupported_claim": None, "rubric_evidence": ""})
            items.append(item)
    document = {"schema_version": 1, "review_instructions": [
        "Rate each response independently against the item reference, evidence, and preregistered rubric.",
        "score must be 1 for fully satisfying the rubric or 0 otherwise; partial answers receive 0.",
        "Do not infer truth from keyword overlap. Negation, invented attribution, unsupported details, and failed constraints matter.",
        "Known-book paraphrases test training-exposed ideas under unseen questions, not unseen factual knowledge.",
        "Memorization and secondary diagnostics are excluded from the primary metric.",
        "Identify unsupported source/action claims separately, and give a concrete reason and rubric evidence.",
        "Model labels are hidden and order varies by item; this is procedural blinding, not an independent human study."],
        "items": items}
    write_json(output, document)
    write_json(key_path, {"schema_version": 1, "packet_sha256": file_hash(output), "shuffle_seed": seed,
                          "source_reports_sha256": sources, "response_mapping": mapping})
    write_json(template_path, {"reviewer": "", "review_method": "", "reviewed_unix": None,
                              "packet_sha256": file_hash(output), "ratings": ratings})
    print(f"Blinded packet: {output}; keep mapping away from reviewers: {key_path}")


def paired_interval(pairs, draws=10000, seed=3411):
    """Exploratory paired item bootstrap, stratified by the four fixed domains."""
    if set(pairs) != set(PRIMARY_GROUPS) or any(not group for group in pairs.values()):
        raise ValueError("All four primary domains must have at least one paired rating")
    rng = random.Random(seed)
    values = []
    for _ in range(draws):
        values.append(statistics.mean(
            statistics.mean(rng.choice(pairs[group]) for _ in pairs[group]) for group in PRIMARY_GROUPS))
    values.sort()
    return {"low": values[int(.025 * (draws - 1))], "high": values[int(.975 * (draws - 1))],
            "draws": draws, "seed": seed, "method": "paired item bootstrap stratified by primary domain",
            "limitation": "Conditional on these benchmark items and this training seed; not a training-seed or population uncertainty estimate."}


def validate_packet_mapping(document, key):
    """One immutable answer identity for every model/item, without trusting order."""
    mapping = key["response_mapping"]
    item_ids, response_ids = set(), set()
    for item in document["items"]:
        identity = (item["suite"], item["id"])
        if identity in item_ids:
            raise ValueError("Duplicate suite/item identity in review packet")
        item_ids.add(identity)
        models = []
        for response in item["responses"]:
            response_id = response["response_id"]
            if not isinstance(response_id, str) or not response_id.strip() or response_id in response_ids:
                raise ValueError("Blinded response IDs must be unique nonempty strings")
            response_ids.add(response_id)
            if response_id not in mapping:
                raise ValueError("Blinded response is missing from the model key")
            linked = mapping[response_id]
            if (linked.get("suite"), linked.get("item_id")) != identity:
                raise ValueError("Response key points to another suite/item")
            models.append(linked.get("model"))
        if len(models) != len(MODELS) or set(models) != set(MODELS):
            raise ValueError("Require exactly one response from each of baseline/A/B/C per item")
    if set(mapping) != response_ids:
        raise ValueError("Packet and model key have different response IDs")


def aggregate(document, key, ratings):
    validate_packet_mapping(document, key)
    mapping = key["response_mapping"]
    if (any(not isinstance(ratings.get(field), str) or not ratings[field].strip() for field in ("reviewer", "review_method")) or
            type(ratings.get("reviewed_unix")) not in (int, float) or ratings["reviewed_unix"] <= 0):
        raise ValueError("Semantic ratings must identify the reviewer, method, and review time")
    rows = ratings["ratings"]
    by_id = {row["response_id"]: row for row in rows}
    if len(by_id) != len(rows) or set(by_id) != set(mapping):
        raise ValueError("Require exactly one semantic rating for every blinded response")
    for row in rows:
        if type(row.get("score")) is not int or row["score"] not in (0, 1):
            raise ValueError("Semantic scores must be integer 0 or 1")
        if any(not isinstance(row.get(field), str) or not row[field].strip() for field in ("reason", "rubric_evidence")):
            raise ValueError("Every semantic score must have a reason and rubric evidence")
        if not isinstance(row.get("unsupported_claim"), bool):
            raise ValueError("Every response needs an explicit unsupported-claim flag")
        if row["score"] == 1 and row["unsupported_claim"]:
            raise ValueError("A substantive unsupported claim cannot receive full credit under the frozen rubric")
    items = {(row["suite"], row["id"]): row for row in document["items"]}
    grouped = {name: defaultdict(list) for name in MODELS}
    per_item = defaultdict(dict)
    for response_id, identity in mapping.items():
        suite, item_id, model = identity["suite"], identity["item_id"], identity["model"]
        item = items[(suite, item_id)]
        metadata = item["metadata"]
        primary = metadata.get("primary_group") if suite == "benchmark" else None
        if primary and primary not in PRIMARY_GROUPS:
            raise ValueError(f"Unknown primary metric domain: {primary}")
        group = primary or ("memorization" if suite == "memorization" else metadata.get("secondary_group") or "secondary")
        rating = dict(by_id[response_id], primary_group=primary, group=group, model=model, suite=suite, item_id=item_id)
        grouped[model][group].append(rating)
        per_item[(suite, item_id)][model] = rating
    summary = {}
    for name in MODELS:
        if any(not grouped[name][group] for group in PRIMARY_GROUPS):
            raise ValueError("Missing one of the four primary domains")
        groups = {group: {"n": len(values), "correct": sum(row["score"] for row in values),
                          "accuracy": statistics.mean(row["score"] for row in values),
                          "unsupported_claims": sum(row["unsupported_claim"] for row in values)}
                  for group, values in grouped[name].items()}
        summary[name] = {"groups": groups,
                         "primary_macro_accuracy": statistics.mean(groups[group]["accuracy"] for group in PRIMARY_GROUPS)}
    for name in MODELS[1:]:
        pairs = {group: [] for group in PRIMARY_GROUPS}
        improved = regressed = unchanged = 0
        for item in per_item.values():
            row = item[name]
            if row["primary_group"]:
                delta = row["score"] - item["baseline"]["score"]
                pairs[row["primary_group"]].append(delta)
                improved += delta > 0
                regressed += delta < 0
                unchanged += delta == 0
        summary[name].update(primary_delta_vs_base=summary[name]["primary_macro_accuracy"] - summary["baseline"]["primary_macro_accuracy"],
                             paired_items={"improved": improved, "regressed": regressed, "unchanged": unchanged},
                             exploratory_delta_95pct_interval=paired_interval(pairs))
    maximum = max(summary[name]["primary_macro_accuracy"] for name in MODELS[1:])
    leaders = [name for name in MODELS[1:] if summary[name]["primary_macro_accuracy"] == maximum]
    return {"metric": "Unweighted mean of strict semantic accuracy in persona, known_paraphrase, uncertainty, general_skills",
            "summary": summary, "highest_observed_adapter_score": leaders,
            "greatest_observed_delta": maximum - summary["baseline"]["primary_macro_accuracy"],
            "rank_interpretation": "Observed ranking for this small frozen benchmark and one seed; not proof of general superiority.",
            "ratings_by_item": [dict(suite=suite, item_id=item_id, ratings=values) for (suite, item_id), values in per_item.items()]}


def compare(reports_dir, runs_dir, packet_path, key_path, ratings_path, output):
    json_path, md_path = output.with_suffix(".json"), output.with_suffix(".md")
    examples_path = output.with_name(output.stem + "_examples.md")
    ensure_new(json_path, md_path, examples_path)
    document, key, ratings = (json.loads(path.read_text(encoding="utf-8")) for path in (packet_path, key_path, ratings_path))
    packet_hash = file_hash(packet_path)
    if key["packet_sha256"] != packet_hash or ratings["packet_sha256"] != packet_hash:
        raise ValueError("Review packet or rating provenance hash differs")
    for path, expected_hash in key["source_reports_sha256"].items():
        if file_hash(path) != expected_hash:
            raise ValueError("Source model report changed after blinding")
    reports = load_reports(reports_dir)
    memorization = load_reports(reports_dir, "_memorization")
    result = aggregate(document, key, ratings)
    result["provenance"] = {"packet_sha256": packet_hash, "key_sha256": file_hash(key_path),
                            "ratings_sha256": file_hash(ratings_path), "reviewer": ratings["reviewer"],
                            "review_method": ratings["review_method"], "reviewed_unix": ratings["reviewed_unix"],
                            "source_reports_sha256": key["source_reports_sha256"]}
    result["training"] = {}
    for name in MODELS[1:]:
        report = json.loads((runs_dir / name / "training_report.json").read_text(encoding="utf-8"))
        result["training"][name] = {field: report.get(field) for field in (
            "global_step", "starting_global_step", "ending_epoch", "elapsed_seconds", "metrics",
            "cuda_peak_allocated_mib", "cuda_peak_reserved_mib", "nvidia", "adapter_reload_verified",
            "initial_lora_sha256", "final_lora_sha256", "trainable_parameters", "dataset")}
    result["token_limit_counts"] = {name: sum(row.get("hit_token_limit", False) for row in reports[name]["results"]) for name in MODELS}
    result["limitations"] = [
        "AI/manual semantic review is judgment, not an automated factual oracle; reviewer identity and rationales are preserved.",
        "Small benchmark, one seed, fixed 32 optimizer steps; no hyperparameter search or significance claim.",
        "A/B/C use different training distributions and assistant-token totals; the mixed run receives fewer updates per domain.",
        "Exact training-question reproduction is reported only as memorization, outside the primary score.",
        "Paraphrase success supports access to trained ideas under new wording, not acquisition of previously unseen facts.",
        "Book ideas may be common knowledge; correct source attribution alone does not demonstrate novel knowledge acquisition.",
        "Held-out chapter and composition tests are separate diagnostics and do not enter the four-domain primary score.",
        "Token-limit outputs were scored as generated without adaptive reruns or imagined continuations."]
    write_json(json_path, result)
    lines = ["# AGATA A/B/C: измеренное сравнение", "", result["metric"], "",
             "| Модель | Характер | Перефразирование книги | Неизвестность | Общие навыки | Macro | Δ к base |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name in MODELS:
        row = result["summary"][name]
        cells = [f"{row['groups'][group]['correct']}/{row['groups'][group]['n']}" for group in PRIMARY_GROUPS]
        lines.append(f"| {name} | " + " | ".join(cells) + f" | {row['primary_macro_accuracy']:.3f} | {row.get('primary_delta_vs_base', 0):+.3f} |")
    lines += ["", "Наивысший наблюдаемый результат среди адаптеров: " + ", ".join(result["highest_observed_adapter_score"]) + ".",
              "Это результат одного короткого запуска на малом фиксированном наборе, а не доказательство превосходства метода.", "",
              "| Адаптер | Исправлено / ухудшено | 95% интервал Δ (bootstrap вопросов) | Шаги | VRAM NVIDIA, MiB |",
              "|---|---|---|---:|---:|"]
    for name in MODELS[1:]:
        row, training = result["summary"][name], result["training"][name]
        interval, paired = row["exploratory_delta_95pct_interval"], row["paired_items"]
        lines.append(f"| {name} | {paired['improved']} / {paired['regressed']} | [{interval['low']:+.3f}, {interval['high']:+.3f}] | "
                     f"{training['global_step']} | {training['nvidia'].get('peak_used_mib')} |")
    lines += ["", "Bootstrap отражает вариативность вопросов внутри четырёх категорий при данном seed. Он не измеряет вариативность обучения.",
              "", "## Дополнительные группы", "", "```json",
              json.dumps({name: {group: values for group, values in result["summary"][name]["groups"].items() if group not in PRIMARY_GROUPS} for name in MODELS}, ensure_ascii=False, indent=2),
              "```", "", "## Ограничения", ""]
    lines.extend("- " + item for item in result["limitations"])
    lines += ["", f"Оценщик: {ratings['reviewer']}. Метод: {ratings['review_method']}.",
              f"SHA256 рейтингов: `{file_hash(ratings_path)}`.", ""]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    examples = ["# AGATA: все ответы до и после обучения", "", "Ответы приведены без выборочного отбора; оценки и источник — в JSON сравнения.", ""]
    for suite, source in (("benchmark", reports), ("memorization", memorization)):
        for index, item in enumerate(source["baseline"]["results"]):
            examples.extend([f"## {suite}: {item['id']}", "", item["prompt"], "", "Эталон: " + str(item["reference"]), ""])
            for name in MODELS:
                response = source[name]["results"][index]
                examples.extend([f"### {name}", "", response["response"], "",
                                 f"Завершение: {response.get('finish_reason', 'unrecorded')}; токенов: {response.get('generated_tokens')}.", ""])
    examples_path.write_text("\n".join(examples), encoding="utf-8")
    print(f"Comparison: {json_path}; all before/after answers: {examples_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("plots", "packet", "compare"))
    parser.add_argument("--reports-dir", default="evaluation/reports/abc_v1")
    parser.add_argument("--runs-dir", default="outputs/abc_v1")
    parser.add_argument("--packet", default="evaluation/reports/abc_v1/review_packet.json")
    parser.add_argument("--key", default="evaluation/reports/abc_v1/review_packet_key.json")
    parser.add_argument("--ratings", default="evaluation/reports/abc_v1/semantic_ratings.json")
    parser.add_argument("--output", default="evaluation/reports/abc_v1/comparison")
    args = parser.parse_args()
    reports_dir, runs_dir = lab_path(args.reports_dir), lab_path(args.runs_dir)
    if args.stage == "plots":
        plots(runs_dir)
    elif args.stage == "packet":
        packet(reports_dir, lab_path(args.packet))
    else:
        compare(reports_dir, runs_dir, lab_path(args.packet), lab_path(args.key), lab_path(args.ratings), lab_path(args.output))


if __name__ == "__main__":
    main()
