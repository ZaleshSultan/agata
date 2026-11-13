"""Compare two completed reports without loading models or exposing private data."""
import argparse
import json
from common import lab_path, write_json


def compare(before, after):
    for field in ("benchmark_sha256", "generation", "limit", "base_model"):
        if before.get(field) != after.get(field):
            raise ValueError(f"Reports have different {field}; a controlled comparison is not possible")
    if before.get("status") != "complete" or after.get("status") != "complete":
        raise ValueError("Both reports must be complete")
    if [r["id"] for r in before["results"]] != [r["id"] for r in after["results"]]:
        raise ValueError("Reports have different questions or order")
    rows = []
    for old, new in zip(before["results"], after["results"]):
        rows.append({"id": old["id"], "category": old["category"], "prompt": old["prompt"],
                     "source_kind": new.get("source_kind"), "actual_training_exposure": new.get("actual_training_exposure"),
                     "before": old["response"], "after": new["response"],
                     "before_assessment": old["assessment"], "after_assessment": new["assessment"]})
    return {"before_adapter": before.get("adapter"), "after_adapter": after.get("adapter"),
            "benchmark_sha256": before["benchmark_sha256"], "before_summary": before["summary"],
            "after_summary": after["summary"], "results": rows,
            "before_summary_by_source": before.get("summary_by_source"),
            "after_summary_by_source": after.get("summary_by_source"),
            "after_summary_by_training_exposure": after.get("summary_by_training_exposure"),
            "conclusion": "Diagnostic comparison only. Manual scoring and a larger held-out evaluation are required; "
                          "one reproduced answer does not establish reliable knowledge acquisition."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before")
    parser.add_argument("after")
    parser.add_argument("--output", default="evaluation/reports/comparison")
    args = parser.parse_args()
    before, after = [json.loads(lab_path(path).read_text(encoding="utf-8")) for path in (args.before, args.after)]
    result = compare(before, after)
    output = lab_path(args.output)
    if output.with_suffix(".json").exists() or output.with_suffix(".md").exists():
        raise SystemExit(f"Report exists: {output}. Choose a new --output stem.")
    write_json(output.with_suffix(".json"), result)
    lines = ["# Сравнение базовой модели и адаптера", "", result["conclusion"], ""]
    for row in result["results"]:
        lines += [f"## {row['id']} · {row['category']}", "", row["prompt"], "",
                  "До: " + row["before"], "", "После: " + row["after"], "",
                  "Источник: " + str(row.get("source_kind")) + "; воздействие обучения: " + str(row.get("actual_training_exposure")), ""]
    output.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")
    print(output.with_suffix(".md"))


if __name__ == "__main__":
    main()
