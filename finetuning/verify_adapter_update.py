"""Independently verify changes between saved LoRA checkpoints using CPU arrays."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from common import lab_path, write_json


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def compare_checkpoints(before_path, after_path):
    import numpy as np
    from safetensors import safe_open
    before_path, after_path = Path(before_path), Path(after_path)
    before_file = before_path / "adapter_model.safetensors"
    after_file = after_path / "adapter_model.safetensors"
    before_state = json.loads((before_path / "trainer_state.json").read_text(encoding="utf-8"))
    after_state = json.loads((after_path / "trainer_state.json").read_text(encoding="utf-8"))
    if after_state["global_step"] <= before_state["global_step"]:
        raise ValueError("Checkpoints must be in increasing optimizer-step order")
    changes = []
    with safe_open(str(before_file), framework="numpy") as old, safe_open(str(after_file), framework="numpy") as new:
        if set(old.keys()) != set(new.keys()):
            raise ValueError("Checkpoint tensor names differ")
        for key in old.keys():
            if "lora_" not in key:
                raise ValueError("Expected adapter-only checkpoint")
            a, b = old.get_tensor(key), new.get_tensor(key)
            if a.shape != b.shape:
                raise ValueError(f"Checkpoint shape mismatch: {key}")
            if not np.isfinite(a).all() or not np.isfinite(b).all():
                raise ValueError(f"Non-finite parameter: {key}")
            delta = b.astype(np.float64) - a.astype(np.float64)
            changes.append({"name": key, "changed_elements": int(np.count_nonzero(delta)),
                            "maximum_absolute_delta": float(np.abs(delta).max()),
                            "rms_delta": float(np.sqrt(np.square(delta).mean()))})
    changed = [row for row in changes if row["changed_elements"]]
    if not changed:
        raise ValueError("No adapter weights changed between checkpoints")
    return {"status": "passed", "method": "Independent CPU-only comparison of serialized LoRA tensors",
            "before_step": before_state["global_step"], "after_step": after_state["global_step"],
            "before_sha256": digest(before_file), "after_sha256": digest(after_file),
            "changed_tensors": len(changed), "total_tensors": len(changes),
            "changed_elements": sum(row["changed_elements"] for row in changed),
            "changed_by_lora_matrix": dict(Counter("A" if ".lora_A." in row["name"] else "B" for row in changed)),
            "maximum_absolute_delta": max(row["maximum_absolute_delta"] for row in changes),
            "tensors": changes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before")
    parser.add_argument("after")
    parser.add_argument("--output", default="evaluation/reports/checkpoint-update-proof.json")
    args = parser.parse_args()
    report = compare_checkpoints(lab_path(args.before), lab_path(args.after))
    write_json(lab_path(args.output), report)
    print(json.dumps({key: value for key, value in report.items() if key != "tensors"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
