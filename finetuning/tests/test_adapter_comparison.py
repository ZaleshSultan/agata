"""CPU-only integrity tests; no model, CUDA allocation, or original artifacts are used."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import evaluate_adapter_comparison as comparison
from common import file_hash


class ProtocolIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.lab = Path(self.temporary.name)
        self.protocol_path = self.lab / "protocol.json"
        self.document = {
            "status": "frozen_before_comparison",
            "generation": {"do_sample": False, "max_new_tokens": 128, "enable_thinking": False},
            "code_sha256": {}, "benchmarks": {},
            "variants": {"baseline": {"adapter": None}},
        }
        for name in comparison.CODE:
            path = self.lab / name
            path.write_bytes(("frozen code: " + name).encode("utf-8"))
            self.document["code_sha256"][name] = file_hash(path)
        for split in ("benchmark", "recall"):
            path = self.lab / "datasets" / (split + ".jsonl")
            path.parent.mkdir(exist_ok=True)
            path.write_bytes((json.dumps({"id": split, "prompt": "fixture"}) + "\n").encode("utf-8"))
            self.document["benchmarks"][split] = {
                "path": path.relative_to(self.lab).as_posix(), "sha256": file_hash(path),
            }
        for variant in ("original", "half"):
            folder = self.lab / "adapters" / variant
            folder.mkdir(parents=True)
            files = {"adapter_config.json": b'{"r":16}',
                     "adapter_model.safetensors": ("frozen weights: " + variant).encode("ascii")}
            for name, content in files.items():
                (folder / name).write_bytes(content)
            self.document["variants"][variant] = {
                "adapter": folder.relative_to(self.lab).as_posix(),
                "adapter_sha256": {name: file_hash(folder / name) for name in files},
            }
        self.lab_patch = patch.object(comparison, "LAB", self.lab)
        self.lab_patch.start()
        self.addCleanup(self.lab_patch.stop)

    def write_protocol(self, document=None):
        self.protocol_path.write_text(json.dumps(document or self.document), encoding="utf-8")

    def test_valid_frozen_manifest_is_read_only(self):
        self.write_protocol()
        before = {p.relative_to(self.lab): p.read_bytes() for p in self.lab.rglob("*") if p.is_file()}
        self.assertEqual(comparison.checked_protocol(self.protocol_path), self.document)
        after = {p.relative_to(self.lab): p.read_bytes() for p in self.lab.rglob("*") if p.is_file()}
        self.assertEqual(before, after)

    def test_unfrozen_protocol_is_rejected(self):
        self.document["status"] = "draft"
        self.write_protocol()
        with self.assertRaisesRegex(ValueError, "frozen"):
            comparison.checked_protocol(self.protocol_path)

    def test_generation_drift_is_rejected(self):
        for key, value in (("do_sample", True), ("max_new_tokens", 127), ("enable_thinking", True)):
            with self.subTest(key=key):
                document = deepcopy(self.document)
                document["generation"][key] = value
                self.write_protocol(document)
                with self.assertRaises(ValueError):
                    comparison.checked_protocol(self.protocol_path)

    def test_code_hash_drift_is_rejected(self):
        self.write_protocol()
        (self.lab / comparison.CODE[0]).write_bytes(b"changed evaluator")
        with self.assertRaisesRegex(ValueError, "code changed"):
            comparison.checked_protocol(self.protocol_path)

    def test_omitting_code_hashes_cannot_bypass_freeze(self):
        for remove_all in (False, True):
            with self.subTest(remove_all=remove_all):
                document = deepcopy(self.document)
                if remove_all:
                    document["code_sha256"].clear()
                else:
                    document["code_sha256"].pop(comparison.CODE[0])
                self.write_protocol(document)
                with self.assertRaisesRegex(ValueError, "every evaluator source"):
                    comparison.checked_protocol(self.protocol_path)

    def test_missing_variant_or_split_cannot_silently_shrink_comparison(self):
        for field, key in (("variants", "half"), ("benchmarks", "recall")):
            with self.subTest(field=field):
                document = deepcopy(self.document)
                document[field].pop(key)
                self.write_protocol(document)
                with self.assertRaises(ValueError):
                    comparison.checked_protocol(self.protocol_path)

    def test_variant_roles_cannot_be_swapped_or_silently_disabled(self):
        document = deepcopy(self.document)
        document["variants"]["baseline"] = deepcopy(document["variants"]["original"])
        self.write_protocol(document)
        with self.assertRaisesRegex(ValueError, "Baseline must not"):
            comparison.checked_protocol(self.protocol_path)
        for variant in ("original", "half"):
            with self.subTest(variant=variant):
                document = deepcopy(self.document)
                document["variants"][variant]["adapter"] = None
                self.write_protocol(document)
                with self.assertRaisesRegex(ValueError, "Missing adapter"):
                    comparison.checked_protocol(self.protocol_path)

    def test_omitting_adapter_weight_or_config_hash_is_rejected(self):
        for variant in ("original", "half"):
            for name in ("adapter_config.json", "adapter_model.safetensors", None):
                with self.subTest(variant=variant, missing=name):
                    document = deepcopy(self.document)
                    hashes = document["variants"][variant]["adapter_sha256"]
                    hashes.clear() if name is None else hashes.pop(name)
                    self.write_protocol(document)
                    with self.assertRaisesRegex(ValueError, "configuration and weights"):
                        comparison.checked_protocol(self.protocol_path)

    def test_each_benchmark_hash_is_checked(self):
        self.write_protocol()
        for split, record in self.document["benchmarks"].items():
            with self.subTest(split=split):
                path = self.lab / record["path"]
                original = path.read_bytes()
                try:
                    path.write_bytes(original + b"changed")
                    with self.assertRaisesRegex(ValueError, "Benchmark changed: " + split):
                        comparison.checked_protocol(self.protocol_path)
                finally:
                    path.write_bytes(original)

    def test_each_adapter_weights_and_configuration_hash_is_checked(self):
        self.write_protocol()
        for variant in ("original", "half"):
            record = self.document["variants"][variant]
            for name in ("adapter_config.json", "adapter_model.safetensors"):
                with self.subTest(variant=variant, name=name):
                    path = self.lab / record["adapter"] / name
                    original = path.read_bytes()
                    try:
                        path.write_bytes(original + b"changed")
                        with self.assertRaisesRegex(ValueError, "Adapter changed: " + variant):
                            comparison.checked_protocol(self.protocol_path)
                    finally:
                        path.write_bytes(original)

    def test_missing_hashed_weights_fail_closed(self):
        self.write_protocol()
        (self.lab / self.document["variants"]["half"]["adapter"] / "adapter_model.safetensors").unlink()
        with self.assertRaises(FileNotFoundError):
            comparison.checked_protocol(self.protocol_path)


class LoadedTensorIntegrityTests(unittest.TestCase):
    def test_identical_values_have_stable_digest_independent_of_mapping_order(self):
        saved = {"layer.lora_A.weight": torch.tensor([[1., -2.], [.5, 3.]]),
                 "layer.lora_B.weight": torch.tensor([[.25], [-.75]])}
        loaded = {key: value.clone().requires_grad_() for key, value in reversed(list(saved.items()))}
        first = comparison.compare_loaded_tensors(loaded, saved)
        second = comparison.compare_loaded_tensors(dict(reversed(list(loaded.items()))), saved)
        self.assertEqual(first, second)
        self.assertEqual(first["tensor_count"], 2)
        self.assertEqual(len(first["loaded_state_sha256"]), 64)
        self.assertTrue(all(row["exact_after_dtype_conversion"] and row["max_abs_error"] == 0
                            for row in first["tensors"]))
        self.assertTrue(all(value.requires_grad for value in loaded.values()))

    def test_one_changed_weight_is_not_hidden_by_other_matching_tensors(self):
        saved = {"a": torch.ones((2, 3)), "b": torch.zeros((3, 2))}
        loaded = {key: value.clone() for key, value in saved.items()}
        loaded["b"][1, 0] = .125
        with self.assertRaisesRegex(RuntimeError, "values do not match.*b"):
            comparison.compare_loaded_tensors(loaded, saved)

    def test_same_element_count_with_wrong_shape_is_rejected(self):
        source = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        with self.assertRaisesRegex(RuntimeError, "Invalid loaded tensor"):
            comparison.compare_loaded_tensors({"a": source.reshape(3, 2)}, {"a": source})

    def test_missing_extra_and_renamed_keys_are_rejected(self):
        source = torch.ones((2, 2))
        for loaded in ({}, {"a": source, "extra": source}, {"renamed": source}):
            with self.subTest(keys=list(loaded)):
                with self.assertRaisesRegex(RuntimeError, "keys differ"):
                    comparison.compare_loaded_tensors(loaded, {"a": source})

    def test_documented_float_dtype_conversion_is_compared_after_cast(self):
        source = torch.tensor([[.1, -.3], [1.2345, 6.789]], dtype=torch.float32)
        original = source.clone()
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                actual = source.to(dtype)
                self.assertFalse(torch.equal(actual.float(), source))
                result = comparison.compare_loaded_tensors({"a": actual}, {"a": source})
                self.assertEqual(result["tensors"][0]["loaded_dtype"], str(dtype))
                self.assertEqual(result["tensors"][0]["saved_dtype"], "torch.float32")
                self.assertEqual(result["tensors"][0]["max_abs_error"], 0)
                altered = actual.clone()
                altered[0, 0] += .125
                with self.assertRaisesRegex(RuntimeError, "values do not match"):
                    comparison.compare_loaded_tensors({"a": altered}, {"a": source})
        self.assertTrue(torch.equal(source, original))

    def test_noncontiguous_tensor_with_equal_values_is_valid(self):
        source = torch.arange(6, dtype=torch.float32).reshape(2, 3).transpose(0, 1)
        self.assertFalse(source.is_contiguous())
        result = comparison.compare_loaded_tensors({"a": source}, {"a": source.contiguous()})
        self.assertEqual(result["tensors"][0]["shape"], [3, 2])
        self.assertTrue(result["tensors"][0]["exact_after_dtype_conversion"])

    def test_nonfinite_loaded_or_saved_weights_are_rejected(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=value):
                bad = torch.tensor([[value]])
                with self.assertRaisesRegex(RuntimeError, "Invalid loaded tensor"):
                    comparison.compare_loaded_tensors({"a": bad}, {"a": bad.clone()})
                with self.assertRaisesRegex(RuntimeError, "values do not match"):
                    comparison.compare_loaded_tensors({"a": torch.zeros((1, 1))}, {"a": bad})


class BaseFingerprintTests(unittest.TestCase):
    @staticmethod
    def fake_model(state):
        class StateOnly:
            def state_dict(self):
                return state
        return StateOnly()

    def test_peft_wrapper_and_lora_do_not_change_base_fingerprint(self):
        base = {"model.layer.q_proj.weight": torch.tensor([[1, 2], [3, 4]], dtype=torch.uint8),
                "model.layer.q_proj.weight.quant_state.bitsandbytes__nf4": torch.tensor([5, 6], dtype=torch.uint8),
                "model.layer.q_proj.weight.absmax": torch.tensor(1.5)}
        wrapped = {"base_model.model." + name.replace("q_proj.", "q_proj.base_layer."): value.clone()
                   for name, value in base.items()}
        wrapped["base_model.model.model.layer.q_proj.lora_A.default.weight"] = torch.ones((2, 3))
        wrapped["base_model.model.model.layer.q_proj.lora_B.default.weight"] = torch.ones((3, 2))
        self.assertEqual(comparison.base_weight_fingerprint(self.fake_model(base), False),
                         comparison.base_weight_fingerprint(self.fake_model(wrapped), True))

    def test_quantization_state_changes_are_detected(self):
        state = {"model.q_proj.weight": torch.tensor([[1, 2]], dtype=torch.uint8),
                 "model.q_proj.weight.absmax": torch.tensor([.5, 1.])}
        original = comparison.base_weight_fingerprint(self.fake_model(state), False)
        changed = {key: value.clone() for key, value in state.items()}
        changed["model.q_proj.weight.absmax"][0] = .75
        modified = comparison.base_weight_fingerprint(self.fake_model(changed), False)
        self.assertNotEqual(original["sha256"], modified["sha256"])
        self.assertEqual(original["tensors"]["model.q_proj.weight"], modified["tensors"]["model.q_proj.weight"])


if __name__ == "__main__":
    unittest.main()
