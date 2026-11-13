"""Download a pinned official portable llama.cpp release into ignored lab outputs."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
from urllib.request import urlopen
import zipfile

from common import lab_path, write_json

TAG = "b12110"
COMMIT = "cd959bae99ff4bed22bef2604f45241c09bbb8c8"
ASSET = "llama-b12110-bin-win-cpu-x64.zip"
ASSET_SHA256 = "98901e327c252875accaf2d1bf364e10e868843b0b63468baad1aa945cdc5da2"
URL = f"https://github.com/ggml-org/llama.cpp/releases/download/{TAG}/{ASSET}"


def unpack(archive, destination):
    destination = destination.resolve()
    with zipfile.ZipFile(archive) as stream:
        for item in stream.infolist():
            target = (destination / item.filename).resolve()
            if not target.is_relative_to(destination) or stat.S_ISLNK(item.external_attr >> 16):
                raise ValueError("Unsafe archive member")
        stream.extractall(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install-converter", action="store_true", help="Also install isolated CPU conversion dependencies (~1 GiB); never changes the training environment")
    args = parser.parse_args()
    if os.name != "nt":
        raise SystemExit("This helper pins the Windows x64 portable release")
    root = lab_path("outputs/tooling")
    root.mkdir(parents=True, exist_ok=True)
    archive = root / ASSET
    if not archive.is_file():
        temporary = archive.with_suffix(".zip.part")
        with urlopen(URL, timeout=120) as response, temporary.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        temporary.replace(archive)
    with archive.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != ASSET_SHA256:
        raise RuntimeError("Official release checksum mismatch; refusing extraction/execution")
    binaries = root / f"llama-{TAG}-windows-cpu"
    if not binaries.exists():
        binaries.mkdir()
        unpack(archive, binaries)
    executables = list(binaries.rglob("llama-quantize.exe"))
    if len(executables) != 1:
        raise RuntimeError("Expected exactly one llama-quantize.exe")
    quantizer = executables[0].resolve()
    # A no-input invocation prints usage and checks DLL availability without a model.
    check = subprocess.run([str(quantizer), "--help"], capture_output=True, text=True)
    if "usage" not in (check.stdout + check.stderr).lower():
        raise RuntimeError(f"Portable quantizer cannot run: {check.returncode}, {check.stderr}")
    source = root / f"llama.cpp-{TAG}"
    if not source.exists():
        subprocess.run(["git", "-c", "core.longpaths=true", "clone", "--depth", "1", "--branch", TAG,
                        "https://github.com/ggml-org/llama.cpp.git", str(source)], check=True)
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if revision != COMMIT:
        raise RuntimeError("llama.cpp checkout differs from the pinned release commit")
    report = {"release": TAG, "stable_release": "v0.9.0", "commit": COMMIT,
              "archive_url": URL, "sha256": digest, "quantizer": str(quantizer),
              "source": str(source.resolve()), "quantizer_help_verified": True,
              "converter_environment_installed": False}
    previous_manifest = root / "export_tooling.json"
    if previous_manifest.is_file():
        previous = json.loads(previous_manifest.read_text(encoding="utf-8"))
        previous_python = previous.get("converter_python")
        if previous.get("commit") == COMMIT and previous_python and Path(previous_python).is_file():
            for key in ("converter_environment_installed", "converter_python", "converter_help_verified"):
                if key in previous:
                    report[key] = previous[key]
    if args.install_converter:
        environment = Path.home() / ".cache" / "agata-ft" / "converter-env"
        python = environment / "Scripts/python.exe"
        if not python.exists():
            subprocess.run([sys.executable, "-m", "venv", str(environment)], check=True)
        # Use the pinned source's own requirements; its torch/transformers versions
        # differ from Unsloth, hence a fully separate venv with no inherited packages.
        pinned_requirements = lab_path("requirements-export.lock.txt")
        requirements = pinned_requirements if pinned_requirements.is_file() else source / "requirements/requirements-convert_hf_to_gguf.txt"
        subprocess.run([str(python), "-m", "pip", "install", "-r", str(requirements)], check=True)
        subprocess.run([str(python), "-m", "pip", "check"], check=True)
        help_result = subprocess.run([str(python), str(source / "convert_hf_to_gguf.py"), "--help"],
                                     capture_output=True, text=True, check=True)
        (root / "converter-help.txt").write_text(help_result.stdout, encoding="utf-8")
        lock = subprocess.check_output([str(python), "-m", "pip", "freeze"], text=True)
        (root / "converter-requirements.lock.txt").write_text(lock, encoding="utf-8")
        report.update({"converter_environment_installed": True, "converter_python": str(python.resolve()),
                       "converter_help_verified": True})
    write_json(root / "export_tooling.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
