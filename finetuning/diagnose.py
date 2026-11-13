"""Read-only host and isolated environment diagnostics; never imports AGATA."""
from __future__ import annotations
import argparse
import importlib.metadata as metadata
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def command(args):
    try:
        p = subprocess.run(args, capture_output=True, timeout=30)
        raw = p.stdout + p.stderr
        encoding = 'utf-16-le' if b'\x00' in raw[:100] else 'utf-8'
        return {'returncode': p.returncode, 'output': raw.decode(encoding, errors='replace').strip()}
    except Exception as exc:
        return {'error': str(exc)}

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, default=ROOT/'logs'/'environment.json')
    p.add_argument('--cuda', action='store_true', help='Import torch and perform an actual GPU matrix operation')
    args = p.parse_args()
    import psutil
    vm = psutil.virtual_memory()
    result = {'timestamp_utc': datetime.now(timezone.utc).isoformat(), 'platform': platform.platform(),
              'python': sys.version, 'executable': sys.executable, 'ram_total': vm.total,
              'ram_available': vm.available, 'disk_free': shutil.disk_usage(ROOT).free,
              'nvidia_smi': command(['nvidia-smi','--query-gpu=name,driver_version,memory.total,memory.used,memory.free','--format=csv']),
              'wsl': command(['wsl','--list','--verbose']), 'ollama': command(['ollama','list']), 'versions': {}}
    for name in ('torch','torchvision','unsloth','unsloth_zoo','transformers','trl','peft','datasets','bitsandbytes','triton-windows','huggingface-hub'):
        try: result['versions'][name] = metadata.version(name)
        except metadata.PackageNotFoundError: result['versions'][name] = None
    if args.cuda:
        try:
            import torch
            result['cuda'] = {'available': torch.cuda.is_available(), 'runtime': torch.version.cuda}
            if torch.cuda.is_available():
                a = torch.ones((10,10),device='cuda')
                value = (a@a)[0,0].item()
                result['cuda'].update(device=torch.cuda.get_device_name(), matmul_value=value,
                                      matmul_pass=value==10, bf16=torch.cuda.is_bf16_supported())
        except Exception as exc:
            result['cuda'] = {'error': repr(exc)}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))
    if args.cuda and not result.get('cuda', {}).get('matmul_pass'):
        raise SystemExit(1)

if __name__ == '__main__': main()
