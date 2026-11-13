"""Download one pinned, small 4-bit model to the user's Hugging Face cache."""
import json
from pathlib import Path
from huggingface_hub import snapshot_download

MODEL = 'unsloth/Qwen3-1.7B-bnb-4bit'
REVISION = 'd1d7781199f38692496fab70012c39598f0c858f'

def main():
    path = snapshot_download(MODEL,revision=REVISION,
        allow_patterns=['*.json','*.safetensors','*.jinja','*.txt','*.md'],max_workers=2)
    result = {'model':MODEL,'revision':REVISION,'snapshot':path}
    output = Path(__file__).resolve().parent/'logs'/'model-download.json'
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__ == '__main__': main()
