"""Compare base model and trained adapter via the same prompts. Run separately from training."""
import sys
from pathlib import Path

import torch
from unsloth import FastLanguageModel
from peft import PeftModel

ROOT = Path(__file__).resolve().parent
MODEL = 'Qwen/Qwen3.5-0.8B'
PROMPTS = [
    'Агата, я проспал пару.',
    'У меня упал домашний сервер, что делать?',
    'Мне очень грустно.',
    'Чем ты можешь помочь?',
    'Я наконец закончил университетский проект!',
]
if not torch.cuda.is_available():
    raise RuntimeError('CUDA unavailable')
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL, max_seq_length=512, load_in_4bit=False, load_in_16bit=True,
)
if '--adapter' in sys.argv:
    adapter_path = ROOT/'outputs'/'agata_lora'
    if not adapter_path.exists():
        raise FileNotFoundError(f'Adapter not found: {adapter_path}')
    model = PeftModel.from_pretrained(model, str(adapter_path))
    print('Mode: AGATA LoRA')
else:
    print('Mode: unmodified base model')
model.eval()
for prompt in PROMPTS:
    inputs = tokenizer.apply_chat_template(
        [{'role':'user','content':prompt}], tokenize=True,
        add_generation_prompt=True, return_tensors='pt'
    ).to('cuda')
    with torch.inference_mode():
        result = model.generate(inputs, max_new_tokens=100, do_sample=False)
    text = tokenizer.decode(result[0][inputs.shape[-1]:], skip_special_tokens=True)
    print(f'\nUSER: {prompt}\nAGATA: {text.strip()}')
