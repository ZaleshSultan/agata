"""Small closed-book evaluation; run base and adapter in separate processes.
Keyword score is only a rough signal. Also read model answers manually.
"""
from pathlib import Path
import argparse, json, re
import torch
from unsloth import FastLanguageModel
from peft import PeftModel

ROOT=Path(__file__).resolve().parent
p=argparse.ArgumentParser()
p.add_argument('--adapter',action='store_true')
p.add_argument('--model',default='Qwen/Qwen3.5-0.8B')
args=p.parse_args()
if not torch.cuda.is_available():
    raise RuntimeError('CUDA is not available')
model, tok=FastLanguageModel.from_pretrained(model_name=args.model,max_seq_length=512,load_in_4bit=False,load_in_16bit=True)
if args.adapter:
    model=PeftModel.from_pretrained(model,str(ROOT/'outputs'/'agata_lora'))
model.eval()
examples=[json.loads(x) for x in (ROOT/'dataset'/'knowledge_benchmark.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()]
scores={}
for row in examples:
    prompt=row['prompt']
    tokens=tok.apply_chat_template([{'role':'user','content':prompt}],tokenize=True,add_generation_prompt=True,return_tensors='pt').to('cuda')
    with torch.inference_mode():
        out=model.generate(tokens,max_new_tokens=110,do_sample=False)
    answer=tok.decode(out[0][tokens.shape[-1]:],skip_special_tokens=True).strip()
    hits=all(key.casefold() in answer.casefold() for key in row['expected_keywords'])
    scores.setdefault(row['group'],[]).append(hits)
    print(f"[{row['group']}] {'PASS' if hits else 'CHECK'}  {prompt}\n  {answer}\n")
print('Keyword-based scores (not a robust benchmark):')
for group, marks in scores.items(): print(group, f'{sum(marks)}/{len(marks)}')
