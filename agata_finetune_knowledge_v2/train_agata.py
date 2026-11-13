"""AGATA first local LoRA experiment. Does NOT change the GitHub repo or Ollama model."""
import os
from pathlib import Path

import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

ROOT = Path(__file__).resolve().parent
MODEL = os.getenv('AGATA_BASE_MODEL', 'Qwen/Qwen3.5-0.8B')
SEQ_LEN = 512

if not torch.cuda.is_available():
    raise RuntimeError('CUDA unavailable: check NVIDIA driver and PyTorch CUDA install.')
print(f'GPU: {torch.cuda.get_device_name(0)}; model: {MODEL}')

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL,
    max_seq_length=SEQ_LEN,
    load_in_4bit=False,  # Qwen3.5: 16-bit LoRA preferred over QLoRA
    load_in_16bit=True,
    full_finetuning=False,
)
model = FastLanguageModel.get_peft_model(
    model,
    r=8,
    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
    lora_alpha=8,
    lora_dropout=0,
    bias='none',
    use_gradient_checkpointing='unsloth',
    random_state=3407,
    max_seq_length=SEQ_LEN,
)

train = load_dataset('json', data_files=str(ROOT/'dataset'/'train.jsonl'), split='train')

def chat_to_text(example):
    return {'text': tokenizer.apply_chat_template(example['messages'], tokenize=False, add_generation_prompt=False)}

train = train.map(chat_to_text, remove_columns=train.column_names)
print(f'Training examples: {len(train)}')

trainer = SFTTrainer(
    model=model,
    train_dataset=train,
    processing_class=tokenizer,
    args=SFTConfig(
        dataset_text_field='text',
        max_length=SEQ_LEN,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        learning_rate=2e-4,
        warmup_steps=3,
        logging_steps=2,
        optim='adamw_8bit',
        output_dir=str(ROOT/'outputs'/'checkpoints'),
        save_strategy='epoch',
        report_to='none',
        seed=3407,
        dataset_num_proc=1,
        packing=False,
    ),
)
trainer.train()
output = ROOT/'outputs'/'agata_lora'
model.save_pretrained(str(output))
tokenizer.save_pretrained(str(output))
print(f'LoRA adapter saved to {output}')
