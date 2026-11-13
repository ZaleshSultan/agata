"""CPU-only audit of the exact cached Qwen3 tokenizer and lab supervision.

No model weights are loaded, no dataset is modified, and no network is used.
The JSON report contains private dataset excerpts and must remain ignored.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import re
import time

from common import (LAB, LabelPreservingCollator, encode_assistant_only, file_hash,
                    lab_path, read_config, read_jsonl, render_chat)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def load_cached_tokenizer(config, local_path=None):
    """The revision must already exist locally; never silently select main."""
    from transformers import AutoTokenizer
    if local_path is None:
        from huggingface_hub import snapshot_download
        local_path = snapshot_download(config['model']['name'],
                                      revision=config['model']['revision'], local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(str(local_path), local_files_only=True,
                                             trust_remote_code=False, use_fast=True)
    assert tokenizer.is_fast, 'Real offset mappings require the fast tokenizer'
    tokenizer.padding_side = 'right'
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, Path(local_path)


def tokenizer_identity(tokenizer, path):
    return {'path': str(path), 'class': type(tokenizer).__name__, 'is_fast': tokenizer.is_fast,
            'vocab_size': len(tokenizer), 'padding_side': tokenizer.padding_side,
            'eos_token': tokenizer.eos_token, 'eos_token_id': tokenizer.eos_token_id,
            'pad_token': tokenizer.pad_token, 'pad_token_id': tokenizer.pad_token_id,
            'bos_token_id': tokenizer.bos_token_id,
            'chat_template_sha256': hashlib.sha256(str(tokenizer.chat_template).encode()).hexdigest(),
            'tokenizer_files_sha256': {name: file_hash(path / name) for name in
                ('tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja', 'special_tokens_map.json')
                if (path / name).is_file()}}


def audit_row(tokenizer, row, max_length, include_sample=False):
    """Check real character offsets, target coverage and the generation prefix."""
    rendered = render_chat(tokenizer, row)
    encoded = encode_assistant_only(tokenizer, row, max_length)
    actual = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    assert encoded['input_ids'] == actual['input_ids']
    matches = list(re.finditer(r'<\|im_start\|>assistant\n(.*?)<\|im_end\|>', rendered, re.DOTALL))
    spans = [(m.start(1), m.end()) for m in matches]
    active = [i for i, label in enumerate(encoded['labels']) if label != -100]
    assert len(matches) == sum(m['role'] == 'assistant' for m in row['messages'])
    assert active and active[0] > 0
    assert all(encoded['labels'][i] == encoded['input_ids'][i] for i in active)
    expected = [i for i, (lo, hi) in enumerate(actual['offset_mapping']) if hi > lo
                and any(lo >= start and hi <= end for start, end in spans)]
    assert active == expected
    assert all(encoded['labels'][i] == -100 for i in range(len(encoded['labels'])) if i not in active)

    # Independent coverage check: every non-space character of every assistant
    # body + closing role token must be covered by a supervised real offset.
    covered = {c for i in active for c in range(*actual['offset_mapping'][i])}
    missing = [c for lo, hi in spans for c in range(lo, hi)
               if not rendered[c].isspace() and c not in covered]
    assert not missing, f'Assistant text lost at offsets {missing[:10]}'
    eos = [i for i in active if encoded['labels'][i] == tokenizer.eos_token_id]
    assert len(eos) == len(matches), 'One supervised EOS is required per assistant'
    assert tokenizer.convert_ids_to_tokens(tokenizer.eos_token_id) == '<|im_end|>'
    assert all(encoded['labels'][i + 1] == -100 for i in eos if i + 1 < len(encoded['labels']))
    assert all(encoded['labels'][i] != tokenizer.convert_tokens_to_ids('<|im_start|>') for i in active)

    # ABC uses a final assistant answer. Compare the exact prefix characters and
    # token IDs to what generate() actually places before its first new token.
    assert row['messages'][-1]['role'] == 'assistant'
    prompt_row = dict(row, messages=row['messages'][:-1])
    prefix = render_chat(tokenizer, prompt_row, generation=True)
    prefix_ids = tokenizer(prefix, add_special_tokens=False)['input_ids']
    prefix_text_match = rendered.startswith(prefix)
    prefix_token_match = encoded['input_ids'][:len(prefix_ids)] == prefix_ids
    assert prefix_text_match, 'Train/inference prefix character mismatch'
    assert prefix_token_match, 'Train/inference prefix token mismatch'
    answer_start = len(prefix_ids)
    assert answer_start < len(encoded['input_ids'])
    assert encoded['labels'][answer_start] == encoded['input_ids'][answer_start]
    nonempty_think = re.findall(r'<think>\s*(\S.*?)\s*</think>', rendered, re.DOTALL)
    assert not nonempty_think, 'Unexpected reasoning target in non-thinking ABC data'
    prefix_active = [i for i in active if i < answer_start]
    result = {'id': row.get('id'), 'tokens': len(encoded['input_ids']),
              'supervised_tokens': len(active), 'assistant_turns': len(matches),
              'supervised_eos_count': len(eos), 'supervised_prefix_tokens': len(prefix_active),
              'answer_start_token': answer_start, 'first_answer_logit_index': answer_start - 1,
              'train_generation_prefix_characters_equal': prefix_text_match,
              'train_generation_prefix_tokens_equal': prefix_token_match,
              'assistant_nonspace_characters_covered': not missing,
              'nonempty_thinking_target_count': len(nonempty_think),
              'encoding_sha256': digest(encoded)}
    if include_sample:
        result['private_sample'] = {
            'rendered': rendered, 'generation_prefix': prefix,
            'supervised_decoded': tokenizer.decode([encoded['labels'][i] for i in active], skip_special_tokens=False),
            'generation_prefix_supervised_decoded': tokenizer.decode([encoded['labels'][i] for i in prefix_active], skip_special_tokens=False),
            'token_table': [{'i': i, 'token_id': token, 'token': tokenizer.convert_ids_to_tokens(token),
                             'decoded_piece': tokenizer.decode([token]),
                             'offset': list(actual['offset_mapping'][i]), 'label': encoded['labels'][i],
                             'predicted_by_logit_at': i - 1 if encoded['labels'][i] != -100 else None}
                            for i, token in enumerate(encoded['input_ids'])]}
    return result, encoded


def audit_collator(features, pad_token_id):
    """Use the production collator on different actual sequence lengths."""
    import torch
    batch = LabelPreservingCollator(pad_token_id)(features)
    assert all(t.device.type == 'cpu' and t.dtype == torch.long for t in batch.values())
    width = max(len(f['input_ids']) for f in features)
    for j, feature in enumerate(features):
        length = len(feature['input_ids'])
        assert batch['input_ids'][j, :length].tolist() == feature['input_ids']
        assert batch['labels'][j, :length].tolist() == feature['labels']
        assert batch['attention_mask'][j, :length].tolist() == feature['attention_mask']
        assert batch['input_ids'][j, length:].tolist() == [pad_token_id] * (width - length)
        assert batch['labels'][j, length:].tolist() == [-100] * (width - length)
        assert batch['attention_mask'][j, length:].tolist() == [0] * (width - length)
    return {'batch_size': len(features), 'width': width,
            'padding_positions': int((batch['attention_mask'] == 0).sum()),
            'padding_labels_ignored': True, 'active_labels_preserved': True,
            'device': str(batch['input_ids'].device)}, batch


def audit_causal_loss_shift(batch):
    """Verify HF's installed causal-loss contract with real label-mask positions.

    Actual vocabulary IDs are remapped to seven toy classes, avoiding allocation
    of a CPU tensor with the 151,669-token vocabulary. Mask locations and sequence
    lengths are preserved. This tests the installed loss function's shift, not
    Unsloth's GPU kernel; GPU parity is a separate diagnostic.
    """
    import torch
    import torch.nn.functional as F
    from transformers.loss.loss_utils import ForCausalLMLoss
    labels = batch['labels'].clone()
    active = labels != -100
    indices = torch.arange(labels.shape[1]).unsqueeze(0).expand_as(labels)
    toy_labels = torch.where(active, indices % 7, -100)
    shifted = F.pad(toy_labels, (0, 1), value=-100)[..., 1:].contiguous()
    logits = torch.full((*toy_labels.shape, 7), -12.0)
    for b, t in (shifted != -100).nonzero().tolist():
        logits[b, t, int(shifted[b, t])] = 12.0
    installed = ForCausalLMLoss(logits, toy_labels, vocab_size=7)
    manual = F.cross_entropy(logits.reshape(-1, 7), shifted.reshape(-1), ignore_index=-100)
    wrong_unshifted = F.cross_entropy(logits.reshape(-1, 7), toy_labels.reshape(-1), ignore_index=-100)
    assert torch.allclose(installed, manual, atol=1e-7)
    assert float(installed) < 1e-5
    assert float(wrong_unshifted) > 1.0
    assert not active[:, 0].any()
    return {'installed_function': 'transformers.loss.loss_utils.ForCausalLMLoss',
            'function_source_sha256': hashlib.sha256(inspect.getsource(ForCausalLMLoss).encode()).hexdigest(),
            'oracle_next_token_loss': float(installed), 'manual_shifted_loss': float(manual),
            'wrong_unshifted_loss': float(wrong_unshifted), 'contract_verified': True,
            'vocabulary_remapped_classes': 7,
            'limitation': 'CPU installed HF loss tested; no GPU model or Unsloth fused loss execution in this audit.'}


def run_audit(config_path, dataset_dir, adapter_root, output_path):
    # Set before importing Transformers/Torch; other running processes are unaffected.
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    from extract_book import require_private_destination
    output = lab_path(output_path)
    require_private_destination(output)
    if output.exists():
        raise FileExistsError(f'Preserve previous diagnostic: {output}')
    config = read_config(config_path)
    tokenizer, snapshot = load_cached_tokenizer(config)
    start = time.time()
    report = {'status': 'running', 'created_unix': start, 'config': config['model'],
              'audit_script_sha256': file_hash(__file__), 'common_sha256': file_hash(LAB/'common.py'),
              'cpu_only': True, 'network_used': False,
              'library_versions': {name: importlib.metadata.version(name)
                  for name in ('transformers', 'tokenizers', 'torch', 'huggingface_hub')},
              'native_tokenizer': tokenizer_identity(tokenizer, snapshot),
              'datasets': {}, 'adapter_tokenizers': {}, 'findings': []}
    features = []
    all_rows = {}
    for arm in 'ABC':
        all_rows[arm] = []
        for split in ('train', 'validation'):
            path = lab_path(dataset_dir) / f'{arm}_{split}.jsonl'
            rows = read_jsonl(path)
            checked = []
            for n, row in enumerate(rows):
                result, feature = audit_row(tokenizer, row, config['model']['max_seq_length'], include_sample=n == 0)
                checked.append(result)
                features.append(feature)
                all_rows[arm].append((row, result['encoding_sha256']))
            report['datasets'][f'{arm}_{split}'] = {'sha256': file_hash(path), 'rows': checked,
                'n': len(checked), 'tokens': sum(r['tokens'] for r in checked),
                'supervised_tokens': sum(r['supervised_tokens'] for r in checked),
                'supervised_prefix_tokens': sum(r['supervised_prefix_tokens'] for r in checked),
                'all_prefixes_identical': all(r['train_generation_prefix_tokens_equal'] for r in checked)}
        saved_path = lab_path(adapter_root) / arm / 'adapter'
        saved, saved_path = load_cached_tokenizer(config, saved_path)
        comparisons = []
        for row, expected in all_rows[arm]:
            actual, _ = audit_row(saved, row, config['model']['max_seq_length'])
            comparisons.append(actual['encoding_sha256'] == expected)
        report['adapter_tokenizers'][arm] = {'identity': tokenizer_identity(saved, saved_path),
            'rows_compared': len(comparisons), 'all_encodings_identical_to_native': all(comparisons)}
        assert all(comparisons), f'Saved tokenizer {arm} differs from exact base tokenizer'
    shortest, longest = min(features, key=lambda f:len(f['input_ids'])), max(features, key=lambda f:len(f['input_ids']))
    report['collator'], batch = audit_collator([shortest, longest], tokenizer.pad_token_id)
    report['causal_shift'] = audit_causal_loss_shift(batch)
    report['findings'].append('No assistant masking, EOS, padding, tokenization or train/inference prefix mismatch reproduced on all A/B/C train and validation rows.')
    prefix_count = sum(d['supervised_prefix_tokens'] for d in report['datasets'].values())
    report['findings'].append(f'{prefix_count} supervised tokens belong to the native empty thinking prefix across checked rows. Inference supplies that identical prefix; these are redundant formatting targets, not a train/inference mismatch. No data or common.py change applied.')
    report['limitations'] = ['No model weights, CUDA execution, optimizer update or generation quality was tested.',
        'The native cached tokenizer and the saved A/B/C tokenizers are tested. Parent GPU diagnostics must compare the tokenizer returned by FastLanguageModel at inference time.',
        'HF causal-loss shift is verified on remapped labels at actual mask positions; this does not by itself verify a patched Unsloth fused GPU loss.']
    import torch
    assert not torch.cuda.is_initialized()
    report['cuda_initialized'] = torch.cuda.is_initialized()
    report['status'] = 'passed'
    report['elapsed_seconds'] = time.time() - start
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='datasets/abc_v1/configs/A.yaml')
    parser.add_argument('--dataset-dir', default='datasets/abc_v1')
    parser.add_argument('--adapter-root', default='outputs/abc_v1')
    parser.add_argument('--output', default='evaluation/reports/diagnostics_v1/tokenization_audit.json')
    args = parser.parse_args()
    report = run_audit(args.config, args.dataset_dir, args.adapter_root, args.output)
    print(json.dumps({'status': report['status'], 'rows': sum(d['n'] for d in report['datasets'].values()),
                      'cuda_initialized': report['cuda_initialized'], 'findings': report['findings']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
