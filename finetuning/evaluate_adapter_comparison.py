"""Reload and compare fixed adapters without inventing training provenance for edited weights."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import time

from common import LAB, NvidiaMonitor, file_hash, load_model, read_jsonl, render_chat, write_json
from evaluate_diagnostics import generation_row, raw_generate, teacher_forced_nll, tokenizer_audit, token_hash
from evaluate_large import check_rows, lexical_answers, nll_row
from train_large import checked_config

CODE = ('evaluate_adapter_comparison.py', 'common.py', 'evaluate_large.py',
        'evaluate_diagnostics.py', 'train_large.py')


def checked_protocol(path):
    document = json.loads(path.read_text(encoding='utf-8'))
    if document['status'] != 'frozen_before_comparison':
        raise ValueError('Comparison protocol must be frozen before generation')
    if document['generation'] != {'do_sample': False, 'max_new_tokens': 128, 'enable_thinking': False}:
        raise ValueError('Comparison generation must match the fixed large_v3 settings')
    if set(document['code_sha256']) != set(CODE):
        raise ValueError('Comparison must bind every evaluator source file')
    if set(document['variants']) != {'baseline', 'original', 'half'}:
        raise ValueError('All three comparison variants must be declared')
    if set(document['benchmarks']) != {'benchmark', 'recall'}:
        raise ValueError('Both fixed comparison splits must be declared')
    if document['variants']['baseline']['adapter'] is not None:
        raise ValueError('Baseline must not have an adapter')
    for name, expected in document['code_sha256'].items():
        if file_hash(LAB / name) != expected:
            raise ValueError('Comparison code changed: ' + name)
    for name, record in document['benchmarks'].items():
        if file_hash(LAB / record['path']) != record['sha256']:
            raise ValueError('Benchmark changed: ' + name)
    for variant, record in document['variants'].items():
        if record['adapter'] is None:
            if variant != 'baseline':
                raise ValueError('Missing adapter for ' + variant)
            continue
        if not {'adapter_config.json', 'adapter_model.safetensors'}.issubset(record['adapter_sha256']):
            raise ValueError('Both adapter configuration and weights must be hash-bound')
        folder = LAB / record['adapter']
        for name, expected in record['adapter_sha256'].items():
            if file_hash(folder / name) != expected:
                raise ValueError('Adapter changed: ' + variant + ':' + name)
    return document


def compare_loaded_tensors(loaded, saved):
    """Exact value check after the loader's documented dtype conversion."""
    import torch
    if set(loaded) != set(saved):
        raise RuntimeError('Loaded PEFT keys differ from the selected saved adapter')
    summary, digest = [], hashlib.sha256()
    for name in sorted(saved):
        actual = loaded[name].detach().cpu().contiguous()
        source = saved[name].detach().cpu()
        if actual.shape != source.shape or not torch.isfinite(actual).all():
            raise RuntimeError('Invalid loaded tensor: ' + name)
        expected = source.to(actual.dtype)
        exact = torch.equal(actual, expected)
        error = (actual.float() - expected.float()).abs().max().item()
        summary.append({'name': name, 'saved_dtype': str(source.dtype), 'loaded_dtype': str(actual.dtype),
                        'shape': list(actual.shape), 'exact_after_dtype_conversion': exact, 'max_abs_error': error})
        digest.update(name.encode('utf-8'))
        digest.update(str(actual.dtype).encode('ascii'))
        digest.update(actual.view(torch.uint8).numpy().tobytes())
        if not exact:
            raise RuntimeError('Loaded adapter values do not match selected file: ' + name)
    return {'tensor_count': len(summary), 'loaded_state_sha256': digest.hexdigest(), 'tensors': summary}


def runtime_adapter_proof(model, adapter, tokenizer, probes, output):
    import numpy as np
    import torch
    from peft import PeftModel, get_peft_model_state_dict
    from safetensors.torch import load_file
    from common import render_chat
    active = list(model.active_adapters) if isinstance(model, PeftModel) else []
    layers = [module for module in model.modules() if hasattr(module, 'lora_A') and hasattr(module, 'lora_B')]
    proof = {'is_peft_model': isinstance(model, PeftModel), 'active_adapters': active, 'lora_layers': len(layers)}
    if adapter:
        if len(active) != 1 or not layers:
            raise RuntimeError('Selected adapter is not active')
        if any(layer.disable_adapters or layer.merged_adapters for layer in layers):
            raise RuntimeError('Adapter is disabled or unexpectedly merged')
        proof['state'] = compare_loaded_tensors(get_peft_model_state_dict(model, adapter_name=active[0]),
                                                load_file(str(adapter / 'adapter_model.safetensors'), device='cpu'))
        proof['scaling'] = sorted({float(layer.scaling[active[0]]) for layer in layers})
    elif active or layers:
        raise RuntimeError('Baseline unexpectedly contains a LoRA adapter')
    arrays, checks = {}, []
    for index, row in enumerate(probes):
        prompt = render_chat(tokenizer, generation_row(row), generation=True)
        inputs = tokenizer(prompt, add_special_tokens=False, return_tensors='pt').to(model.device)
        def logits():
            with torch.inference_mode():
                result = model(**inputs, use_cache=False, logits_to_keep=1).logits[0, -1].float().detach().cpu()
            if not torch.isfinite(result).all():
                raise RuntimeError('Nonfinite diagnostic logits')
            return result
        enabled = logits()
        arrays[f'probe_{index}_enabled'] = enabled.numpy()
        check = {'id': row['id'], 'input_token_ids': inputs['input_ids'][0].detach().cpu().tolist(),
                 'enabled_argmax': int(enabled.argmax())}
        if adapter:
            with model.disable_adapter():
                disabled_count = sum(bool(layer.disable_adapters) for layer in layers)
                if disabled_count != len(layers):
                    raise RuntimeError('disable_adapter did not disable every LoRA layer')
                disabled = logits()
            restored = logits()
            delta = enabled - disabled
            check.update(disabled_layers=disabled_count, disabled_argmax=int(disabled.argmax()),
                         enabled_disabled_max_abs=float(delta.abs().max()),
                         enabled_disabled_rms=float(delta.square().mean().sqrt()),
                         restored_max_abs=float((restored-enabled).abs().max()),
                         reenabled_layers=sum(not layer.disable_adapters for layer in layers))
            if check['restored_max_abs'] > 1e-6 or check['reenabled_layers'] != len(layers):
                raise RuntimeError('Adapter was not faithfully restored before generation')
            arrays[f'probe_{index}_disabled'] = disabled.numpy()
            arrays[f'probe_{index}_restored'] = restored.numpy()
        checks.append(check)
    if adapter and not any(check['enabled_disabled_max_abs'] > 1e-6 for check in checks):
        raise RuntimeError('Adapter has no demonstrated effect on forward logits')
    np.savez_compressed(output, **arrays)
    proof.update(forward_checks=checks, logit_artifact=str(output), logit_artifact_sha256=file_hash(output))
    return proof


def base_weight_fingerprint(model, is_adapter):
    """Hash actual loaded base state, including 4-bit quantization state, without LoRA."""
    import torch
    records, digest = {}, hashlib.sha256()
    state = model.state_dict()
    for name in sorted(state):
        if '.lora_' in name:
            continue
        canonical = name
        if is_adapter and canonical.startswith('base_model.model.'):
            canonical = canonical[len('base_model.model.'):]
        canonical = canonical.replace('.base_layer.', '.')
        if canonical in records:
            raise RuntimeError('Ambiguous canonical base parameter: ' + canonical)
        value = state[name].detach().cpu().contiguous()
        raw = value.reshape(-1).view(torch.uint8).numpy().tobytes()
        records[canonical] = {'dtype': str(value.dtype), 'shape': list(value.shape),
                              'sha256': hashlib.sha256(raw).hexdigest()}
    for name in sorted(records):
        digest.update(json.dumps([name, records[name]], sort_keys=True, separators=(',', ':')).encode('utf-8'))
    return {'tensor_count': len(records), 'sha256': digest.hexdigest(), 'tensors': records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', required=True)
    parser.add_argument('--variant', choices=('baseline', 'original', 'half'), required=True)
    parser.add_argument('--proof-only', action='store_true')
    args = parser.parse_args()
    protocol_path = (LAB / args.protocol).resolve()
    folder = protocol_path.parent
    folder.relative_to((LAB / 'evaluation/reports').resolve())
    protocol = checked_protocol(protocol_path)
    variant = protocol['variants'][args.variant]
    stem = args.variant + ('_probe' if args.proof_only else '')
    output = folder / (stem + '.json')
    logit_path, vram_path = folder / (stem + '_logits.npz'), folder / (stem + '_vram.csv')
    if any(path.exists() for path in (output, logit_path, vram_path)):
        raise FileExistsError('Comparison output exists; no automatic overwrite')
    rows = []
    for split, record in protocol['benchmarks'].items():
        split_rows = read_jsonl(LAB / record['path'])
        check_rows(split_rows)
        rows.extend((split, row) for row in split_rows)
    if len({row['id'] for _, row in rows}) != len(rows):
        raise ValueError('Question IDs overlap between comparison splits')
    lookup = {row['id']: row for _, row in rows}
    probes = [lookup[item] for item in protocol['functional_probe_ids']]
    config = checked_config('config.yaml', 'lr5e-5')
    if config['model'] != protocol['base_model']:
        raise ValueError('Base model configuration changed')
    adapter = LAB / variant['adapter'] if variant['adapter'] else None
    report = {'status': 'running', 'created_unix': time.time(), 'variant': args.variant,
              'protocol_sha256': file_hash(protocol_path), 'base_model': config['model'],
              'adapter': str(adapter) if adapter else None, 'adapter_sha256': variant.get('adapter_sha256'),
              'generation': protocol['generation'], 'seed': protocol['seed'],
              'code_sha256': protocol['code_sha256'], 'results': []}
    write_json(output, report)
    with NvidiaMonitor(vram_path) as monitor:
        model, tokenizer = load_model(config, adapter=adapter)
        import torch
        torch.manual_seed(protocol['seed'])
        torch.cuda.manual_seed_all(protocol['seed'])
        model.eval()
        report['tokenizer'] = tokenizer_audit(tokenizer)
        effective_generation = model.generation_config.to_dict()
        effective_generation.update(max_new_tokens=128, do_sample=False, use_cache=True,
                                    pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        report['effective_generation_config'] = effective_generation
        for _, row in rows:
            rendered = render_chat(tokenizer, generation_row(row), generation=True)
            ids = tokenizer(rendered, add_special_tokens=False)['input_ids']
            if len(ids) > 1024 or token_hash(ids) != protocol['input_token_ids_sha256'][row['id']]:
                raise ValueError('Rendered input differs from frozen baseline: ' + row['id'])
        report['all_frozen_input_ids_verified'] = True
        report['runtime'] = {'python': platform.python_version(), 'torch_cuda': torch.version.cuda,
                             'gpu': torch.cuda.get_device_name(0), 'packages': {
                                 name: importlib.metadata.version(name) for name in
                                 ('torch', 'unsloth', 'unsloth_zoo', 'transformers', 'peft', 'bitsandbytes', 'safetensors')}}
        report['loaded_base_weight_fingerprint'] = base_weight_fingerprint(model, bool(adapter))
        report['runtime_adapter_proof'] = runtime_adapter_proof(model, adapter, tokenizer, probes, logit_path)
        write_json(output, report)
        print(json.dumps({'event': 'adapter_proof_passed', 'variant': args.variant}), flush=True)
        if not args.proof_only:
            for split, row in rows:
                generated = raw_generate(model, tokenizer, row, protocol['generation']['max_new_tokens'])
                if len(generated['input_token_ids']) > config['model']['max_seq_length']:
                    raise ValueError('Generation input exceeded sequence limit')
                report['results'].append({'id': row['id'], 'split': split, 'prompt': row['prompt'],
                                          'metadata': row['metadata'], **generated,
                                          'lexical': lexical_answers(generated['response'], row['metadata']),
                                          'teacher_forced': teacher_forced_nll(model, tokenizer, nll_row(row), 1024)})
                write_json(output, report)
                print(json.dumps({'event': 'generated', 'variant': args.variant, 'completed': len(report['results'])}), flush=True)
    report.update(status='proof_complete' if args.proof_only else 'complete', finished_unix=time.time(), nvidia=monitor.summary())
    write_json(output, report)


if __name__ == '__main__':
    main()
