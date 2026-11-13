"""Audit and freeze private authored components before any large-v3 model run."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from difflib import SequenceMatcher
import json
from pathlib import Path
import random
import re
import time

from common import LAB, encode_assistant_only, file_hash, lab_path, read_config, read_jsonl, write_json
from audit_tokenization import load_cached_tokenizer
from extract_book import require_private_destination
from evaluate_diagnostics import generation_row, token_hash, tokenizer_audit
from train_large import PROTOCOL_RULES, planned_steps
from run_abc import versions

SYSTEM = ('Ты — Агата, русскоязычный ИИ-ассистент. Отвечай естественно и кратко. '
          'Факты и серьёзные проблемы важнее шуток. Если сведений нет, скажи об этом. '
          'Не выдумывай выполненные действия. Здесь доступны только текстовые ответы; '
          'результаты инструментов, если они даны в условии, пересказывай как сведения из условия.')
DOMAINS = ('persona', 'book', 'unknown', 'general')
PARTITIONS = ('train', 'validation', 'test')
TASKS = {'persona','book_grounded','known_exact','known_paraphrase','unknown','entity_discrimination','false_action','general'}


def normal(value):
    return ' '.join(re.findall(r'\w+', value.casefold().replace('ё','е')))


def validate_components(rows, probes):
    if not rows or len({r['id'] for r in rows+probes}) != len(rows)+len(probes):
        raise ValueError('Empty components or duplicate IDs')
    for row in rows+probes:
        for field in ('prompt','answer'):
            text=row.get(field,'')
            if '\ufffd' in text or re.search(r'\?{3,}',text):
                raise ValueError('Possible encoding corruption in authored '+field+': '+row['id'])
    prompts, families, facts = {}, defaultdict(set), defaultdict(set)
    for row in rows:
        if row.get('domain') not in DOMAINS or row.get('partition') not in PARTITIONS:
            raise ValueError('Unknown domain/partition')
        if not row.get('family_id') or not row.get('prompt','').strip() or not row.get('answer','').strip():
            raise ValueError('Missing family, question or answer')
        if row.get('metadata',{}).get('review_status') != 'approved':
            raise ValueError('Unreviewed component example: '+row['id'])
        key = normal(row['prompt'])
        if key in prompts:
            old = prompts[key]
            kind = 'conflicting targets' if normal(old['answer']) != normal(row['answer']) else 'duplicate question'
            raise ValueError(f'{kind}: {old["id"]}, {row["id"]}')
        prompts[key] = row
        families[row['family_id']].add(row['partition'])
        if row.get('fact_id'):
            facts[row['fact_id']].add(row['partition'])
    if any(len(v)>1 for v in families.values()):
        raise ValueError('Thematic family crosses independent splits')
    if any(len(v)>1 for v in facts.values()):
        raise ValueError('Book proposition crosses independent splits')
    train = {normal(r['prompt']):r for r in rows if r['partition']=='train'}
    train_facts = {r['fact_id'] for r in rows if r['partition']=='train' and r.get('fact_id')}
    seen_probes = set()
    for row in probes:
        kind = row.get('probe_type',row.get('metadata',{}).get('task_type'))
        key = normal(row['prompt'])
        if key in seen_probes:
            raise ValueError('Duplicate probe question')
        seen_probes.add(key)
        if kind == 'known_exact':
            if key not in train or normal(train[key]['answer']) != normal(row['answer']):
                raise ValueError('Exact recall probe must reproduce a training question/answer')
        elif key in prompts:
            raise ValueError('Non-exact probe leaks an existing question')
        if kind in ('known_exact','known_paraphrase') and row.get('fact_id') not in train_facts:
            raise ValueError('Known recall fact was not trained')
    return {'independent_rows':len(rows),'independent_questions_unique':True,
            'same_prompt_conflicting_targets':0,'thematic_families':len(families),
            'family_split_overlap':0,'fact_split_overlap':0,
            'intentional_exact_question_overlap':sum(r.get('probe_type',r.get('metadata',{}).get('task_type'))=='known_exact' for r in probes),
            'families_by_split':{s:sorted(k for k,v in families.items() if s in v) for s in PARTITIONS},
            'fact_ids_by_split':{s:sorted(k for k,v in facts.items() if s in v) for s in PARTITIONS}}


def check_book_evidence(rows, chunks):
    by_id = {c['chunk_id']:c for c in chunks}
    checked = []
    for row in rows:
        if row['domain'] != 'book':
            continue
        meta = row['metadata']
        chunk = by_id.get(meta.get('evidence_chunk_id'))
        if not chunk or not meta.get('evidence_quote'):
            raise ValueError('Book evidence missing: '+row['id'])
        if meta['evidence_quote'] not in chunk['text']:
            raise ValueError('Book evidence not verbatim in its source chunk: '+row['id'])
        if chunk['page'] not in meta['source_pages'] or chunk['source']['source_id'] != meta['source_id']:
            raise ValueError('Book source/page provenance mismatch: '+row['id'])
        checked.append(row['id'])
    return {'exact_evidence_checks':len(checked),'all_passed':True,
            'limitations':'Exact source matching establishes provenance, not semantic truth. All targets additionally receive source-based AI review.'}


def standardized(row, evaluation=False):
    meta = deepcopy(row['metadata'])
    domain = row['domain']
    meta.update(domain=domain, family_id=row['family_id'], partition=row['partition'],
                reference_answer=row['answer'], reference=row['answer'])
    if row.get('fact_id'):
        meta['fact_id'] = row['fact_id']
    if row.get('rubric'):
        meta['rubric'] = row['rubric']
    if 'rubric' not in meta:
        meta['rubric'] = 'Смысл эталона сохранён; нет выдуманных фактов, действий и полного эха запроса; явные ограничения формата соблюдены.'
    kind = row.get('probe_type',meta.get('task_type'))
    if domain == 'persona':
        kind = 'persona'
    elif domain == 'book':
        if kind in ('entity_mismatch','entity_confusion','false_premise'):
            kind = 'entity_discrimination'
        elif row['partition'] in ('validation','test'):
            kind = 'book_grounded'
        elif row['partition']=='train':
            kind = 'known_exact'
    if kind not in TASKS:
        raise ValueError('Unsupported task type '+str(kind)+' for '+row['id'])
    meta['task_type'] = kind
    if evaluation and kind in ('known_exact','known_paraphrase'):
        meta['expected_answers'] = [row['answer']]
    messages = [{'role':'system','content':SYSTEM},{'role':'user','content':row['prompt']}]
    if not evaluation:
        messages.append({'role':'assistant','content':row['answer']})
    result = {'id':row['id'], 'category':domain, 'family_id':row['family_id'], 'tools':[],
              'messages':messages,'metadata':meta}
    if evaluation:
        result.update(prompt=row['prompt'],reference=row['answer'])
        generation_row(result)
    return result


def near_pairs(rows, threshold=.82):
    strings = [normal(r['prompt']) for r in rows]
    words = [set(s.split()) for s in strings]
    found=[]
    for i,a in enumerate(rows):
        for j in range(i+1,len(rows)):
            common = len(words[i]&words[j])/max(1,len(words[i]|words[j]))
            if common<.50:
                continue
            similarity=SequenceMatcher(None,strings[i],strings[j]).ratio()
            if similarity>=threshold:
                b=rows[j]
                found.append({'id_a':a['id'],'id_b':b['id'],'similarity':round(similarity,4),
                              'same_split':a['partition']==b['partition'],
                              'same_family':a['family_id']==b['family_id'],
                              'same_fact':bool(a.get('fact_id') and a.get('fact_id')==b.get('fact_id'))})
    return found


def build(data_dir, freeze=False):
    require_private_destination(data_dir)
    protocol_path=data_dir/'protocol.json'
    if protocol_path.exists():
        raise FileExistsError('Existing frozen protocol is never overwritten')
    components=data_dir/'components'
    source_paths=[components/(d+'.jsonl') for d in DOMAINS]
    rows=[r for p in source_paths for r in read_jsonl(p)]
    probe_path=components/'book_recall_probes.jsonl'
    probes=read_jsonl(probe_path)
    audit=validate_components(rows,probes)
    source_chunks=LAB/'books/processed/polivalina_2013/chunks.jsonl'
    audit['book_evidence']=check_book_evidence(rows+probes,read_jsonl(source_chunks))
    config=read_config()
    tokenizer,tokenizer_path=load_cached_tokenizer(config)
    corpora={split:[standardized(r) for r in rows if r['partition']==split] for split in PARTITIONS}
    random.Random(3407).shuffle(corpora['train'])
    evaluations={'evaluation_validation':[standardized(r,True) for r in rows if r['partition']=='validation'],
                 'benchmark':[standardized(r,True) for r in rows if r['partition']=='test'],
                 'recall_probes':[standardized(r,True) for r in probes]}
    audit['tokenization']={}
    for name,items in {**corpora,**evaluations}.items():
        features=[]
        for row in items:
            target=row if name in corpora else {'id':row['id'],'tools':[], 'messages':row['messages']+[{'role':'assistant','content':row['reference']}]}
            feature=encode_assistant_only(tokenizer,target,1024)
            features.append({'id':row['id'],'sequence_tokens':len(feature['input_ids']),
                             'target_tokens':sum(x!=-100 for x in feature['labels'][1:]),
                             'token_ids_sha256':token_hash(feature['input_ids'])})
        audit['tokenization'][name]={'rows':len(items),'max_sequence_tokens':max(f['sequence_tokens'] for f in features),
                                    'target_tokens':sum(f['target_tokens'] for f in features),'overlength_rows':0,
                                    'examples':features}
    audit['counts']={split:dict(Counter(r['category'] for r in items)) for split,items in corpora.items()}
    audit['evaluation_tasks']={split:dict(Counter(r['metadata']['task_type'] for r in items)) for split,items in evaluations.items()}
    audit['planned_training']=planned_steps(len(corpora['train']))
    audit['near_pairs']=near_pairs(rows)
    audit['component_sha256']={p.name:file_hash(p) for p in source_paths+[probe_path]}
    audit['actual_tokenizer']=tokenizer_audit(tokenizer)
    audit['source_chunk_sha256']=file_hash(source_chunks)
    audit['review_files']={p.name:file_hash(p) for p in components.glob('*review*.json')}
    audit['status']='preflight_passed_pending_cross_review_and_freeze'
    write_json(data_dir/'preflight.json',audit)
    if not freeze:
        return audit
    approval_path=components/'freeze_review.json'
    approval=json.loads(approval_path.read_text(encoding='utf-8'))
    if approval.get('status')!='approved' or approval.get('component_sha256')!=audit['component_sha256']:
        raise ValueError('A review approving these exact component bytes is required before freeze')
    if approval.get('reviewed_near_pairs')!=len(audit['near_pairs']):
        raise ValueError('Every flagged near-pair needs an explicit review before freeze')
    for name,items in {**corpora,**evaluations}.items():
        path=data_dir/(name+'.jsonl')
        if path.exists():
            raise FileExistsError('Prepared output already exists: '+str(path))
        path.write_bytes(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in items).encode('utf-8'))
    audit['status']='passed_before_training'
    write_json(data_dir/'data_audit.json',audit)
    protocol={'status':'frozen_before_baseline','experiment':'large_v3','created_unix':time.time(),
              'base_model':config['model'],'system_prompt':SYSTEM,'rules':PROTOCOL_RULES,
              'counts':audit['counts'],'evaluation_tasks':audit['evaluation_tasks'],'planned':audit['planned_training'],
              'component_sha256':audit['component_sha256'],'freeze_review_sha256':file_hash(approval_path),
              'source_files':{name:file_hash(LAB/name) for name in ('prepare_large.py','train_large.py','evaluate_large.py','report_large.py','common.py','train.py')},
              'prompts_py_sha256':file_hash(LAB.parent/'prompts.py'),'runtime':versions(),
              'split_interpretation':{'independent':'Train/validation/test have disjoint thematic family and book proposition IDs; validation/test book tasks provide short source context.',
                                      'recall':'Known-exact intentionally repeats12 train questions; known-paraphrase asks12 new questions about trained propositions; entity probes test substitution. Not an independent fact holdout.',
                                      'general':'New heldout thematic contexts and combinations of learned skills, not a claim of unseen-skill evaluation.'},
              'selection_policy':'Only validation loss and complete validation semantic guards choose a checkpoint. No test tuning. All saved checkpoints are subsequently evaluated on the same independent test and recall sets.',
              'publication':'Source book, derived data, answers, reviews and weights remain ignored/untracked locally.',
              'limitations':['AI-authored and cross-reviewed; no independent human certification.','Available book is a98-page preview; insufficient verified material for250 diverse book examples.','No artificial duplication to hit1000; real counts are recorded.','One training seed; tiny evaluation category counts and correlated questions limit precision.'],
              'artifact_sha256':{p.name:file_hash(p) for p in data_dir.iterdir() if p.is_file() and p.name not in ('protocol.json','protocol.sha256.json')}}
    write_json(protocol_path,protocol)
    write_json(data_dir/'protocol.sha256.json',{'protocol.json':file_hash(protocol_path)})
    return audit


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',default='datasets/large_v3')
    parser.add_argument('--freeze',action='store_true')
    args=parser.parse_args()
    audit=build(lab_path(args.data_dir),args.freeze)
    print(json.dumps({k:audit[k] for k in ('status','counts','evaluation_tasks','planned_training')},ensure_ascii=False))


if __name__=='__main__':
    main()
