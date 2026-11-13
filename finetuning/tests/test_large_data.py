"""Refuse split leakage, incorrect recall labels and ungrounded book provenance."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from prepare_large import validate_components, check_book_evidence, standardized


def sample(identifier='x',split='train',family='f',prompt='Question?',answer='Answer.',fact=None):
    row={'id':identifier,'domain':'general','family_id':family,'partition':split,
         'prompt':prompt,'answer':answer,'metadata':{'review_status':'approved','task_type':'general'}}
    if fact:
        row['fact_id']=fact
    return row


class LargeDataTest(unittest.TestCase):
    def test_family_leak(self):
        with self.assertRaisesRegex(ValueError,'family crosses'):
            validate_components([sample(),sample('y','test',prompt='Different?')],[])

    def test_fact_leak_even_with_different_family(self):
        with self.assertRaisesRegex(ValueError,'proposition crosses'):
            validate_components([sample(fact='a'),sample('y','test','g','Different?',fact='a')],[])

    def test_conflicting_answer(self):
        with self.assertRaisesRegex(ValueError,'conflicting targets'):
            validate_components([sample(),sample('y',answer='Other')],[])

    def test_duplicate_question_after_punctuation(self):
        with self.assertRaisesRegex(ValueError,'duplicate question'):
            validate_components([sample(),sample('y',prompt='question !')],[])

    def test_exact_probe_intentional_overlap_is_allowed(self):
        row=sample(fact='a');probe=sample('probe',split='probe',fact='a')
        probe['probe_type']='known_exact'
        self.assertEqual(validate_components([row],[probe])['intentional_exact_question_overlap'],1)

    def test_paraphrase_cannot_repeat_question(self):
        row=sample(fact='a');probe=sample('probe',split='probe',fact='a')
        probe['probe_type']='known_paraphrase'
        with self.assertRaisesRegex(ValueError,'Non-exact probe'):
            validate_components([row],[probe])

    def test_unseen_fact_cannot_be_called_known(self):
        probe=sample('probe',split='probe',prompt='New?',fact='b');probe['probe_type']='known_paraphrase'
        with self.assertRaisesRegex(ValueError,'was not trained'):
            validate_components([sample(fact='a')],[probe])

    def test_generation_has_no_reference_message(self):
        row=standardized(sample(),True)
        self.assertEqual([m['role'] for m in row['messages']],['system','user'])
        self.assertNotIn('Answer.',str(row['messages']))
        self.assertEqual(row['metadata']['reference_answer'],'Answer.')

    def test_encoding_corruption_in_probe_is_rejected(self):
        probe=sample('probe',split='probe',prompt='New?',answer='?????')
        probe['probe_type']='known_paraphrase'
        with self.assertRaisesRegex(ValueError,'encoding corruption'):
            validate_components([sample()],[probe])

    def test_source_quote_must_exist_at_verified_page(self):
        row=sample();row['domain']='book'
        row['metadata'].update(evidence_chunk_id='c',evidence_quote='Made up',source_pages=[2],source_id='s')
        chunk={'chunk_id':'c','text':'Real evidence','page':2,'source':{'source_id':'s'}}
        with self.assertRaisesRegex(ValueError,'not verbatim'):
            check_book_evidence([row],[chunk])
        row['metadata']['evidence_quote']='Real evidence'
        row['metadata']['source_pages']=[3]
        with self.assertRaisesRegex(ValueError,'provenance mismatch'):
            check_book_evidence([row],[chunk])
