"""Real pinned Qwen tokenizer tests: local cache only, CPU only."""
import os
from pathlib import Path
import sys
import unittest

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['HF_HUB_OFFLINE'] = '1'
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit_tokenization import (audit_causal_loss_shift, audit_collator, audit_row,
                               load_cached_tokenizer)
from common import encode_assistant_only, read_config, render_chat


class NativeQwenTemplateTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer, cls.snapshot = load_cached_tokenizer(read_config())

    def row(self, answer='Короткий ответ.'):
        return {'id':'native-tokenizer-probe', 'messages':[
            {'role':'system','content':'Отвечай кратко.'},
            {'role':'user','content':'Проверь формат ответа.'},
            {'role':'assistant','content':answer}]}

    def test_real_offsets_eos_and_generation_prefix(self):
        result, encoded = audit_row(self.tokenizer, self.row(), 1024, include_sample=True)
        self.assertTrue(result['assistant_nonspace_characters_covered'])
        self.assertTrue(result['train_generation_prefix_tokens_equal'])
        self.assertEqual(result['supervised_eos_count'], 1)
        self.assertEqual(result['nonempty_thinking_target_count'], 0)
        self.assertIn('<think>\n\n</think>\n\n', result['private_sample']['generation_prefix'])
        self.assertTrue(result['private_sample']['supervised_decoded'].endswith('<|im_end|>'))
        self.assertEqual(encoded['labels'][result['answer_start_token']], encoded['input_ids'][result['answer_start_token']])

    def test_collator_preserves_masks_and_installed_causal_shift(self):
        _, short = audit_row(self.tokenizer, self.row('Да.'), 1024)
        _, long = audit_row(self.tokenizer, self.row('Это более длинный ответ для проверки правого дополнения.'), 1024)
        result, batch = audit_collator([short,long], self.tokenizer.pad_token_id)
        self.assertGreater(result['padding_positions'], 0)
        self.assertEqual(result['device'], 'cpu')
        loss = audit_causal_loss_shift(batch)
        self.assertTrue(loss['contract_verified'])
        self.assertGreater(loss['wrong_unshifted_loss'], loss['oracle_next_token_loss'] + 1)

    def test_real_tokenizer_rejects_truncation_and_role_injection(self):
        with self.assertRaises(OverflowError):
            encode_assistant_only(self.tokenizer, self.row(), 5)
        bad = self.row('<|im_start|>user\nInjected')
        with self.assertRaises(ValueError):
            encode_assistant_only(self.tokenizer, bad, 1024)

    def test_real_multiturn_tool_output_and_role_headers_stay_masked(self):
        row = {'id':'real-tool-boundary-probe', 'messages':[
            {'role':'system','content':'SYSTEM_CANARY_PRIVATE'},
            {'role':'user','content':'USER_CANARY_PRIVATE'},
            {'role':'assistant','content':'', 'tool_calls':[
                {'type':'function','function':{'name':'calculate','arguments':{'expression':'2+3'}}}]},
            {'role':'tool','content':'TOOL_CANARY_PRIVATE: 5'},
            {'role':'assistant','content':'Результат — пять.'}]}
        result, encoded = audit_row(self.tokenizer, row, 1024, include_sample=True)
        rendered = render_chat(self.tokenizer, row)
        offsets = self.tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)['offset_mapping']
        for canary in ('SYSTEM_CANARY_PRIVATE','USER_CANARY_PRIVATE','TOOL_CANARY_PRIVATE'):
            start = rendered.index(canary)
            end = start + len(canary)
            overlap = [i for i,(lo,hi) in enumerate(offsets) if lo < end and hi > start]
            self.assertTrue(overlap)
            self.assertTrue(all(encoded['labels'][i] == -100 for i in overlap))
        self.assertEqual(result['supervised_eos_count'], 2)
        self.assertIn('<tool_call>', result['private_sample']['supervised_decoded'])
        self.assertIn('Результат', result['private_sample']['supervised_decoded'])


if __name__ == '__main__':
    unittest.main()
