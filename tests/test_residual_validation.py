import copy
import unittest

from experiments.e2b_step_overhead.validation import (
    validate_cells, verify_mixed_window, verify_pure_window,
)


class ResidualValidationTest(unittest.TestCase):
    def test_pure_rejects_recomputation(self):
        steps = [[dict(id='d', tokens=16, computed=0, prompt=16)],
                 [dict(id='d', tokens=1, computed=16, prompt=16)],
                 [dict(id='d', tokens=1, computed=17, prompt=16)]]
        self.assertEqual(verify_pure_window(steps, ['d'], 3)['decode_contexts'], [[16], [17]])
        steps[-1][0]['computed'] = 0
        with self.assertRaisesRegex(ValueError, 'Recomputation'):
            verify_pure_window(steps, ['d'], 3)

    def test_mixed_rejects_missing_decoder(self):
        steps = [[dict(id='d', tokens=1, computed=16, prompt=16),
                  dict(id='p', tokens=64, computed=0, prompt=128)]]
        self.assertEqual(verify_mixed_window(steps, ['d'], ['p'], 64, 1)[0]['lm_head_len'], 1)
        steps[0].pop(0)
        with self.assertRaisesRegex(ValueError, 'Decode background'):
            verify_mixed_window(steps, ['d'], ['p'], 64, 1)

    def test_rejects_duplicate_context_and_different_gpu_anchor(self):
        anchor = dict(model='model', tp=1, pc=0, n_decode=1, reps=3,
                      T_pure_L1=[1, 1, 1], T_pure_L2=[2, 2, 2],
                      decoder_prompt_len=16, dtype='bfloat16', L1=96, L2=288,
                      gpu_uuids=['A'], versions={'vllm': '0.19.0'})
        mixed = dict(copy.deepcopy(anchor), pc=64, T_mix=[3, 3, 3])
        validate_cells([anchor, mixed], 'model', 1)
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            validate_cells([anchor, dict(anchor, decoder_prompt_len=32)], 'model', 1)
        mixed['gpu_uuids'] = ['B']
        with self.assertRaisesRegex(ValueError, 'gpu_uuids'):
            validate_cells([anchor, mixed], 'model', 1)


if __name__ == '__main__':
    unittest.main()
