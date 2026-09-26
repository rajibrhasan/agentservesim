from types import SimpleNamespace as NS
from unittest.mock import patch

import pandas as pd

from serving.core import trace_generator as tg


def test_actual_attention_coverage_warns_even_when_metadata_claims_full_grid():
    db = dict(hardware='test', model='test', variant='test',
              meta={'engine_effective': {'max_num_batched_tokens': 16384, 'max_num_seqs': 128}},
              tables={1: {'attention': {'pc_vals': [0, 2048]}}})
    with patch.dict(tg._perf_db_cache, {}, clear=True), patch.object(tg.logger, 'warning') as warning:
        tg.warn_if_runtime_exceeds_profiled(db, 16384, 128, 1)
        assert warning.call_count == 1
        assert 'actual attention' in warning.call_args.args[0]
        tg.warn_if_runtime_exceeds_profiled(db, 16384, 128, 1)
        assert warning.call_count == 1


def test_signed_residual_applied_once_and_off_disables_it(monkeypatch):
    table = tg._build_step_overhead_table(pd.DataFrame([
        dict(prefill_chunk=0, n_decode=4, overhead_us=-2),
        dict(prefill_chunk=0, n_decode=8, overhead_us=3)]))
    ctx = NS(perf_db={'tables': {1: {'step_overhead': table}}}, tp_size=1, config={})
    batch = NS(step_adjust=None, prefill_chunk=0, n_decode=4, kv_prefill=0,
               kv_decode_mean=16, kv_decode_max=16, kv_decode_min=16, lm_head_len=4)
    monkeypatch.setenv('STEP_OVERHEAD_MODE', 'auto')
    with patch.object(tg, 'step_kernel_ns', return_value=10000) as kernel:
        assert tg._step_adjustment(ctx, batch) == (0.8, 0)
        assert tg._step_adjustment(ctx, batch) == (0.8, 0)
        assert kernel.call_count == 1
    batch.step_adjust = None
    batch.n_decode = 8
    assert tg._step_adjustment(ctx, batch) == (1, 3000)
    batch.step_adjust = None
    monkeypatch.setenv('STEP_OVERHEAD_MODE', 'off')
    assert tg._step_adjustment(ctx, batch) == (1, None)
