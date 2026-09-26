import pytest
from bench.core.kv_measurement import TurnKV, attach_turn_measurements


def test_overlap_and_repeated_computation_do_not_inflate_reuse():
    m = TurnKV('p', 1, 37, 35)
    m.compute(0, 16)
    m.compute(0, 16)  # re-preemption repeats actual work
    m.restored = [(16, 35), (16, 35)]
    m.compute(35, 37)
    m.complete = True
    row = m.result()
    assert row['kv_reused_tokens'] == row['cpu_restored_tokens'] == 19
    assert row['gpu_reused_tokens'] == 0
    assert row['prefill_computed_tokens'] == 34
    assert row['recomputed_context_tokens'] == 32


def test_preserved_context_excludes_new_prefill_and_decode():
    m = TurnKV('p', 1, 37, 35)
    m.compute(35, 38)
    m.complete = True
    row = m.result()
    assert row['gpu_reused_tokens'] == row['kv_reused_tokens'] == 35
    assert row['cpu_restored_tokens'] == 0
    assert row['prefill_computed_tokens'] == 2
    records = [dict(program_id='p', turn_idx=1, input_toks=37)]
    attach_turn_measurements(records, [dict(turn_kv_measurements=[row])])
    assert records[0]['kv_reused_tokens'] == 35


def test_missing_engine_measurement_is_an_error():
    with pytest.raises(KeyError):
        attach_turn_measurements([dict(program_id='p', turn_idx=0, input_toks=1)],
                                 [dict(turn_kv_measurements=[])])
