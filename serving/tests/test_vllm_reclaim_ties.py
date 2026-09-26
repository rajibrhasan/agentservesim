"""Execute the real port's method without importing CUDA/vLLM dependencies.

Set VLLM_RECLAIM_AUDIT_SOURCE to the base port's block_pool.py. This verifies
the installed method and the minimal source repair, rather than a rewrite of
its algorithm inside the test.
"""
import ast
from collections import defaultdict
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.validation.fix_kv_reclaim_ties import corrected_source


@pytest.fixture
def source():
    path = os.environ.get('VLLM_RECLAIM_AUDIT_SOURCE')
    if not path:
        pytest.skip('set VLLM_RECLAIM_AUDIT_SOURCE for the base-port audit')
    return Path(path).read_text()


def reclaim(source, entries, count):
    cls = next(n for n in ast.parse(source).body
               if isinstance(n, ast.ClassDef) and n.name == 'BlockPool')
    method = next(n for n in cls.body
                  if isinstance(n, ast.FunctionDef) and n.name == '_reclaim_protected')
    namespace = {'time': SimpleNamespace(time=lambda: 10.0), 'KVCacheBlock': object}
    exec(compile(ast.Module(body=[method], type_ignores=[]), '<real-reclaim>', 'exec'), namespace)
    pool = SimpleNamespace(_protected=dict(entries), blocks=list(range(32)),
                           protection_stats=defaultdict(int))
    chosen = namespace['_reclaim_protected'](pool, count)
    return chosen, pool


def test_base_port_counterexample_and_repair(source):
    # Physical block 0 is the prompt head; free() parks 3,2,1,0.
    entries = [(3, 20.0), (2, 20.0), (1, 20.0), (0, 20.0)]
    before, _ = reclaim(source, entries, 1)
    after, pool = reclaim(corrected_source(source), entries, 1)
    assert before == [0]  # First-page loss makes the entire prefix unmatchable.
    assert after == [3]
    assert set(pool._protected) == {0, 1, 2}
    assert pool.protection_stats['reclaimed_forced'] == 1


def test_latest_deadline_still_precedes_earlier_deadline(source):
    chosen, _ = reclaim(corrected_source(source), [(3, 20), (2, 20), (1, 30), (0, 30)], 3)
    assert chosen == [1, 0, 3]


def test_expired_blocks_still_precede_forced_reclaims(source):
    chosen, pool = reclaim(corrected_source(source), [(3, 20), (2, 20), (1, 5), (0, 5)], 3)
    assert chosen == [1, 0, 3]
    assert pool.protection_stats['reclaimed_expired'] == 2
    assert pool.protection_stats['reclaimed_forced'] == 1


def test_equal_deadline_contexts_preserve_parking_order(source):
    chosen, _ = reclaim(corrected_source(source), [(3, 20), (2, 20), (7, 20), (6, 20)], 3)
    assert chosen == [3, 2, 7]


def test_patch_is_idempotent_and_rejects_unknown_code(source):
    fixed = corrected_source(source)
    assert corrected_source(fixed) == fixed
    with pytest.raises(ValueError):
        corrected_source('unknown port')
