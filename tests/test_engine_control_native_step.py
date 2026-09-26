"""Exercise installed native step/FutureWrapper without starting GPUs."""
import ast
import __future__
import os
import queue
import threading
from collections import deque
from concurrent.futures import Future, InvalidStateError
from contextlib import nullcontext, suppress
from pathlib import Path
from types import SimpleNamespace as NS

import pytest


def extract(path, name, namespace):
    tree = ast.parse(path.read_text())
    node = next(n for n in ast.walk(tree)
                if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec',
                 flags=__future__.annotations.compiler_flag), namespace)
    return namespace[name]


@pytest.mark.parametrize('enabled', [False, True])
def test_native_lazy_execution_and_sampling(monkeypatch, enabled):
    root = os.environ.get('TEST_VLLM_ROOT')
    if not root:
        pytest.skip('Set TEST_VLLM_ROOT to the patched checkout')
    root = Path(root)
    wrapper = extract(root/'vllm/v1/executor/multiproc_executor.py',
                      'FutureWrapper', dict(Future=Future, suppress=suppress,
                                            InvalidStateError=InvalidStateError))
    step = extract(root/'vllm/v1/engine/core.py', 'step',
                   dict(os=os, EngineCoreRequestType=NS(UTILITY='utility')))
    monkeypatch.setenv('VLLM_SERVICE_RETENTION_DURING_EXECUTION', str(int(enabled)))
    q = queue.Queue()
    owner = threading.get_ident()
    phases = []
    release = threading.Event()
    def dispatch(*args):
        assert threading.get_ident() == owner
        phases.append('rpc')
        release.set()
    def lazy(value):
        release.clear()
        if enabled:
            q.put(('utility', (0, 1, 'kv_protect', ['tag', 2.])))
        def respond():
            if enabled:
                assert release.wait(2), 'Owner did not service RPC during execution'
            return value
        pending = deque()
        result = wrapper(pending)
        pending.appendleft((result, respond))
        return result
    def sample(grammar, non_block=False):
        phases.append('sample')
        return lazy('output') if non_block else 'output'
    scheduler = NS(has_requests=lambda: True,
                   schedule=lambda: NS(total_num_scheduled_tokens=1),
                   get_grammar_bitmask=lambda _: None,
                   update_from_output=lambda _, result: result)
    engine = NS(scheduler=scheduler, _agent_policy=None,
                model_executor=NS(execute_model=lambda *a, **kw: lazy(None),
                                  sample_tokens=sample),
                log_error_detail=lambda _: nullcontext(),
                log_iteration_details=lambda _: nullcontext(),
                _process_aborts_queue=lambda: None,
                input_queue=q, _handle_client_request=dispatch)
    try:
        assert step(engine) == ('output', True)
        assert phases == (['rpc', 'sample', 'rpc'] if enabled else ['sample'])
    finally:
        if hasattr(engine, '_policy_control_wait'):
            engine._policy_control_wait.close()
