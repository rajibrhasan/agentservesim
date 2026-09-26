import queue
import threading
from types import SimpleNamespace

import pytest

from bench.core.engine_control_wait import EngineControlWait


def test_protect_runs_on_owner_before_model_finishes():
    owner = threading.get_ident()
    done = threading.Event()
    observed = []
    q = queue.Queue()
    q.put(('utility', (0, 1, 'kv_protect', ['tag', 2.0])))

    def dispatch(*request):
        observed.append(threading.get_ident())
        done.set()

    def result():
        assert threading.get_ident() != owner
        assert done.wait(2), 'RPC blocked behind model completion'
        return 42

    waiter = EngineControlWait()
    try:
        assert waiter.result(SimpleNamespace(result=result),
                             SimpleNamespace(input_queue=q,
                                             _handle_client_request=dispatch),
                             'utility') == 42
        assert observed == [owner]
        assert waiter.serviced == 1
    finally:
        waiter.close()


@pytest.mark.parametrize('head', [('add', None), ('abort', None),
                                 ('utility', (0, 1, 'kv_evict', ['tag'])),
                                 ('utility', (0, 1, 'kv_release', ['tag']))])
def test_unsafe_head_and_following_protect_preserve_order(head):
    q = queue.Queue()
    q.put(head)
    q.put(('utility', (0, 2, 'kv_protect', ['tag', 2.0])))
    def dispatch(*args):
        pytest.fail('Unsafe message or reordered protection dispatched')
    def result():
        threading.Event().wait(0.02)
        return 7
    waiter = EngineControlWait()
    try:
        assert waiter.result(SimpleNamespace(result=result),
                             SimpleNamespace(input_queue=q,
                                             _handle_client_request=dispatch),
                             'utility') == 7
        assert q.get_nowait() == head
        assert q.qsize() == 1
    finally:
        waiter.close()


def test_model_failure_propagates():
    def fail():
        raise ValueError('worker failure')
    waiter = EngineControlWait()
    try:
        with pytest.raises(ValueError, match='worker failure'):
            waiter.result(SimpleNamespace(result=fail),
                          SimpleNamespace(input_queue=queue.Queue()), 'utility')
    finally:
        waiter.close()
