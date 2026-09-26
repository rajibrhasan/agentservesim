"""Standalone unit test for the DAG dependency logic in router.py.

Exercises the Router's DAG / chain / flat handling directly (no ASTRA-Sim
backend, no schedulers) to verify:
  1. fan-out: all root (in-degree 0) nodes are queued at arrival
  2. fan-in / barrier: a join node releases only when its LAST parent finishes,
     at max(parent_end + edge_delay)  <-- the barrier wait
  3. workflow state is cleaned up after all nodes complete
  4. backward compat: linear chain (sub_requests) and flat requests still work
"""
import os, sys, types

# Import the Router from the repo (tests/ -> repo root on sys.path).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from serving.core.router import Router


def make_router():
    dummy = types.SimpleNamespace(pd_type="prefill", enable_prefix_caching=False,
                                  instance_id=0, model="m")
    return Router(num_instances=1, schedulers=[dummy], req_num=0, routing_policy="RR")


def pending_index_to_arrival(r):
    return {p['index']: p['arrival_time_ns'] for p in r._pending_requests}


def test_fanout_fanin_barrier():
    r = make_router()
    debate = {
        "workflow_id": "wf0", "arrival_time_ns": 0,
        "nodes": [
            {"node_id": "d0", "input_toks": 512, "output_toks": 128},
            {"node_id": "d1", "input_toks": 512, "output_toks": 128},
            {"node_id": "d2", "input_toks": 512, "output_toks": 128},
            {"node_id": "judge", "input_toks": 600, "output_toks": 200},
        ],
        "edges": [
            {"src": "d0", "dst": "judge", "delay_ns": 0},
            {"src": "d1", "dst": "judge", "delay_ns": 0},
            {"src": "d2", "dst": "judge", "delay_ns": 0},
        ],
    }
    r._load_dag_workflow(debate, enable_prefix_caching=False)

    # node indices: d0=0, d1=1, d2=2, judge=3
    released = {p['node_id'] for p in r._pending_requests}
    assert released == {"d0", "d1", "d2"}, f"roots should be queued, got {released}"
    assert "judge" not in released, "judge must NOT be released before parents finish"
    assert r.has_deferred_sessions(), "judge is unreleased -> deferred work remains"

    # Complete debaters out of order; judge must wait for the SLOWEST (300).
    r.notify_request_completed(0, 100)   # d0 done
    assert "judge" not in {p['node_id'] for p in r._pending_requests}, "1 parent done"
    r.notify_request_completed(2, 200)   # d2 done
    assert "judge" not in {p['node_id'] for p in r._pending_requests}, "2 parents done"
    r.notify_request_completed(1, 300)   # d1 done (slowest) -> releases judge

    arr = pending_index_to_arrival(r)
    assert 3 in arr, "judge (index 3) should now be released"
    assert arr[3] == 300, f"barrier wait: judge starts at max parent end (300), got {arr[3]}"
    # Once released, the judge is in the pending queue (has_pending_requests guards
    # exit); has_deferred_sessions only tracks UNRELEASED work, so it is now False.
    assert r.has_pending_requests(), "judge is queued as a pending request"
    assert "wf0" in r._workflows, "workflow alive until all nodes complete"
    assert not r.has_deferred_sessions(), "no unreleased work remains once judge is out"

    # Complete judge -> workflow fully done, state cleaned up.
    r.notify_request_completed(3, 500)
    assert "wf0" not in r._workflows, "workflow state should be freed after all nodes done"
    assert not r.has_deferred_sessions(), "no deferred work remains"
    print("PASS test_fanout_fanin_barrier (barrier wait = 300 = slowest debater)")


def test_edge_delay_in_barrier():
    """Edge delay (inter-agent message/think time) adds to the barrier."""
    r = make_router()
    wf = {
        "workflow_id": "wf1", "arrival_time_ns": 0,
        "nodes": [
            {"node_id": "a", "input_toks": 10, "output_toks": 10},
            {"node_id": "b", "input_toks": 10, "output_toks": 10},
            {"node_id": "join", "input_toks": 10, "output_toks": 10},
        ],
        "edges": [
            {"src": "a", "dst": "join", "delay_ns": 50},
            {"src": "b", "dst": "join", "delay_ns": 1000},
        ],
    }
    r._load_dag_workflow(wf, enable_prefix_caching=False)
    r.notify_request_completed(0, 400)   # a done at 400 -> ready via a: 450
    r.notify_request_completed(1, 200)   # b done at 200 -> ready via b: 1200 (slower edge)
    arr = pending_index_to_arrival(r)
    assert arr[2] == 1200, f"barrier = max(400+50, 200+1000) = 1200, got {arr[2]}"
    print("PASS test_edge_delay_in_barrier (barrier = 1200 via slow edge)")


def test_chain_backward_compat():
    r = make_router()
    session = {
        "session_id": "s0", "arrival_time_ns": 0,
        "sub_requests": [
            {"input_toks": 10, "output_toks": 10, "tool_duration_ns": 100},
            {"input_toks": 20, "output_toks": 20, "tool_duration_ns": 0},
        ],
    }
    r._load_agentic_session(session, enable_prefix_caching=False)
    assert len(r._pending_requests) == 1, "only first sub-request queued"
    r.notify_request_completed(0, 500)   # release sub_request[1] at 500+100
    arr = pending_index_to_arrival(r)
    assert arr[1] == 600, f"chain release = completion + tool_duration = 600, got {arr[1]}"
    print("PASS test_chain_backward_compat (chain still works)")


def test_flat_noop():
    r = make_router()
    r._load_flat_request({"input_toks": 10, "output_toks": 10, "arrival_time_ns": 0},
                         enable_prefix_caching=False)
    r.notify_request_completed(0, 100)   # flat -> no-op, must not raise
    assert not r.has_deferred_sessions()
    print("PASS test_flat_noop (flat request no-op)")


def test_workflow_metrics():
    """Per-workflow JCT is recorded on completion = last node end - arrival."""
    r = make_router()
    wf = {
        "workflow_id": "wf0", "arrival_time_ns": 1000,
        "nodes": [
            {"node_id": "d0", "input_toks": 10, "output_toks": 10},
            {"node_id": "d1", "input_toks": 10, "output_toks": 10},
            {"node_id": "judge", "input_toks": 10, "output_toks": 10},
        ],
        "edges": [
            {"src": "d0", "dst": "judge", "delay_ns": 0},
            {"src": "d1", "dst": "judge", "delay_ns": 0},
        ],
    }
    r._load_dag_workflow(wf, enable_prefix_caching=False)
    assert not r.has_workflow_metrics(), "no metric until workflow completes"
    r.notify_request_completed(0, 1500)   # d0
    r.notify_request_completed(1, 1800)   # d1 (slowest) -> judge releases at 1800
    r.notify_request_completed(2, 4000)   # judge (last node) -> workflow done
    assert r.has_workflow_metrics()
    s = r.workflow_metrics_summary()
    assert s["num_workflows"] == 1
    # JCT = last node end (4000) - workflow arrival (1000) = 3000
    assert s["jct_mean_ns"] == 3000, f"expected JCT 3000, got {s['jct_mean_ns']}"
    assert s["jct_p99_ns"] == 3000
    print("PASS test_workflow_metrics (JCT = last node end - arrival = 3000)")


if __name__ == "__main__":
    test_fanout_fanin_barrier()
    test_edge_delay_in_barrier()
    test_chain_backward_compat()
    test_flat_noop()
    test_workflow_metrics()
    print("\nAll router DAG tests passed.")
