"""What a RECOMPUTE preemption must not change about a request's metrics.

These drive the real Scheduler -- `add_done` and `_preempt_recompute` -- so
that reverting either production fix fails them. An earlier version of this
file replayed the accounting rules in its own helper and would have passed
against the bug it was written for.

They pin an interaction three separately-correct changes passed through:

  * the recompute prompt must include the token sampled just before the
    preemption (vLLM's `request.num_tokens`), or the cache-hit cap lands one
    below a page boundary;
  * because it does, the resumed prefill ends one position further on and
    emits a genuinely NEW output token, so it counts;
  * but it is not the FIRST token, so it also carries an inter-token
    interval -- the one holding the preemption stall.

Guarding the count to fix an earlier double-count made it under-count
instead.
"""

import pytest

from serving.core.request import Batch, Request
from serving.core.scheduler import Scheduler

PROMPT, TOTAL = 16, 49          # 33 output tokens
STALL = 20
MODEL = "meta-llama/Llama-3.1-8B"


def _scheduler():
    # Prefix caching off: these assertions are about token accounting, and the
    # cache paths would drag in profiled sizing this test has no need of.
    return Scheduler(
        model=MODEL, node_id=0, instance_id=0, max_num_seqs=128,
        max_num_batched_tokens=4096, num_npus=1, tp_size=1, pp_size=1,
        npu_mem=80.0, cpu_mem=128.0, start_npu=0, pd_type=None, fp=2,
        block_size=16, req_num=1, prioritize_prefill=False,
        enable_prefix_caching=False, enable_prefix_sharing=False,
        prefix_pool=None, prefix_storage=None, enable_chunked_prefill=True)


def _step(sched, req, chunk, now, batch_id):
    """One batch carrying `req`, driven through the real add_done."""
    req.chunk_len = chunk
    b = Batch(batch_id=batch_id, model=MODEL, total_len=chunk, kv_len=0,
              q_list=[], k_list=[], num_prefill=0, num_decode=0,
              prefill_q_list=[], prefill_k_list=[], decode_k_list=[],
              batch_time=0, kv_size=0)
    b.requests = [req]
    sched.inflight.append(b)
    return sched.add_done(batch_id + 1, sched.start_npu, now)


def _run(n_preemptions=0):
    sched = _scheduler()
    req = Request(id=1, model=MODEL, input=PROMPT, output=TOTAL, arrival=0,
                  instance_id=0)
    gen_total, now, bid = 0, 0, 0
    preempts_left = n_preemptions

    now += 1
    _, g, _ = _step(sched, req, PROMPT, now, bid); bid += 1
    gen_total += g

    while req not in sched.done:
        if preempts_left and req.num_computed_tokens >= req.original_input + 8:
            sched._preempt_recompute(req)
            preempts_left -= 1
            now += STALL
            _, g, _ = _step(sched, req, req.original_input, now, bid); bid += 1
            gen_total += g
            continue
        now += 1
        _, g, _ = _step(sched, req, 0, now, bid); bid += 1
        gen_total += g
    return req, gen_total


@pytest.mark.parametrize("n", [0, 1, 2])
def test_preemption_changes_no_reported_quantity(n):
    req, gen = _run(n)
    assert gen == 33, "output tokens counted"
    assert len(req.itl) == 32, "inter-token intervals"
    assert req.ttft == 1, "first-token time, not the resume"
    assert req.input == PROMPT, "recompute expansion leaked into the report"
    assert req.output - req.input == 33, "implied output length"


@pytest.mark.parametrize("n", [1, 2])
def test_every_stall_appears_once_in_the_inter_token_series(n):
    req, _ = _run(n)
    assert sum(1 for i in req.itl if i >= STALL) == n, (
        "re-basing the inter-token origin on a resume hides the stall")


def test_recompute_prompt_includes_the_sampled_token():
    """vLLM: max_cache_hit_length = request.num_tokens - 1, and num_tokens
    counts the token sampled before the preemption."""
    sched = _scheduler()
    req = Request(id=1, model=MODEL, input=PROMPT, output=TOTAL, arrival=0,
                  instance_id=0)
    req.num_computed_tokens = 32
    sched._preempt_recompute(req)
    assert req.original_input == 33
    assert max(0, req.original_input - 1) // 16 * 16 == 32, (
        "cap must not floor to one page below the cached context")


def test_a_prefill_preemption_adds_no_sampled_token():
    sched = _scheduler()
    req = Request(id=1, model=MODEL, input=PROMPT, output=TOTAL, arrival=0,
                  instance_id=0)
    req.num_computed_tokens = 8          # still prefilling, nothing sampled
    sched._preempt_recompute(req)
    assert req.original_input == PROMPT
