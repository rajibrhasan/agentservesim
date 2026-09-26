"""Program-aware batch scheduling.

The old scheduler is request-scoped: it forms batches under a token budget and
knows nothing about the program a turn belongs to, so program-level ordering had
to be bolted on by stamping priorities from outside. This plane asks the
orchestrator instead, and keeps no program state of its own.

It owns exactly two things: which requests are waiting, and which are running.
Everything else it reads -- program state from `program_orchestrator`, memory
from `program_kv`.

Engine types, not new ones
-------------------------
This plane schedules `Request` objects and emits the engine's `Batch`. It does
NOT define its own turn or batch type, for the same reason the KV plane does not
keep its own block records: `Request` already carries everything a turn needs
(`num_computed_tokens`, `input_hash_ids`, `priority`, `queuing_delay`,
`first_sched_ts`, `npu_cache_hit`, `n_preempted`) and everything downstream --
`generate_trace`, the output writer, the router -- speaks these types. A
parallel type would need translating at every boundary and would drift from the
originals at the first field anyone added.

Program identity rides on the request the way the engine already attaches it:
`session_id` / `sub_request_index` for agentic sessions, `workflow_id` /
`node_id` for DAG workflows. All four are initialised in `Request.__init__`
(to None), so they are read directly -- AGENTS.md forbids `getattr` fallbacks
on Request attributes, and a fallback here would hide the case where identity
was never attached instead of surfacing it.

Three decisions are the policy's, and they are the three the contract names:

    priority(program, now)         rank a waiting turn
    admit(program, pressure, now)  run it now, or hold it back
    victim(candidates, now)        who gets preempted when KV runs out

They arrive as plain callables over engine state, not as contract objects, so
the engine stays independent of the policy contract and the adapter translates.
A policy that declines -- returns None -- gets the engine's default, every time.

Preemption is RECOMPUTE, as in vLLM v1: the victim's KV is released and its
prefill progress discarded, so it re-prefills when it next runs. That is the
expensive path, and counting it is how this plane is checked against real vLLM.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .program_kv import ProgramKVManager
from .program_orchestrator import ProgramOrchestrator, ProgramState
from .request import Batch, Request


#: Two disjoint id spaces for tokens the trace does not name, each a million
#: wide per request. Real token ids are a tokenizer's, far below either.
_PRIVATE_PROMPT = 0
_PRIVATE_DECODE = 1_000_000_000_000


@dataclass(frozen=True)
class QueueSnapshot:
    """Aggregate engine state at one scheduling tick, plus the waiting turn's
    own two token counts. Nothing here identifies another program: an
    admission gate that could see its neighbours would be reading state the
    real gateway has no access to, and a decision it could not reproduce there
    is not a decision the benchmark can score."""

    n_running: int
    n_waiting: int
    n_inflight: int
    kv_utilization: float
    kv_free_tokens: int
    kv_evictable_tokens: int
    prompt_tokens: int
    cached_tokens: int


@dataclass(frozen=True)
class RunningView:
    """One running request as the victim rule sees it."""

    request_id: str
    state: object                 # ProgramState; typed loosely to stay local
    priority: int
    prompt_tokens: int
    computed_tokens: int
    generated_tokens: int
    is_prefill: bool


def _ns(v):
    """A whole nanosecond count, written as an integer.

    This plane carries the clock as a float because the main loop does; the old
    plane carries it as an int. The values are the same whole nanoseconds, but
    `17001004787.0` and `17001004787` are not the same nine bytes, and anything
    downstream parsing the column with `int()` sees only one of them.
    """
    f = float(v)
    return int(f) if f.is_integer() else f


def program_of(req: Request) -> Optional[str]:
    """The program a request belongs to, however the engine tagged it.

    Agentic sessions carry `session_id`; DAG workflows carry `workflow_id`.
    A flat request belongs to no program and is scheduled, never ranked -- its
    policy hooks would have nothing to read.
    """
    return req.session_id or req.workflow_id


def turn_of(req: Request) -> int:
    idx = req.sub_request_index
    if idx is None:
        idx = req.node_id
    return int(idx) if idx is not None else 0



def _insort_by_arrival(queue: List[Request], req: Request) -> None:
    """Insert keeping (arrival, id) order.

    Hand-rolled rather than `bisect.insort(..., key=...)`: that keyword needs
    Python 3.10, which the simulator container has and the test venv does not.
    The old scheduler uses the keyword form, which is why its queue ordering has
    never been unit-tested off the container.
    """
    k = (req.arrival, req.id)
    lo, hi = 0, len(queue)
    while lo < hi:
        mid = (lo + hi) // 2
        if (queue[mid].arrival, queue[mid].id) <= k:
            lo = mid + 1
        else:
            hi = mid
    queue.insert(lo, req)

class ProgramBatchScheduler:
    """The scheduling plane for ONE instance."""

    def __init__(self, instance: int, orchestrator: ProgramOrchestrator,
                 kv: ProgramKVManager, model: str = "",
                 max_num_batched_tokens: int = 16384,
                 max_num_seqs: int = 128,
                 long_prefill_token_threshold: Optional[int] = None,
                 start_npu: int = 0, num_npus: int = 1,
                 pd_type: str = 'both', pp_size: int = 1,
                 enable_prefix_caching: bool = True,
                 priority_fn: Optional[Callable] = None,
                 admit_fn: Optional[Callable] = None,
                 victim_fn: Optional[Callable] = None) -> None:
        self.instance = instance
        self.orch = orchestrator
        self.kv = kv
        self.model = model
        self.max_model_len = None
        self.max_num_batched_tokens = max_num_batched_tokens
        self.max_num_seqs = max_num_seqs
        self.long_prefill_token_threshold = long_prefill_token_threshold
        self.start_npu = start_npu
        self.num_npus = num_npus
        self.enable_prefix_caching = enable_prefix_caching
        self.priority_fn = priority_fn
        self.admit_fn = admit_fn
        self.victim_fn = victim_fn
        #: Retention runs on EVENTS, not on a per-step hook, so it is attached
        #: rather than passed as a function: one object that sees turn
        #: completion, turn arrival and turn admission in order. Set by
        #: `ProgramPolicyAdapter`; None means the engine's own LRU, which is
        #: what every unpinned run should get.
        self.policy = None

        self.waiting: List[Request] = []
        self.done: List[Request] = []

        # Names the router and main loop read off a scheduler. They are part of
        # the scheduler protocol rather than of this plane's design, so they sit
        # together here instead of being scattered where each is first needed.
        self.instance_id = instance          # main and the router use this name
        self.pd_type = pd_type               # prefill / decode / both
        self.pp_size = pp_size               # max batches in flight
        self.cleanup_et = False              # set by main from --cleanup-inputs
        self.policy_hooks = None             # old adapter; unused by these planes
        self.scheduling_policy = None        # ordering lives in priority_fn here
        self.num_preemptions = 0             # mirrors counters['preemptions']
        self.preemption_log: List[tuple] = []
        self._none_reason = None             # why schedule() returned nothing
        self.running: List[Request] = []
        self.inflight: List[Batch] = []
        # Seeded at -1 so the first batch is numbered 0, matching the engine
        # (scheduler.py:72). ASTRA-Sim reports a batch as id+1 and `add_done`
        # decrements before looking it up, so starting at 0 makes every lookup
        # miss by one -- the first batch then never completes, the pipeline
        # gate blocks every later one, and the run spins forever while the
        # clock advances.
        self._batch_id = -1
        self.step_no = 0
        #: Prefix-cache hit accounting, in the shape `return_prefix_info`
        #: reports. Kept here rather than in the KV plane because the hit is a
        #: property of a REQUEST -- which prompt, how much of it was already
        #: resident -- and the KV plane sees only token ids.
        self.prefix_requested_tokens = 0
        self.prefix_hit_tokens = 0
        #: Monotone admission stamp; preemption picks the highest.
        self._admit_counter = 0
        self.counters: Dict[str, int] = {
            "priority_stamps": 0, "admission_holds": 0,
            "victim_overrides": 0, "preemptions": 0,
        }
        #: What the last step decided. Kept here rather than on the Batch, which
        #: is a kernel description and has no room for decisions.
        self.last_admitted: List[Request] = []
        self.last_held: List[Request] = []
        self.last_preempted: List[Request] = []

    def get_batch_id(self) -> int:
        self._batch_id += 1
        return self._batch_id

    def is_request_empty(self) -> bool:
        return not self.waiting and not self.running

    # ------------------------------------------------------------- arrival

    def add_request(self, req, is_init=True, workflow_id=None, node_id=None,
                    priority=None, session_id=None, sub_request_index=None):
        """A request enters the system. Queued, not running.

        Signature matches the old scheduler because the router calls it: `req`
        is a LIST of Request fields, not a Request, and the scheduler is what
        constructs one. Program identity arrives as keywords and is attached
        the same way here.

        The queue is kept sorted on insert rather than sorted on read: turns
        released mid-run by a tool gap arrive out of arrival order, and
        appending would let a late arrival sit ahead of an earlier one whenever
        the policy declines to rank.
        """
        new_req = Request(*(req), is_init=is_init)
        if self.max_model_len is not None and new_req.output > self.max_model_len:
            raise ValueError(f'Request {new_req.id} context {new_req.output} exceeds '
                             f'max_model_len={self.max_model_len}')
        new_req.workflow_id = workflow_id
        new_req.node_id = node_id
        new_req.session_id = session_id
        new_req.sub_request_index = sub_request_index
        if priority is not None:
            new_req.priority = priority

        pid = program_of(new_req)
        if pid is not None and self.policy is not None:
            # BEFORE `on_turn_arrival` clears the gap record: that call is what
            # folds the gap's duration away, and this is the only moment a
            # policy can still see it. Ordering, not politeness -- a policy
            # that learns tool times learns nothing if it runs second.
            prior = self.orch.get(pid)
            if prior is not None:
                self.policy.observe_arrival(prior, new_req.arrival)
        if pid is not None:
            # A turn whose whole context cannot fit the pool even when the pool
            # is empty can never run: no reclaim frees enough, and there is
            # nobody to preempt but itself. Left alone it presents as a hang --
            # the scheduler returns no batch, forever -- so it is refused here,
            # at the earliest moment the numbers are known, the way vLLM refuses
            # a request needing more blocks than the cache has.
            if new_req.output > self.kv.capacity_tokens:
                raise ValueError(
                    f"turn {new_req.id} needs {new_req.output:,} tokens of KV "
                    f"(prompt {new_req.input:,} + generated "
                    f"{new_req.output - new_req.input:,}) but the pool holds "
                    f"{self.kv.capacity_tokens:,}. No reclaim or preemption can "
                    f"make this fit. Raise --kv-pool-tokens above "
                    f"{new_req.output:,}, or run a workload with shorter turns.")
            self.orch.on_turn_arrival(pid, turn_of(new_req), new_req.arrival)
            self.orch.place(pid, self.instance)
            if self.policy is not None:
                self.policy.on_turn_arrival(
                    self.kv, self.orch.get(pid), new_req.arrival)

        _insort_by_arrival(self.waiting, new_req)
        return new_req

    def add_decode(self, req):
        """A prefill instance handing a request to this decode instance."""
        req.instance_id = self.instance
        _insort_by_arrival(self.waiting, req)

    @property
    def request(self):
        """The waiting queue under the name the main loop reads."""
        return self.waiting

    def save_output(self, output_file, is_append=False):
        """Per-request rows, byte-compatible with the old scheduler's.

        The file is the artifact -- the arena and the paper read it -- so the
        columns mean what they have always meant, not what is convenient here:

        * `output` is GENERATED tokens, `req.output - req.input`. The field is
          a cumulative target internally (prompt + generated); writing it raw
          reported 209 where the old plane reported 20.
        * the path is resolved the same way, because the simulator has chdir'd
          into `astra-sim/` and a relative `--output` that is not prefixed goes
          silently into the wrong tree.

        `request id` is `program:turn` rather than the old plane's global
        counter, which is the one deliberate difference: that counter belonged
        to the old router, and the identity it stood for is carried by the
        `program_id` and `turn_idx` columns on both paths.
        """
        import csv
        if not os.path.isabs(output_file):
            output_file = f"../{output_file}"
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        mode = "a" if is_append else "w"
        with open(output_file, mode, newline="") as f:
            w = csv.writer(f)
            if not is_append:
                w.writerow(["instance id", "request id", "model", "input",
                            "output", "arrival", "end_time", "latency",
                            "queuing_delay", "TTFT", "TPOT", "ITL",
                            "cache_hit", "n_preempted", "program_id",
                            "turn_idx", "first_sched_ts", "first_cache_hit"])
            for r in self.done:
                w.writerow([self.instance, r.id, r.model, r.input,
                            r.output - r.input,
                            _ns(r.arrival), _ns(r.end_time), _ns(r.latency),
                            _ns(r.queuing_delay), _ns(r.ttft), _ns(r.tpot),
                            [_ns(v) for v in r.itl],
                            r.npu_cache_hit, r.n_preempted,
                            program_of(r), turn_of(r),
                            _ns(r.first_sched_ts), r.first_cache_hit])

    def print_result(self):
        n = len(self.done)
        print(f"[instance {self.instance}] program-aware planes: "
              f"{n} requests done, {self.counters['preemptions']} preemptions, "
              f"{self.counters['admission_holds']} admission holds")

    # ------------------------------------------------------------ ordering

    def _state(self, req: Request) -> Optional[ProgramState]:
        pid = program_of(req)
        return self.orch.get(pid) if pid else None

    def _rank(self, now: float) -> List[Request]:
        """Waiting requests in the order the policy wants them.

        A policy returning None leaves a request at its arrival order, so a
        partial ranking is legal and the default is FCFS. Stamps are counted
        because a scheduling policy that never stamps is inert, and an inert
        policy still produces a perfectly plausible JCT.
        """
        arrived = [r for r in self.waiting if r.arrival <= now]
        if self.priority_fn is None:
            return sorted(arrived, key=lambda r: (r.arrival, r.id))

        # Which requests the policy ranked THIS call, tracked rather than
        # inferred: `Request.priority` defaults to 0, so a sentinel check
        # ("is it None?") would read every unstamped request as top priority.
        # It is also the more honest question -- a request may carry a stamp
        # from the workload or from an earlier round, and neither means the
        # policy has an opinion about it now.
        stamped: Dict[str, int] = {}
        for req in arrived:
            state = self._state(req)
            if state is None:
                continue
            stamp = self.priority_fn(state, now)
            if stamp is not None:
                req.priority = int(stamp)
                stamped[req.id] = int(stamp)
                self.counters["priority_stamps"] += 1

        # Ranked requests first, in rank order; the rest keep arrival order
        # among themselves. "No opinion" must not mean "go last arbitrarily",
        # or one ranked program silently reorders every unranked one.
        return sorted(
            arrived,
            key=lambda r: ((0, stamped[r.id], r.arrival, r.id)
                           if r.id in stamped else (1, 0, r.arrival, r.id)))

    # ------------------------------------------------------------- the step

    def schedule(self, now: float, sys=None, batch_id=-1) -> Optional[Batch]:
        """Form one batch, or None when there is nothing to run.

        `sys` and `id` are accepted and ignored: the main loop passes the
        ASTRA-Sim system and batch ids positionally, and this plane does not
        use them -- it does not multiplex batches across NPU ids the way the
        old scheduler does.

        Running requests are served before waiting ones, as vLLM v1 does: a
        request that has already paid its prefill is cheaper to continue than a
        new one is to start, and reversing that changes throughput without
        changing any policy.
        """
        # A batch is formed ONLY on the instance's start NPU. Every other NPU
        # of the instance polls for the batch that already exists and marks
        # itself as having fired it. Forming a new batch on each poll leaves
        # batches in `inflight` that nothing will ever complete, and `inflight`
        # is what the main loop reads to decide the system is idle -- so the
        # loop stops fast-forwarding through tool gaps and crawls forward a
        # millisecond per step. That cost 112,000 steps for 14 requests.
        if sys is not None and sys != self.start_npu:
            if not self.inflight:
                return None
            batch = next((b for b in self.inflight
                          if b.batch_id == batch_id), None)
            if batch is None or sys in batch.fired:
                return None
            batch.fired.append(sys)
            return batch

        # Pipeline depth: at most `pp_size` batches in flight at once. Without
        # this the scheduler keeps forming batches the backend has not finished.
        if len(self.inflight) >= self.pp_size:
            self._none_reason = f"t={now} site=pp_inflight inflight={len(self.inflight)}"
            return None

        # Nothing has arrived yet: do not form an empty batch, so the main loop
        # sees an idle system and jumps the clock to the next arrival.
        if self.waiting and min(r.arrival for r in self.waiting) > now and not self.running:
            self._none_reason = f"t={now} site=future_arrival"
            return None

        self.step_no += 1
        self.last_admitted, self.last_held, self.last_preempted = [], [], []
        budget = self.max_num_batched_tokens
        scheduled: List[Tuple[Request, int]] = []

        # 1. running requests continue. A running request that cannot get
        #    memory is the ONLY thing that justifies preemption: it is already
        #    admitted and cannot proceed, so something must give.
        #
        #    Decodes before prefills, which is vLLM v1's order under chunked
        #    prefill and the old plane's (scheduler.py: `batch_req = decodes +
        #    prefills`, then budget to decodes first). A decode costs one token;
        #    letting a mid-chunk prefill take the budget ahead of it drops
        #    decodes out of the batch, and the batch's shape is exactly what
        #    indexes the profiled latency table. It changes no token count,
        #    which is why it showed up only as a few hundred microseconds of
        #    TTFT on requests whose prompt, hit and schedule time were all
        #    identical.
        ordered = ([r for r in self.running if not r.is_prefill()]
                   + [r for r in self.running if r.is_prefill()])
        for req in ordered:
            if budget <= 0:
                break
            if req not in self.running:
                # Preempted earlier in THIS loop, by a request that came before
                # it. The list is a snapshot, so without this check the victim
                # keeps its turn and preempts the request that just preempted
                # it -- they swap the memory back and forth, both re-lock, and
                # neither ever finishes. Two requests ran 6,000 steps to zero
                # completions that way.
                continue
            if req.is_prefill():
                self._prefix_match(req)
                take = self._chunk(req, budget)
            else:
                take = 1
            if take == 0:
                continue
            if not self._acquire(req, take, now, may_preempt=True):
                # STOP the step. `_acquire` fails here only after preempting
                # every victim it could find and still not fitting -- including
                # when the only candidate left is this request itself. Nothing
                # the next running request asks for can succeed where that
                # failed. vLLM v1 sets `can_schedule = False` and breaks out of
                # the running loop; the old plane breaks its fit loop on
                # `total_useable_size < kv_size` (scheduler.py:581).
                #
                # Measured as behaviour-neutral on a 14-request tight-pool
                # driver (52 preemptions either way): `_acquire` nearly always
                # succeeds after taking one victim, so this path is rare. Kept
                # because it is what both references do, NOT as a fix for
                # anything -- it is not the source of the 968-vs-590 gap.
                break
            req.chunk_len = take
            scheduled.append((req, take))
            budget -= take

        # 2. admit from the waiting set, in policy order. A request that cannot
        #    get memory simply waits: preempting running work to start new work
        #    is churn, and it is how a scheduler manufactures preemptions the
        #    engine it models never performs.
        for req in self._rank(now):
            if budget <= 0 or len(self.running) >= self.max_num_seqs:
                break
            if req in self.last_preempted:
                # Preempted THIS step. Re-admitting it now hands back the very
                # memory it was preempted to provide, and the request that did
                # the preempting is no better off -- both then make no progress
                # while one of them is re-prefilled from scratch every step.
                # vLLM does not reconsider a request it preempted in the same
                # pass either.
                continue
            if not self._admit(req, now):
                self.last_held.append(req)
                continue
            if req.is_prefill():
                self._prefix_match(req)
                take = self._chunk(req, budget)
            else:
                # A decode can reach the waiting queue legitimately -- the
                # prefill/decode transfer puts one there with `add_decode`.
                # `_chunk` returns 0 for anything past prefill, so sizing an
                # admission with it alone made every such request permanently
                # unschedulable.
                take = 1
            if take == 0:
                continue
            if not self._acquire(req, take, now, may_preempt=False):
                break                  # pool is full; later requests will not fit
            self.waiting.remove(req)
            # Admission ORDER, which is what preemption is chosen by. Not the
            # same as arrival: a request preempted and re-admitted carries its
            # original arrival but a fresh admit_seq, and vLLM preempts the
            # freshest because it is the cheapest to undo.
            self._admit_counter += 1
            req.admit_seq = self._admit_counter
            self.running.append(req)
            pid = program_of(req)
            if pid is not None:
                self.orch.on_turn_scheduled(pid, now)
                if self.policy is not None:
                    self.policy.on_turn_scheduled(
                        self.kv, self.orch.get(pid), now)
            self.last_admitted.append(req)
            req.chunk_len = take
            scheduled.append((req, take))
            budget -= take

        # Footprints exist for policies to read. Computing them costs a tree
        # walk per resident program, so with no policy attached it is pure
        # waste -- and it is per step, which is where an order of magnitude of
        # runtime went on the first end-to-end attempt.
        if self.priority_fn or self.admit_fn or self.victim_fn:
            self.kv.sync_footprints()
        # Drop anyone preempted AFTER being scheduled in this same pass. A
        # request appended to `scheduled` can still be chosen as a victim by a
        # later request's `_acquire`; `_preempt` takes it out of `self.running`
        # and releases its hold, but `scheduled` is a local list it cannot
        # reach. Building the batch from it anyway put a preempted request back
        # into a batch -- the same ghost the in-flight fix removed, one level
        # earlier -- and `add_done` then committed it and unlocked a hold that
        # was already gone.
        scheduled = [(r, n) for r, n in scheduled if r in self.running]
        if not scheduled:
            return None
        return self._build_batch(scheduled, now)

    # -------------------------------------------------------------- batch

    def _build_batch(self, scheduled: Sequence[Tuple[Request, int]],
                     now: float) -> Batch:
        """Turn scheduling decisions into the kernel description the trace
        generator consumes.

        Same shape the old scheduler emits, because `generate_trace` reads these
        lists directly to index the profiled latency tables: q is what is
        computed this step, k is the KV already present.
        """
        total_len = kv_len = num_prefill = num_decode = 0
        q_list: List[int] = []
        k_list: List[int] = []
        prefill_q_list: List[int] = []
        prefill_k_list: List[int] = []
        decode_k_list: List[int] = []

        for req, take in scheduled:
            if req.is_prefill():
                total_len += take
                if req.is_init:                 # first chunk only
                    req.set_que_delay(now)
                    if req.first_sched_ts < 0:
                        req.first_sched_ts = now
                    if req.first_cache_hit < 0:
                        req.first_cache_hit = req.npu_cache_hit
                q_list.append(take)
                prefill_q_list.append(take)
                prefill_k_list.append(req.num_computed_tokens)
                num_prefill += 1
            else:
                total_len += 1
                q_list.append(1)
                kv_len += req.num_computed_tokens
                decode_k_list.append(req.num_computed_tokens)
                num_decode += 1

        batch = Batch(self.get_batch_id(), self.model, total_len, kv_len,
                      q_list, k_list, num_prefill, num_decode,
                      prefill_q_list, prefill_k_list, decode_k_list,
                      now, kv_size=0)
        batch.fired.append(self.start_npu)
        batch.requests.extend(req for req, _ in scheduled)
        self.inflight.append(batch)
        return batch

    def _prefix_match(self, req: Request) -> None:
        """Advance this request past the prompt tokens already in the cache.

        The KV plane already declines to charge for them -- `allocate` probes
        the tree and reserves only the delta -- but memory is half the story: a
        cached token must also not be RECOMPUTED. Without this the prefix cache
        saves pool space and nothing else, every turn re-prefills its whole
        prompt, and the retention policies this simulator exists to compare are
        scored in a world where keeping a prefix buys nothing.

        Re-matched on every attempt until the prefix is locked, because vLLM
        looks computed blocks up at schedule time: a hit taken on an earlier
        attempt can be evicted while the request waits, and acting on a stale
        hit means skipping compute for tokens that are no longer resident.

        The cap at `input - 1` is vLLM's, verbatim in intent
        (v1/core/kv_cache_manager.py: "When all tokens hit the cache, we must
        recompute the last token to obtain logits"). Without it a fully-cached
        prompt reaches `num_computed_tokens == original_input`, `is_prefill()`
        goes false, and the request is never scheduled at all.
        """
        if not self.enable_prefix_caching or not req.is_prefill():
            return
        if not req.input_hash_ids:
            return
        if not (req.num_computed_tokens == 0
                or (not req._prefix_locked
                    and req.num_computed_tokens <= req.npu_cache_hit)):
            return
        # original_input, not input: after a RECOMPUTE preemption the prompt IS
        # the old prompt plus what the request had generated, and vLLM matches
        # against the whole current sequence. Keying on the submitted length
        # capped the rematch at the original prompt and re-prefilled the
        # generated context every time (memory_model.prefix_match, same fix).
        max_hit = max(0, req.original_input - 1)
        prev_hit = req.npu_cache_hit
        hit = self.kv.probe(self._key(req, max_hit))
        req.npu_cache_hit = hit
        req.prefix_cache_hit = hit
        # Hit rate is counted once per request, at its FIRST match, and against
        # the whole prompt -- the old plane's rule (radix_tree.py:371), where
        # the same request re-inserted after each chunk must not be counted
        # again. Counting per match instead would divide by a denominator that
        # grows with how often a request happened to be re-ranked.
        if not req._prefix_npu_stats_counted:
            self.prefix_requested_tokens += req.original_input
            self.prefix_hit_tokens += hit
            req._prefix_npu_stats_counted = True
        # From a standing start, or when every token of progress so far IS the
        # credited hit -- then the fresh probe replaces it OUTRIGHT, downwards
        # included. A request mid-prefill that has computed past its hit is left
        # alone: pulling it back would recompute what it just did.
        #
        # The retraction is the point. A hit credited on an earlier attempt can
        # be evicted while the request waits -- by the reclaim pass run for
        # somebody else, which sees an unlocked chain belonging to a request
        # that is not running. Keeping the stale count then skips compute for
        # tokens that are gone, and `commit` republishes the whole context on a
        # reservation sized for one chunk: the pool goes over capacity by the
        # difference and nothing can reclaim it.
        if req.num_computed_tokens == 0 or req.num_computed_tokens <= prev_hit:
            req.num_computed_tokens = hit

    def _chunk(self, req: Request, budget: int) -> int:
        """How much prefill this request gets this step.

        Chunked prefill: a long prompt is split so one request cannot monopolise
        a step. Cached prefix is not recomputed -- that is the whole value of the
        prefix cache, and charging for it would make every cache-rescuing policy
        look worthless.
        """
        if not req.is_prefill():
            return 0
        take = min(req.original_input - req.num_computed_tokens, budget)
        if self.long_prefill_token_threshold:
            take = min(take, self.long_prefill_token_threshold)
        return max(0, take)

    def _queue_snapshot(self, req: Request) -> QueueSnapshot:
        """What an admission gate is allowed to read, built fresh each tick."""
        occ = self.kv.occupancy()
        return QueueSnapshot(
            n_running=len(self.running),
            n_waiting=len(self.waiting),
            n_inflight=len(self.inflight),
            kv_utilization=self.kv.pressure(),
            kv_free_tokens=occ["free"],
            kv_evictable_tokens=occ["cached"],
            prompt_tokens=req.original_input,
            cached_tokens=max(0, req.npu_cache_hit),
        )

    def _pressure_fn(self):
        """The policy's say in who gives up context, or None for the valve.

        Resolved per call rather than stored, because a policy is attached
        after construction and may not implement this hook at all -- and a
        `None` here is the difference between the engine's LRU and a policy
        that always declines, which the counters would report identically.
        """
        if self.policy is None:
            return None
        return getattr(self.policy, "pressure_fn", None)

    def _admit(self, req: Request, now: float) -> bool:
        """Ask the policy whether this request may start now."""
        if self.admit_fn is None:
            return True
        state = self._state(req)
        if state is None:
            return True
        try:
            ok = self.admit_fn(state, self._queue_snapshot(req), now)
        except Exception:           # a bad candidate must not stall the queue
            return True
        if ok is False:
            self.counters["admission_holds"] += 1
            return False
        return True

    # ---------------------------------------------------------------- KV

    def _key(self, req: Request, upto: int) -> List[int]:
        """The token ids this request holds, up to `upto` tokens.

        Real ids where the engine supplied them (`input_hash_ids`), because a
        prefix tree only shares what actually matches. A request without them
        gets a private range, which shares with nobody -- deliberately
        pessimistic, since a request silently sharing a prefix it does not have
        would understate memory and overstate cache hits.
        """
        ids = req.input_hash_ids
        if ids:
            key = list(ids)[:upto]
        else:
            base = _PRIVATE_PROMPT + (abs(hash(req.id)) % 1_000_000) * 1_000_000
            key = list(range(base, base + upto))
        if len(key) >= upto:
            return key
        # Past the prompt these are GENERATED tokens, and the engine supplies no
        # ids for them -- `input_hash_ids` is the prompt. They occupy KV all the
        # same: a five-hundred-token answer is five hundred tokens of cache.
        # Stopping at the prompt is what made a turn's footprint stop growing
        # the moment prefill ended, so the pool never filled, nothing was ever
        # reclaimed, and every retention policy was scored against a cache under
        # no pressure at all.
        #
        # Private to this request, because a generated token is shared with
        # nobody: vLLM hashes a block by its token ids AND the prefix behind it,
        # so two requests that happened to produce the same text still hold two
        # different blocks. The namespace is disjoint from the no-ids fallback
        # above, so a request with real prompt ids and one without can never
        # collide on a decode token.
        want = upto - len(key)
        # REAL ids where the trace knows them. A turn's generated tokens are
        # verbatim the opening of its successor's prompt, so the orchestrator
        # can hand them over, and the blocks they form are matchable by the very
        # next turn of this program -- which is what happens on real hardware.
        #
        # Numbering them privately instead made a class of block that nothing
        # can ever match, yet that still occupies the pool and still refreshes
        # its LRU stamp on every decode step. Running requests' unmatchable
        # decode blocks then outranked waiting programs' matchable context and
        # the LRU took the context: 34 turns of 797 re-prefilled from a
        # 48-token header instead of their ~10,000-token prefix, 87,040 hit
        # tokens, all of it at turn index 9+ where a program has lived long
        # enough to go stale during a tool gap.
        pid = program_of(req)
        real = ()
        if pid is not None:
            real = self.orch.generated_ids(pid, req.sub_request_index, want)
        key.extend(real)
        if len(key) < upto:
            # No successor (the last turn), or a trace with no ids. Private,
            # because a token nobody will present must not collide with one
            # they will.
            base = _PRIVATE_DECODE + (abs(hash(req.id)) % 1_000_000) * 1_000_000
            key.extend(range(base + len(real), base + len(real) + (upto - len(key))))
        return key

    def _acquire(self, req: Request, n_tokens: int, now: float,
                 may_preempt: bool) -> bool:
        """Get KV for this request's next `n_tokens`, reclaiming if needed.

        The order sets the preemption rate, so it is explicit:

          1. ask the KV plane outright
          2. on failure, reclaim -- cached blocks and expired pins cost only a
             later cache miss
          3. only then, and only for a RUNNING request, preempt someone

        Skipping step 2 is how a simulator preempts many times more often than
        the engine it models while every aggregate still looks plausible.
        """
        pid = program_of(req)
        if pid is None:
            return True                     # flat request: no program KV plane

        # Hold the prefix `_prefix_match` just credited, BEFORE anything below
        # reclaims. Those blocks are CACHED, and a request that was preempted a
        # moment ago owns the least recently used chain in the tree -- so the
        # reclaim pass run on this request's behalf takes this request's own
        # prefix, and then `commit` republishes the whole context having been
        # charged for one chunk. The pool goes over capacity by the difference.
        #
        # vLLM's order, and the old plane's (scheduler.py:575): take the
        # computed blocks, then allocate the new ones, and give the computed
        # ones back if the allocation does not happen.
        took_hold = False
        if (req.is_prefill() and not req._prefix_locked
                and req.num_computed_tokens > 0
                and self.kv.hold_prefix(
                    req.id, self._key(req, req.num_computed_tokens))):
            req._prefix_locked = took_hold = True

        def give_up() -> bool:
            # Only a hold taken on THIS call. A running request's lock is on
            # the tail it has committed and is not ours to drop.
            if took_hold:
                self.kv.drop_hold(req.id)
                req._prefix_locked = False
            return False

        if self.kv.allocate(pid, n_tokens, now, owner=req.id) is not None:
            return True
        # Ask pressure for what is actually SHORT, not for the step's token
        # count. A decode step computes one token but the block it lands in may
        # be thirteen tokens from fitting; reclaiming one and failing leaves the
        # engine stuck with a cache full of blocks it never asked to have back.
        self.kv.on_pressure(self.kv.reclaim_target(n_tokens), now,
                            policy=self._pressure_fn())
        if self.kv.allocate(pid, n_tokens, now, owner=req.id) is not None:
            return True
        if not may_preempt:
            return give_up()

        while self.running:
            victim = self._choose_victim(now)
            if victim is None or victim is req:
                break
            self._preempt(victim, now)
            # Preemption RELEASES the victim's blocks; it does not free them --
            # they become ordinary cache, which is what vLLM does too. But
            # `can_fit` counts only genuinely free tokens, so without asking
            # pressure to take the cache back the retry fails on memory that is
            # sitting right there, and the preemption bought nothing. That is
            # how one request stalled at 601 tokens forever while the other was
            # preempted on every single step.
            self.kv.on_pressure(self.kv.reclaim_target(n_tokens), now,
                                policy=self._pressure_fn())
            if self.kv.allocate(pid, n_tokens, now, owner=req.id) is not None:
                return True
        return give_up()

    # ----------------------------------------------------------- preemption

    def _choose_victim(self, now: float) -> Optional[Request]:
        """Policy first, then the engine's default.

        The default is the lowest-priority running request, breaking ties by the
        latest arrival -- the vLLM-v1 shape, where the most recently admitted
        work is the cheapest to undo.
        """
        if not self.running:
            return None
        if self.victim_fn is not None:
            try:
                chosen = self.victim_fn(
                    [RunningView(
                        request_id=r.id,
                        state=self._state(r),
                        priority=r.priority,
                        prompt_tokens=r.input,
                        computed_tokens=r.num_computed_tokens,
                        generated_tokens=max(0, r.num_computed_tokens - r.input),
                        is_prefill=r.is_prefill(),
                    ) for r in self.running], now)
            except Exception:
                chosen = None
            if chosen is not None:
                for req in self.running:
                    if req.id == chosen:
                        self.counters["victim_overrides"] += 1
                        return req
        # vLLM v1 preempts `running[-1]`: the most recently ADMITTED request,
        # because it has computed the least and so is the cheapest to undo. The
        # old plane spells the same thing `max(running_in, key=r.admit_seq)`
        # (scheduler.py:613), and switches to the priority key only under
        # priority scheduling (scheduler.py:611).
        #
        # This plane used the priority key unconditionally, and ranked by
        # ARRIVAL. Those differ precisely for a request that was preempted and
        # re-admitted: fresh admit_seq, original arrival. vLLM takes it again;
        # ranking by arrival treats it as senior and takes a longer-running
        # request instead -- discarding more computed tokens, which are then
        # re-prefilled, which raises pressure, which preempts again. 968
        # preemptions on the board cell against the engine's 590.
        if self.priority_fn is not None:
            return max(self.running,
                       key=lambda r: (r.priority if r.priority is not None
                                      else -1, r.arrival, r.id))
        return max(self.running,
                   key=lambda r: (r.admit_seq if r.admit_seq is not None
                                  else -1, r.arrival, r.id))

    def _preempt(self, req: Request, now: float) -> int:
        """Recompute preemption: release the KV, discard prefill progress, and
        put the request back at the FRONT of the waiting set.

        Front, not back: it has already waited once, and re-queueing it behind
        everything turns a memory event into a fairness decision nobody made.
        """
        pid = program_of(req)
        freed = 0
        if pid:
            self.kv.drop_reservation(pid, owner=req.id)      # its step will never complete
            # Drop the victim's OWN lock first. `evict` refuses locked nodes by
            # design -- a live turn's KV is not a policy's to take -- so without
            # this, preemption removes the request and frees nothing: its blocks
            # stay locked by a request that no longer exists, and the pool loses
            # them for the rest of the run. Preempting then became a way to
            # shrink the pool rather than to reclaim it.
            freed = self.kv.release(
                self._key(req, req.num_computed_tokens), owner=req.id)
        # Released, not evicted. vLLM's preemption frees the request's blocks to
        # the free queue where they stay hashed and cache-hittable until
        # something actually reuses them; the pressure valve is what takes them.
        # Deleting them here instead would destroy the prefix the request is
        # about to recompute against, and charge every preemption a full
        # re-prefill the real engine does not always pay.
        # Recompute preemption re-prefills the prompt PLUS whatever was
        # generated: those tokens are context now and their KV is gone. Zeroing
        # computed tokens without folding them in would silently shorten the
        # turn. scheduler.py:1072 does the same.
        # vLLM counts the token it just sampled: request.num_tokens is prompt +
        # output_token_ids, appended before the preemption, so after computing
        # N the sequence is N+1 long and max_cache_hit_length is N. Without the
        # +1 the recompute prompt is one short, the cap lands one below a page
        # boundary, and a fully cached context floors to a hit a page smaller.
        # scheduler.py::_preempt_recompute does the same; the two planes
        # disagreeing on it made their preemption costs incomparable. Only in
        # decode -- a request preempted mid-prefill has sampled nothing.
        if req.num_computed_tokens >= req.original_input:
            generated = req.num_computed_tokens - req.original_input + 1
        else:
            generated = 0
        req.original_input += generated
        req.num_computed_tokens = 0
        req.chunk_len = 0
        req.admit_seq = None         # no longer running; re-stamped if re-admitted
        req._prefix_locked = False   # `release` above dropped the hold
        req.n_preempted += 1
        self.running.remove(req)
        # Leave any batch still in flight. A preempted request keeps its place
        # in `batch.requests` otherwise, and when that batch reports,
        # `add_done` advances the victim's progress and RE-COMMITS its KV --
        # undoing the preemption from inside a step that already ended, and
        # taking back the memory preemption just released. It also leaves the
        # victim at `num_computed_tokens == original_input`: a DECODE sitting
        # in the waiting queue, which `_chunk` sizes at zero, so it is never
        # admitted again. That is the deadlock: 12 requests waiting, nothing
        # running, 12 locks held by requests that are not running, and the head
        # of the queue needing one token that was free.
        for batch in self.inflight:
            if req in batch.requests:
                batch.requests.remove(req)
        self.waiting.insert(0, req)
        self.last_preempted.append(req)
        self.counters["preemptions"] += 1
        self.num_preemptions += 1
        return freed

    def ensure_capacity(self, need_tokens: int, now: float) -> List[Request]:
        """Free `need_tokens`, preempting running requests if the KV plane
        cannot. Reclaim first: cached blocks and broken pins cost only a later
        cache miss, where a preemption costs a whole prefill."""
        reclaim = self.kv.on_pressure(need_tokens, now,
                                      policy=self._pressure_fn())
        if reclaim.tokens >= need_tokens:
            return []
        evicted: List[Request] = []
        still = need_tokens - reclaim.tokens
        while still > 0 and self.running:
            victim = self._choose_victim(now)
            if victim is None:
                break
            still -= self._preempt(victim, now)
            evicted.append(victim)
        return evicted

    def add_done(self, id, sys, finish):
        """A batch has finished on one NPU. Advance only when ALL of them have.

        This is the engine's completion protocol and the signature the main loop
        calls. Three things happen here and nowhere else, because until every
        NPU of the instance reports, none of them is true:

          * `num_computed_tokens` advances by the chunk this batch scheduled,
            which is what makes the next `schedule` see progress instead of
            reissuing the same chunk
          * a finished prefill is committed to the prefix tree, becoming
            matchable by other programs. Publishing at decision time would let a
            second program hit on a prefix nobody had computed yet.
          * finished requests are handed back so the router can release the next
            turn of their program

        The barrier is not a detail: a batch fires on `num_npus` NPUs and each
        reports separately, so advancing on the first report would count every
        chunk `num_npus` times on any TP>1 instance -- which is every arena cell.
        """
        prompt_t = 0
        gen_t = 0
        end_reqs: List[Request] = []
        if not self.inflight:
            return prompt_t, gen_t, end_reqs

        id -= 1
        batch = next((b for b in self.inflight if b.batch_id == id), None)
        if batch is None or sys in batch.end:
            return prompt_t, gen_t, end_reqs

        batch.end.append(sys)
        if (self.start_npu not in batch.end
                or (self.start_npu + self.num_npus - 1) not in batch.end):
            return prompt_t, gen_t, end_reqs      # not every NPU is in yet

        for req in batch.requests:
            pid = program_of(req)
            if req.is_prefill():
                chunk = req.chunk_len or (req.original_input
                                          - req.num_computed_tokens)
                req.num_computed_tokens += chunk
                req.chunk_len = 0
                if req.num_computed_tokens >= req.original_input:
                    # prefill finished this step: publish it and start decoding
                    if pid is not None and self.enable_prefix_caching:
                        self.kv.commit(pid, self._key(req,
                                                      req.num_computed_tokens),
                                       owner=req.id)
                    req.is_init = False
                    prompt_t += chunk + req.prefix_cache_hit
                    req.set_ttft(finish)
                    # The last prefill token through lm_head produces an output
                    # token, and on a RESUMED prefill it is a real one: the +1
                    # in _preempt above puts the recompute prompt one position
                    # further on, so it emits the next token rather than a
                    # repeat. It is only not the FIRST token -- and the interval
                    # of a resumed one is where the preemption stall shows up.
                    # scheduler.py's completion path does the same.
                    gen_t += 1
                    if req.first_token_counted:
                        req.add_itl(finish)
                    else:
                        req.first_token_counted = True
                else:
                    prompt_t += chunk         # more chunks to come
            else:
                req.num_computed_tokens += 1
                gen_t += 1
                # Inter-token latency, one entry per interval. The first token
                # lands on the step prefill finishes and `set_ttft` starts the
                # clock there, so twenty generated tokens give nineteen
                # intervals -- which is what the old plane writes.
                req.add_itl(finish)
                if pid is not None and self.enable_prefix_caching:
                    self.kv.commit(pid, self._key(req, req.num_computed_tokens),
                                   owner=req.id)

            # The engine's expression, and it must stay verbatim.
            #
            # `req.output` is NOT a count of generated tokens: in the traces it
            # is a cumulative target, input + output, so a request with a 5,714
            # token prompt carries output=5,734 for 20 generated tokens. The
            # counter it is compared against likewise runs from 0 through the
            # prompt and on into decode, so the two are on the same scale.
            #
            # A version of this expressed against a decode count instead --
            # `decoded >= output - 1` -- demanded 5,733 decode steps where this
            # demands 19, which is how a five-request trace failed to finish in
            # six minutes. Measured on the mini trace, 2026-09-13.
            if req.output <= req.num_computed_tokens + 1:
                req.end_time = finish
                req.add_latency(finish)
                self.done.append(req)
                if req in self.running:
                    self.running.remove(req)
                if pid is not None:
                    self.kv.release(self._key(req, req.num_computed_tokens),
                                    owner=req.id)
                end_reqs.append(req)
                # A request finishing IS a turn ending -- on this plane there is
                # no third party who knows better, so there is no reason to make
                # the main loop say it again. Except on a prefill instance,
                # where the request is only half done and continues on a decode
                # instance; that one's `add_done` reports the turn.
                if self.pd_type != "prefill":
                    self.turn_complete(req, finish)

        self.inflight.remove(batch)
        return prompt_t, gen_t, end_reqs

    def turn_complete(self, req: Request, now: float) -> None:
        """Tell the orchestrator a program's turn is over and its gap begins.

        Everything it needs is already known to one of the two planes, so
        nothing is passed in but the request and the clock. Which node finished
        is on the request; what that node was is the orchestrator's in-flight
        record; whether more turns follow is the trace, which the orchestrator
        holds. The old router had to be told all three, and being told a fact
        you could look up is how two copies of it start to differ.

        `service_s` is measured the way the real driver measures it -- first
        schedule to last token, prefill included (bench/core/runner.py:
        `service_s = max(0.0, lt - st)`). An earlier version used
        `latency - queuing_delay`, which excludes prefill and so under-charges
        exactly the large-prompt programs a service-ranking policy exists to
        deprioritise.
        """
        if req in self.running:
            self.running.remove(req)
        pid = program_of(req)
        if pid is None:
            return
        if req.end_time >= 0 and req.first_sched_ts >= 0:
            service_s = max(0.0, (req.end_time - req.first_sched_ts) / 1e9)
        else:
            service_s = 0.0
        self.orch.on_turn_complete(pid, now, service_s,
                                   node_id=req.sub_request_index)
        # After, not before: the retention decision reads the gap the turn is
        # entering (its tool, its context), and none of that is true of the
        # state until the completion has been recorded.
        if self.policy is not None:
            self.policy.on_turn_complete(self.kv, self.orch.get(pid),
                                         req.id, now)

    # -------------------------------------------------------- completion

    # ---------------------------------------------------------- reporting

    def return_prefix_info(self):
        """`((npu_requested, npu_hit), (cpu_requested, cpu_hit))`.

        Same shape the old plane returns, because the caller prints both and
        should not have to know which plane it is talking to. The second pair is
        zero: this plane has one tier, so there is no CPU-resident prefix to
        report rather than a number waiting to be filled in.
        """
        return ((self.prefix_requested_tokens, self.prefix_hit_tokens), (0, 0))

    def mem_report(self) -> Dict[str, float]:
        """The same questions `scheduler.mem_report` answers, from the planes.

        `weight` has no entry: model-weight memory is not KV and this plane does
        not account for it. `npu_used` and `npu_mem` are therefore KV tokens,
        not bytes -- the pool is pinned in tokens because that is the quantity
        two different hosts can be held equal on.
        """
        occ = self.kv.occupancy()
        used = occ["locked"] + occ["pinned"] + occ["cached"]
        return {
            "waiting": len(self.waiting),
            "batched": sum(len(b.requests) for b in self.inflight),
            "referenced_tokens": sum(r.num_computed_tokens for r in self.running),
            "cache_total": used,
            "cache_evictable": occ["cached"],
            "cache_protected": occ["locked"],
            "kv_util": self.kv.pressure(),
            "reserved_tokens": self.kv.inflight_tokens(),
            "npu_used": used,
            "npu_mem": self.kv.capacity_tokens,
            "cpu_used": 0,
        }

    def teardown(self) -> bool:
        """Nothing to free: the tree is in-memory and no weight is held here.
        Clean means every live reference was released."""
        return self.kv.occupancy()["locked"] == 0

    def snapshot(self) -> Dict[str, int]:
        return {"waiting": len(self.waiting), "running": len(self.running),
                "inflight": len(self.inflight), **self.counters}
