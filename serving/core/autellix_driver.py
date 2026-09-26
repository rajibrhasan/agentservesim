"""Simulator-side executor for Autellix's MLFQ scheduler.

The decision logic is `policies.autellix_runtime.AutellixRuntime` -- the same
object `bench/core/policy_engine.py` drives on real vLLM -- so the simulator
runs Algorithm 1 of arXiv:2502.13965 itself rather than the attained-service
approximation `--scheduling plas` provides: per-call quanta, demotion on
quantum exhaustion, service inheritance across a program's turns, starvation
promotion, and optional extra admission candidates. The simulator replans
each batch; it does not implement the native adapter's N-step planning window.

What this driver supplies is the engine half of that contract: cumulative
batch feasibility against the simulator's own memory model and token budget,
the actual batch membership, and the measured execution interval of every
batch. One runtime per instance, because queues and clocks are per engine.

Limits worth stating. The runtime assumes one batch in flight at a time, which
holds only at `pp_size == 1` (`Scheduler.schedule_with_prefix` returns early
once `len(inflight) >= pp_size`), so the driver refuses anything else. Calls
the planner drops from the resident set yield before admission, even without
memory pressure. Optional host copies preserve their full cached blocks;
without them, or when the copy queue is full, they resume by recomputation.
This remains an approximation of native physical-block swapping.
Flat (non-program) requests are invisible to
the runtime and keep the engine's own ordering.
"""
from policies.autellix_runtime import AutellixRuntime, QueueConfig


class AutellixDriver:
    """One engine's worth of Autellix scheduling state."""

    def __init__(self, config: QueueConfig, overprovision: int = 0):
        self.runtime = AutellixRuntime(config)
        self.overprovision = int(overprovision)
        #: request id -> program id, for every call the runtime has admitted.
        self.known = {}
        #: ids handed to start_batch, so finish_batch reports the same set.
        self._in_flight = None
        self._queues_at_start = {}
        self.stats = {"planned": 0, "selected": 0, "overprovisioned": 0,
                      "plan_preempt": 0, "demoted": 0, "promoted": 0}

    # ------------------------------------------------------------------
    # bookkeeping
    # ------------------------------------------------------------------
    @staticmethod
    def _rid(req):
        return str(req.id)

    def _admit(self, req, program_id, now_s):
        """Register a call the first time the planner sees it."""
        rid = self._rid(req)
        if rid in self.known:
            return True
        if program_id is None:
            return False
        self.runtime.arrive(rid, str(program_id), now_s)
        self.known[rid] = str(program_id)
        return True

    def drop(self, req_id, now_s):
        """A call left the engine without finishing in a batch."""
        rid = str(req_id)
        if rid in self.known and self._in_flight is None:
            self.runtime.cancel(rid, now_s)
            del self.known[rid]

    # ------------------------------------------------------------------
    # planning
    # ------------------------------------------------------------------
    def plan(self, waiting, running, pcb_of, now_s, fits):
        """Return the waiting requests Algorithm 1 selects, in queue order.

        `fits` is the read-only cumulative feasibility callback; it receives
        the candidate Request and the Requests already selected. Requests the
        runtime does not know (no program) are appended after the plan so the
        engine's own order still applies to them.
        """
        by_id = {}
        for req in list(running) + list(waiting):
            pcb = pcb_of(req)
            program_id = None if pcb is None else pcb.program_id
            if self._admit(req, program_id, now_s):
                by_id[self._rid(req)] = req

        resident = tuple(self._rid(r) for r in running if self._rid(r) in self.known)
        # plan() is where starvation promotion happens; demotion happens in
        # finish_batch, so the two are counted at their own sites.
        before = {rid: self.runtime.calls[rid].queue
                  for rid in self.known if rid in self.runtime.calls}

        def can_fit(rid, selected_ids):
            req = by_id.get(rid)
            if req is None:          # resident on another instance; not ours
                return False
            return fits(req, tuple(by_id[i] for i in selected_ids if i in by_id))

        plan = self.runtime.plan(now_s, resident, can_fit, self.overprovision)

        for rid, was in before.items():
            call = self.runtime.calls.get(rid)
            if call is not None and call.queue < was:
                self.stats["promoted"] += 1
        self.stats["planned"] += 1
        self.stats["selected"] += len(plan.selected)
        self.stats["overprovisioned"] += len(plan.overprovisioned)
        self.stats["plan_preempt"] += len(plan.preempt)

        chosen = [by_id[r] for r in plan.selected + plan.overprovisioned if r in by_id]
        running_ids = {self._rid(r) for r in running}
        ordered = [r for r in chosen if self._rid(r) not in running_ids]
        unknown = [r for r in waiting if self._rid(r) not in self.known]
        self.last_preempt = tuple(by_id[r] for r in plan.preempt if r in by_id)
        return ordered + unknown

    # ------------------------------------------------------------------
    # batch lifecycle
    # ------------------------------------------------------------------
    def batch_started(self, requests, now_s):
        ids = tuple(dict.fromkeys(self._rid(r) for r in requests
                                  if self._rid(r) in self.known
                                  and self._rid(r) in self.runtime.calls))
        if not ids:
            self._in_flight = None
            self._queues_at_start = {}
            return
        self.runtime.start_batch(ids, now_s)
        self._in_flight = ids
        self._queues_at_start = {r: self.runtime.calls[r].queue for r in ids}

    def batch_finished(self, finished_ids, now_s, execution_s):
        if self._in_flight is None:
            return
        done = tuple(r for r in dict.fromkeys(str(i) for i in finished_ids)
                     if r in self._in_flight)
        self.runtime.finish_batch(now_s, done, execution_s=max(0.0, execution_s))
        for rid, was in self._queues_at_start.items():
            call = self.runtime.calls.get(rid)
            if call is not None and call.queue > was:
                # Quantum exhausted: Algorithm 1 demotes the call one queue.
                self.stats["demoted"] += 1
        for rid in done:
            self.known.pop(rid, None)
        self._in_flight = None
        self._queues_at_start = {}


def queue_config(service_boundaries_s, quanta_s, starvation_ratio):
    """Build the queue configuration, refusing to invent the parameters.

    Autellix publishes no transferable defaults for these, and the engine
    adapter already treats them as explicit experiment inputs, so the
    simulator does the same rather than making a number up and having a
    later reader mistake it for the paper's.
    """
    if not service_boundaries_s or not quanta_s or starvation_ratio is None:
        raise ValueError(
            "--scheduling autellix-mlfq needs --autellix-service-boundaries, "
            "--autellix-quanta (one more entry than boundaries) and "
            "--autellix-starvation-ratio: the queue thresholds are experiment "
            "inputs, not paper defaults.")
    return QueueConfig(tuple(float(x) for x in service_boundaries_s),
                       tuple(float(x) for x in quanta_s),
                       float(starvation_ratio))
