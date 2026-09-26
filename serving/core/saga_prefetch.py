"""SAGA's prefetch: recompute an evicted context before its tool returns.

This mirrors what the engine integration does (`bench/core/runner.py::prefetch`
-- a one-token generation over the session's prompt, pinned until the result
is due), not a host-to-device transfer. SAGA prefetches by *recomputation*, so
the simulator does the same: a synthetic prefill is submitted for a session
whose learned tool gap is nearly elapsed, and its prefix is protected until
the real successor arrives and hits it.

Timing comes from the learned per-tool distribution that `--retention
saga-tool-ttl` already maintains (`SagaToolTTL.predicted_gap_s`, whose own
docstring names this caller). Without that estimator there is no schedule to
prefetch against, and this refuses rather than guessing a gap.

The cost is real and is paid where it falls: the synthetic prefill occupies the
batch, consumes KV, and competes with live work, so a prefetch issued too
early is visibly expensive rather than free. Its request is marked so metrics
never count it as a program turn.
"""
NS = 1e9
#: Prefetch request ids live above any workload id so they cannot collide.
ID_BASE = 1_000_000_000


class SagaPrefetcher:
    """Decides which sessions to recompute ahead, and when."""

    def __init__(self, predicted_gap_s, margin_s=0.5, max_in_flight=1):
        if predicted_gap_s is None:
            raise ValueError(
                "--saga-prefetch needs --retention saga-tool-ttl: the lead time "
                "comes from that policy's learned per-tool gap distribution, "
                "and there is no other estimate here to use.")
        if margin_s < 0:
            raise ValueError("the prefetch margin cannot be negative")
        self.predicted_gap_s = predicted_gap_s
        self.margin_s = float(margin_s)
        self.max_in_flight = int(max_in_flight)
        #: program -> (token ids, instance) captured when its gap began
        self.parked = {}
        #: program ids with a prefetch submitted and not yet consumed
        self.in_flight = {}
        self._next_id = ID_BASE
        self.stats = {"issued": 0, "cancelled": 0, "skipped_resident": 0}

    # ---------------------------------------------------------------- events
    def note_gap_start(self, program_id, req_obj, instance, now_ns):
        """Remember what a paused session would have to recompute."""
        if program_id is None or req_obj is None:
            return
        ids = list(req_obj.input_hash_ids or []) + list(req_obj.output_hash_ids or [])
        if len(ids) > 1:
            self.parked[str(program_id)] = (ids[:-1], int(instance), int(now_ns))

    def note_turn_arrival(self, program_id):
        """The real successor arrived: the prefetch, if any, has done its job."""
        self.parked.pop(str(program_id), None)
        self.in_flight.pop(str(program_id), None)

    # ----------------------------------------------------------------- timing
    def due(self, programs, now_ns, resident):
        """Sessions whose predicted gap is within the margin of returning.

        `resident(program_id, ids, instance)` reports whether the context is
        still on that instance; a resident context needs no recompute.
        """
        out = []
        if len(self.in_flight) >= self.max_in_flight:
            return out
        now_s = now_ns / NS
        for program_id, (ids, instance, _) in list(self.parked.items()):
            if program_id in self.in_flight:
                continue
            if not programs.known(program_id):
                continue
            pcb = programs.get(program_id)
            if not pcb.in_gap or pcb.gap_started_ts is None:
                continue
            gap = self.predicted_gap_s(pcb.tool_name)
            if gap is None:
                continue
            elapsed = now_s - pcb.gap_started_ts
            if elapsed < gap - self.margin_s:
                continue        # the tool is not due back yet
            if resident(program_id, ids, instance):
                self.stats["skipped_resident"] += 1
                self.parked.pop(program_id, None)
                continue
            out.append((program_id, ids, instance))
            if len(out) + len(self.in_flight) >= self.max_in_flight:
                break
        return out

    def next_id(self):
        self._next_id += 1
        return self._next_id

    def note_issued(self, program_id, request_id):
        self.in_flight[str(program_id)] = request_id
        self.parked.pop(str(program_id), None)
        self.stats["issued"] += 1

    def cancel(self, program_id):
        if self.in_flight.pop(str(program_id), None) is not None:
            self.stats["cancelled"] += 1
