from policies.saga_runtime import TaskEstimate, fair_shares

NS = 1e9


class SagaFairness:
    """Per-tenant shares from live program state."""

    def __init__(self, overdue_slack_s=1.0):
        if not (overdue_slack_s > 0):
            raise ValueError("--saga-fairness-slack must be positive: the paper "
                             "leaves deadline-past behaviour unspecified, so the "
                             "minimum slack is a recorded port parameter")
        self.overdue_slack_s = float(overdue_slack_s)
        #: program -> (tenant, deadline seconds)
        self.registry = {}
        #: measured service seconds per completed turn, and the count
        self._service_sum = 0.0
        self._service_turns = 0
        #: turns completed -> [further turns seen, programs seen]
        self._further = {}
        self.shares = {}
        self.stats = {"computed": 0, "tenants": 0}

    # ------------------------------------------------------------- registry
    def register(self, program_id, session):
        """Record a session's tenant and deadline, or say what is missing."""
        tenant = session.get("tenant")
        deadline = session.get("deadline_ns")
        if tenant is None or deadline is None:
            missing = [n for n, v in (("tenant", tenant), ("deadline_ns", deadline))
                       if v is None]
            raise ValueError(
                f"--saga-fairness needs {' and '.join(missing)} on session "
                f"{session.get('session_id')!r}: AFS is a per-tenant score "
                "against a deadline, and neither can be guessed from the trace.")
        self.registry[str(program_id)] = (str(tenant), float(deadline) / NS)

    # --------------------------------------------------------- observations
    def note_turn_service(self, service_s):
        if service_s is not None and service_s >= 0:
            self._service_sum += float(service_s)
            self._service_turns += 1

    def note_turn_complete(self, turns_done):
        """A program reached `turns_done` completed turns. Bucket k counts the
        programs seen at k, against how many of them went on to a (k+1)-th."""
        self._further.setdefault(int(turns_done), [0, 0])[1] += 1

    def note_turn_arrival(self, turns_done):
        """A program with `turns_done` completed turns started another."""
        if turns_done:
            self._further.setdefault(int(turns_done), [0, 0])[0] += 1

    # ------------------------------------------------------------- estimate
    def mean_service_s(self):
        return (self._service_sum / self._service_turns) if self._service_turns else None

    def expected_further_turns(self, turns_done):
        """Turns still to come, from programs already seen at this index."""
        further, seen = self._further.get(int(turns_done), (0, 0))
        if not seen:
            return None
        # A geometric tail from the observed continuation rate: p/(1-p).
        p = min(0.999, further / seen)
        return p / (1.0 - p)

    def remaining_gpu_s(self, turns_done):
        service = self.mean_service_s()
        further = self.expected_further_turns(turns_done)
        if service is None or further is None:
            return None
        return service * further

    # --------------------------------------------------------------- shares
    def compute(self, programs, now_ns):
        """Tenant -> share, or None while nothing has been observed yet."""
        now_s = now_ns / NS
        tasks = []
        for program_id, (tenant, deadline_s) in self.registry.items():
            pcb = programs.get(program_id) if programs.known(program_id) else None
            if pcb is None:
                continue
            remaining = self.remaining_gpu_s(pcb.turns_completed)
            if remaining is None:
                return None
            tasks.append(TaskEstimate(program_id=program_id, tenant=tenant,
                                      remaining_gpu_s=remaining,
                                      deadline_s=max(0.0, deadline_s)))
        if not tasks or not any(t.remaining_gpu_s > 0 for t in tasks):
            # Every program's predicted remaining work is zero, so the scores
            # are all zero and Eq. 9 would hand back a flat table of zeros.
            # There is nothing to apportion; say so instead of stamping the
            # queue with a share that carries no information.
            return None
        self.shares = fair_shares(tasks, now_s, self.overdue_slack_s)
        self.stats["computed"] += 1
        self.stats["tenants"] = len(self.shares)
        return self.shares

    def share_of(self, program_id):
        entry = self.registry.get(str(program_id))
        return None if entry is None else self.shares.get(entry[0])
