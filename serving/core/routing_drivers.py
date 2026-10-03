
from policies.saga_runtime import SagaPlacement, WorkerObservation

NS = 1e9


def _sessions_waiting(sched):
    """(session, arrival seconds) for turns queued on this instance."""
    out = []
    for req in sched.request:
        session = req.session_id if req.session_id is not None else req.workflow_id
        if session is not None:
            out.append((str(session), max(0.0, req.arrival / NS)))
    return tuple(out)


class SagaRouter:
    """Observed-cache affinity plus guarded, acknowledged stealing."""

    def __init__(self, num_instances, affinity_limit=0.8, idle_s=0.1,
                 load_ratio=2.0, seed=0, observation_interval_s=0.1):
        self.placement = SagaPlacement(seed=seed, affinity_limit=affinity_limit,
                                       idle_s=idle_s, load_ratio=load_ratio)
        self.num_instances = int(num_instances)
        #: instance -> simulator seconds its queue last became empty
        self._empty_since = {}
        self.observation_interval_s = observation_interval_s
        self._observation_time = None
        self._routing_workers = ()
        self._routing_cached = {}
        self.stats = {"routed": 0, "affinity_hits": 0, "steals_proposed": 0,
                      "steals_completed": 0, "steals_abandoned": 0}

    # ------------------------------------------------------------ observation
    def observe(self, schedulers, now_s, utilization):
        """A WorkerObservation per instance, from live scheduler state."""
        workers = []
        for i, sched in enumerate(schedulers):
            queued = _sessions_waiting(sched)
            if queued:
                # A busy queue has no empty-since time; the runtime rejects
                # an observation carrying both.
                self._empty_since.pop(i, None)
                empty_since = None
            else:
                empty_since = self._empty_since.setdefault(i, now_s)
            workers.append(WorkerObservation(
                worker=i, load=max(0.0, float(utilization(sched))),
                queued_sessions=queued, empty_since_s=empty_since))
        return tuple(workers)

    @staticmethod
    def cached_on(schedulers, session):
        """Instances whose prefix cache still holds a node owned by session."""
        out = set()
        for i, sched in enumerate(schedulers):
            cache = getattr(sched.memory, "npu_prefix_cache", None)
            if cache is None:
                continue
            stack = [cache.root_node]
            while stack:
                node = stack.pop()
                stack.extend(node.children.values())
                if str(session) in {str(o) for o in node.owners}:
                    out.add(i)
                    stack = []
                    break
        return out

    # ---------------------------------------------------------------- routing
    def route(self, session, schedulers, now_s, utilization):
        if (self._observation_time is None
                or now_s - self._observation_time >= self.observation_interval_s):
            self._routing_workers = self.observe(schedulers, now_s, utilization)
            cached = {}
            for i, sched in enumerate(schedulers):
                cache = sched.memory.npu_prefix_cache
                if cache is None:
                    continue
                stack = list(cache.root_node.children.values())
                while stack:
                    node = stack.pop()
                    stack.extend(node.children.values())
                    for owner in node.owners:
                        cached.setdefault(str(owner), set()).add(i)
            self._routing_cached = cached
            self._observation_time = now_s
        workers = self._routing_workers
        cached = self._routing_cached.get(str(session), set())
        home = self.placement.home.get(str(session))
        choice = self.placement.route(str(session), workers, cached)
        from runtime.routing_trace import record_saga_route
        record_saga_route('sim', session, now_s, self._observation_time,
                          home, workers, cached, choice)
        self.stats["routed"] += 1
        if home is not None and choice == home and home in cached:
            self.stats["affinity_hits"] += 1
        self.placement.home[str(session)] = choice
        return int(choice)

    # --------------------------------------------------------------- stealing
    def steal(self, schedulers, now_s, utilization, move):
        """Let one idle worker take the oldest queued session from a busy one.

        `move(session, source, destination)` performs the transfer and returns
        True when the turn actually changed instance. A proposal that cannot be
        carried out is reported back as not published, so SagaPlacement clears
        it instead of leaving the session permanently un-stealable.
        """
        workers = self.observe(schedulers, now_s, utilization)
        moved = []
        for worker in workers:
            proposal = self.placement.propose_steal(worker.worker, workers, now_s)
            if proposal is None:
                continue
            self.stats["steals_proposed"] += 1
            published = bool(move(proposal.session, proposal.source,
                                 proposal.destination))
            self.placement.complete_steal(proposal, published)
            if published:
                self.stats["steals_completed"] += 1
                moved.append(proposal)
            else:
                self.stats["steals_abandoned"] += 1
            # One steal per tick: the observations are now stale.
            break
        return tuple(moved)


class AutellixRouter:
    """Short prompts to the least loaded engine, long prompts to their home."""

    def __init__(self, num_instances, long_prompt_tokens=2048):
        if long_prompt_tokens <= 0:
            raise ValueError("the long-prompt threshold must be positive")
        self.num_instances = int(num_instances)
        self.long_prompt_tokens = int(long_prompt_tokens)
        self.home = {}
        self.stats = {"short_routed": 0, "long_routed": 0, "home_hits": 0}

    def route(self, session, prompt_tokens, schedulers, outstanding):
        if int(prompt_tokens) > self.long_prompt_tokens:
            home = self.home.get(str(session))
            self.stats["long_routed"] += 1
            if home is not None and 0 <= home < len(schedulers):
                self.stats["home_hits"] += 1
                return int(home)
            choice = min(range(len(schedulers)),
                         key=lambda i: (outstanding(schedulers[i]), i))
            self.home[str(session)] = choice
            return int(choice)
        self.stats["short_routed"] += 1
        # A short prompt does not establish a long-call home: the paper homes
        # a program by where its expensive prefills ran.
        return int(min(range(len(schedulers)),
                       key=lambda i: (outstanding(schedulers[i]), i)))
