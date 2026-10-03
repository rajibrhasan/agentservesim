from policies.saga_runtime import CacheObservation, Successor, eviction_order

NS = 1e9


class SagaEvictionOrder:
    """Per-instance observations feeding SAGA's reclaim ranking."""

    def __init__(self):
        #: session -> simulator ns of its last turn boundary
        self.last_access_ns = {}
        #: session -> turns completed so far
        self.turns_done = {}
        #: turns completed -> [sessions seen to continue, sessions seen at all]
        self._survival = {}
        #: running mean of measured prefix overlap on successor turns
        self._overlap_sum = 0.0
        self._overlap_n = 0
        self.stats = {"ranked": 0, "evicted_tokens": 0, "lru_fallback": 0}

    # ---------------------------------------------------------- observations
    def note_turn_complete(self, session, now_ns):
        if session is None:
            return
        self.last_access_ns[session] = int(now_ns)
        done = self.turns_done.get(session, 0) + 1
        self.turns_done[session] = done
        bucket = self._survival.setdefault(done, [0, 0])
        bucket[1] += 1

    def note_turn_arrival(self, session, now_ns, overlap=None):
        """A successor turn arrived: the previous turn's context was reused."""
        if session is None:
            return
        self.last_access_ns[session] = int(now_ns)
        done = self.turns_done.get(session, 0)
        if done:
            self._survival.setdefault(done, [0, 0])[0] += 1
        if overlap is not None:
            self._overlap_sum += max(0.0, min(1.0, float(overlap)))
            self._overlap_n += 1

    # ------------------------------------------------------------- estimates
    def reuse_probability(self, session):
        """P(a (k+1)-th turn | k completed), from sessions already seen."""
        continued, total = self._survival.get(self.turns_done.get(session, 0),
                                              (0, 0))
        return (continued / total) if total else None

    def overlap(self):
        return (self._overlap_sum / self._overlap_n) if self._overlap_n else None

    def informed(self):
        """True once a successor has actually been observed."""
        return self._overlap_n > 0 and any(t for _, t in self._survival.values())

    # ---------------------------------------------------------- resident size
    @staticmethod
    def resident_tokens(cache):
        """Tokens each session holds, shared nodes split across their owners.

        A SWE-bench scaffold head is inserted by every program that shares it,
        so charging its full length to each would make every session look
        larger than the pool. Splitting keeps the sum equal to the tree.
        """
        out = {}
        stack = [cache.root_node]
        while stack:
            node = stack.pop()
            stack.extend(node.children.values())
            if node is cache.root_node or not node.owners or node.lock_ref > 0:
                continue
            share = len(node.key) / len(node.owners)
            for owner in node.owners:
                out[owner] = out.get(owner, 0.0) + share
        return out

    # ----------------------------------------------------------------- order
    def ranked_sessions(self, memory, now_ns):
        """Sessions worst-to-keep first, or None when nothing is observed."""
        if not self.informed():
            return None
        cache = memory.npu_prefix_cache
        sizes = self.resident_tokens(cache)
        now_s = now_ns / NS
        overlap = self.overlap()
        entries = []
        for session, tokens in sizes.items():
            if tokens <= 0:
                continue
            last = self.last_access_ns.get(session)
            if last is None:
                continue
            p = self.reuse_probability(session)
            successors = () if p is None else (Successor(min(1.0, p), overlap),)
            entries.append(CacheObservation(
                session=str(session),
                last_access_s=min(now_s, last / NS),
                size_bytes=max(1, int(round(tokens * cache.kv_size))),
                successors=successors))
        if not entries:
            return None
        idle = [now_s - e.last_access_s for e in entries]
        ranked = eviction_order(entries, now_s, max(idle) if idle else 0.0)
        self.stats["ranked"] += 1
        return tuple(e.session for e in ranked)

    def node_key(self, memory, now_ns):
        """A radix eviction key: SAGA's rank, then the tree's own LRU.

        Nodes whose owners are unknown to the estimator sort after the ranked
        sessions rather than ahead of them, so an unattributed node is not
        reclaimed in preference to one SAGA actually scored.
        """
        order = self.ranked_sessions(memory, now_ns)
        if order is None:
            self.stats["lru_fallback"] += 1
            return None
        rank = {s: i for i, s in enumerate(order)}
        unknown = len(rank)

        def key(node):
            owners = [rank.get(str(o), unknown) for o in node.owners] or [unknown]
            # A shared node is only as evictable as its best-kept owner.
            return (min(owners), node.last_access_time)

        return key
