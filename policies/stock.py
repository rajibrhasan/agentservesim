"""vLLM v1 as shipped. The baseline every entrant must run.

No pinning, no stamping, no placement rule: the engine's own LRU, FCFS and
least-loaded. Classes rather than `None` so that "ran the baseline" and "no
policy was configured" are different states in the log.
"""
from .base import KVPolicy, RoutingPolicy, SchedulingPolicy

PAPER = "vLLM v1 baseline"


class StockKV(KVPolicy):
    """Stock vLLM: prefix caching on, blocks compete in the LRU free
    queue. No runtime calls. This is the default tuple's retention."""

    engine_flags = {"enable_prefix_caching": True, "kv_protection": False}


class NoCacheKV(KVPolicy):
    """Realized by launching the engine with prefix caching off; every
    turn re-prefills. No runtime calls, no protection mechanism."""

    engine_flags = {"enable_prefix_caching": False, "kv_protection": False}


class StockScheduling(SchedulingPolicy):
    """Stock vLLM FCFS queue; nothing stamped. The default tuple's
    scheduling value."""

    engine_args = {}


class RoundRobinRouting(RoutingPolicy):
    """Cyclic placement, load-blind. The default tuple's routing."""

    @classmethod
    def from_config(cls, cfg):
        return cls(cfg.num_instances)

    def __init__(self, num_instances: int) -> None:
        super().__init__(num_instances)
        self._next = 0

    def route(self, pcb, now):
        instance = self._next
        self._next = (self._next + 1) % self.num_instances
        return instance, None


class LeastLoadedRouting(RoutingPolicy):
    """Fewest in-flight turns; lowest index on ties."""

    @classmethod
    def from_config(cls, cfg):
        return cls(cfg.num_instances)

    def route(self, pcb, now):
        return self._least_loaded(), None
