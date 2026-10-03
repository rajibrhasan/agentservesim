
import math
from statistics import NormalDist
from typing import Optional

from .base import KVPolicy, RoutingPolicy

PAPER = "SAGA 2605.00528"


class SagaKV(KVPolicy):
    """SAGA-style (arXiv:2605.00528, Alg. 1) TTL that shrinks under memory
    pressure: tau_eff = tau * (1 - 0.5 * m), with m the KV utilisation
    normalised between a soft threshold (0.7) and a hard threshold (0.9),
    clipped to [0, 1]. Requires the kv_utilization signal; without it the
    policy degrades to plain TTL."""

    @classmethod
    def from_config(cls, cfg):
        return cls(cfg.tau_s)

    def __init__(self, tau_s: float, soft: float = 0.7, hard: float = 0.9) -> None:
        self.tau_s = tau_s
        self.soft = soft
        self.hard = hard

    def on_turn_complete(self, pcb, request_id, now):
        u = self.signals.kv_utilization
        m = 0.0 if u is None else min(1.0, max(0.0, (u - self.soft) / (self.hard - self.soft)))
        tau = self.tau_s * (1.0 - 0.5 * m)
        return ("protect", now + tau, {"tau": tau, "kv_utilization": u})

    def on_turn_arrival(self, pcb, now):
        return "release"


class SagaToolTTL(SagaKV):
    """Algorithm 1's learned tool TTL, independently selectable from saga-ttl.

    Only completed tool gaps train the per-tool log-normal distribution.
    Cold-start duration and EMA weight are explicit implementation choices:
    the paper does not supply them. This component does not implement SAGA's
    workflow eviction, fairness or migration mechanisms.
    """

    @classmethod
    def from_config(cls, cfg):
        return cls(cold_start_s=cfg.tau_s)

    def __init__(self, cold_start_s, ema_weight=0.2, percentile=0.95,
                 ttl_max_s=300.0, soft=0.7, hard=0.9):
        if cold_start_s is None:
            raise ValueError('saga-tool-ttl needs --retention-tau for cold start')
        if not math.isfinite(cold_start_s) or cold_start_s <= 0:
            raise ValueError('cold-start TTL must be positive and finite')
        if not 0 < ema_weight <= 1 or not 0 < percentile < 1:
            raise ValueError('invalid EMA weight or percentile')
        if not math.isfinite(ttl_max_s) or ttl_max_s <= 0 or not 0 <= soft < hard <= 1:
            raise ValueError('invalid TTL cap or pressure thresholds')
        super().__init__(cold_start_s, soft, hard)
        self.ema_weight = ema_weight
        self.percentile = percentile
        self.ttl_max_s = ttl_max_s
        self._z = NormalDist().inv_cdf(percentile)
        self._history = {}
        self._last_observed = {}

    def observe_arrival(self, pcb, now):
        if pcb.tool_name is None or pcb.gap_started_ts is None:
            return
        duration = (pcb.completed_tool_duration_s
                    if pcb.completed_tool_duration_s is not None
                    else now - pcb.gap_started_ts)
        if not math.isfinite(duration) or duration < 0:
            raise ValueError('tool observation must follow gap start')
        key = (pcb.turn_idx, pcb.gap_started_ts, pcb.tool_name)
        if self._last_observed.get(pcb.program_id) == key:
            return
        self._last_observed[pcb.program_id] = key
        # An immediate continuation supplies no positive log-normal sample.
        if duration == 0:
            return
        value = math.log(duration)
        old = self._history.get(pcb.tool_name)
        if old is None:
            self._history[pcb.tool_name] = (value, 0.0, 1)
        else:
            mean, variance, n = old
            delta = value - mean
            weight = self.ema_weight
            self._history[pcb.tool_name] = (
                mean + weight * delta,
                (1 - weight) * (variance + weight * delta * delta), n + 1)

    def predicted_gap_s(self, tool_name):
        """Learned p-th percentile duration of a tool, or the cold-start window.

        Read by SAGA's prefetcher to time a recompute before the tool returns.
        """
        history = self._history.get(tool_name) if tool_name is not None else None
        if history is None:
            return self.tau_s
        mean, variance, _ = history
        return math.exp(mean + self._z * math.sqrt(max(0.0, variance)))

    def on_turn_complete(self, pcb, request_id, now):
        if pcb.tool_name is None:
            return None
        history = self._history.get(pcb.tool_name)
        u = self.signals.kv_utilization
        if u is None or not math.isfinite(u) or not 0 <= u <= 1:
            raise ValueError('saga-tool-ttl requires measured KV utilization')
        pressure = min(1.0, max(0.0, (u - self.soft) / (self.hard - self.soft)))
        scale = 1 - 0.5 * pressure
        if history is None:
            log_base = math.log(self.tau_s)
            n = 0
        else:
            mean, variance, n = history
            log_base = mean + self._z * math.sqrt(max(0.0, variance))
        # Apply the cap after pressure scaling, without overflowing exp().
        ttl = math.exp(min(math.log(self.ttl_max_s), log_base + math.log(scale)))
        return ('protect', now + ttl, {
            'tau': ttl, 'tool_name': pcb.tool_name, 'samples': n,
            'percentile': self.percentile, 'ema_weight': self.ema_weight,
            'cold_start': history is None, 'kv_utilization': u})


class SagaRouting(RoutingPolicy):
    """Follow the instance that already holds the program's context
    (pcb.kv_instance), least-loaded on first contact, capacity fallback (re-pin)
    when that instance is at or above capacity_limit in-flight turns."""

    @classmethod
    def from_config(cls, cfg):
        return cls(cfg.num_instances, capacity_limit=cfg.capacity_limit)

    def __init__(
        self, num_instances: int, capacity_limit: Optional[int] = None
    ) -> None:
        super().__init__(num_instances)
        self.capacity_limit = capacity_limit

    def route(self, pcb, now):
        pinned = pcb.kv_instance
        if pinned is None:
            return self._least_loaded(), {"pin": "new"}
        if (
            self.capacity_limit is not None
            and self.inflight[pinned] >= self.capacity_limit
        ):
            instance = self._least_loaded()
            return instance, {"pin": "fallback", "fallback": True, "from": pinned}
        return pinned, None
