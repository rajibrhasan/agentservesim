# SPDX-License-Identifier: Apache-2.0
"""Scheduler-side admission gate for agent programs (AgentServeSim port).

Mirrors LLMServingSim's ``UnifiedPolicyAdapter.filter_waiting``: at each
scheduling step every WAITING (not preempted) request is shown a QueueView
of the KV pool and the harness scheduling policy's ``admit()`` decides
whether the request may be scheduled this step. A held request keeps its
queue place (it is re-queued through the skipped-waiting path) and is
re-asked next step. When nothing is running, nothing has been scheduled
this step, and no waiting request passes the gate, the queue head is
admitted regardless (the simulator's starvation guard, same rule: it fires
only when the whole queue would otherwise be held).

Enabled by ``VLLM_ADMISSION_GATE=<harness root>``: the directory holding
``harness/`` with an ``evolved_scheduling.py`` drop-in (same scratch copy
the gateway uses). Optional:
  ``VLLM_ADMISSION_LOG``   jsonl path for per-hold records
  ``GATE_UTIL_SEMANTICS``  ``sim`` (default: cached-evictable blocks count
                           as used, LLMServingSim's npu_used) or ``vllm``
                           (kv_cache_usage: evictable blocks are free)
"""

from __future__ import annotations

import json
import os
import sys
import time
from types import SimpleNamespace
from typing import TYPE_CHECKING

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.request import Request

logger = init_logger(__name__)


def build_admission_gate(kv_cache_manager: "KVCacheManager",
                         block_size: int) -> "AdmissionGate | None":
    root = os.environ.get("VLLM_ADMISSION_GATE")
    if not root:
        return None
    if root not in sys.path:
        sys.path.insert(0, root)
    from harness.scheduling import QueueView  # noqa: F401
    from harness.evolved_scheduling import EvolvedScheduling
    policy = EvolvedScheduling()
    gate = AdmissionGate(policy, kv_cache_manager, block_size,
                         util_semantics=os.environ.get("GATE_UTIL_SEMANTICS", "sim"),
                         cached_mode=os.environ.get("GATE_CACHED_TOKENS", "live"),
                         log_path=os.environ.get("VLLM_ADMISSION_LOG"))
    logger.info("Admission gate enabled: %s.%s from %s (util=%s, cached=%s)",
                type(policy).__module__, type(policy).__name__, root,
                gate.util_semantics, gate.cached_mode)
    return gate


class AdmissionGate:

    def __init__(self, policy, kv_cache_manager, block_size, util_semantics="sim",
                 cached_mode="live", log_path=None):
        from harness.scheduling import QueueView
        self._QueueView = QueueView
        self.policy = policy
        self.kvm = kv_cache_manager
        self.block_size = int(block_size)
        self.util_semantics = util_semantics
        # GATE_CACHED_TOKENS: "live" = real prefix hit (the policy as
        # written); "zero" = the degenerate variant the simulator scored
        # before 2026-09-06 (its gate saw cached_tokens=0 for held turns).
        self.cached_mode = cached_mode
        self.stats = {"admission_holds": 0, "admission_admits": 0,
                      "admission_guard_admits": 0}
        self._log = open(log_path, "a", buffering=1) if log_path else None  # line-buffered
        self._snap = None
        self._first_in_step = False

    # -- per-step snapshot of the pool. The sim evaluates every waiting turn
    #    of a tick against one memory view (filter_waiting reads free /
    #    evictable / utilization once); the same here: free-queue size and
    #    the cached-in-free-queue count are taken at step start, so a turn
    #    admitted earlier in the same step does not shrink what later turns
    #    see. The prefix hit stays per request (the sim probes per request).
    def begin_step(self) -> None:
        st = dict(self.kvm.block_pool.get_protection_stats())
        st["free_q"] = int(self.kvm.block_pool.free_block_queue.num_free_blocks)
        self._snap = st
        self._first_in_step = True

    def _view(self, request: "Request", n_running: int, n_waiting: int):
        pool = self.kvm.block_pool
        st = self._snap
        if st is None:
            st = dict(pool.get_protection_stats())
            st["free_q"] = int(pool.free_block_queue.num_free_blocks)
        total = int(st.get("num_gpu_blocks", 0)) - 1  # null block
        free_q = int(st["free_q"])
        cached_free = min(int(st.get("cached_free_blocks", 0)), free_q)
        free_uncached = free_q - cached_free
        util = None
        if total > 0:
            used = total - free_q if self.util_semantics == "vllm" else total - free_uncached
            util = max(0.0, min(1.0, used / total))
        if self.cached_mode == "zero":
            cached_tokens = 0
        else:
            _, cached_tokens = self.kvm.get_computed_blocks(request)
        return self._QueueView(
            n_running=n_running, n_waiting=n_waiting, n_inflight=n_running,
            kv_utilization=util,
            kv_free_tokens=free_uncached * self.block_size,
            kv_evictable_tokens=cached_free * self.block_size,
            prompt_tokens=int(request.num_prompt_tokens),
            cached_tokens=int(cached_tokens))

    def _pcb(self, request: "Request"):
        tag = self.kvm._kv_retention_key(request)
        return SimpleNamespace(
            program_id=tag, arrival_ts=float(request.arrival_time),
            context_tokens=int(request.num_prompt_tokens),
            kv_protected=tag in self.kvm._protected_requests, turn_idx=None)

    def select_victim(self, candidates):
        """Apply the Gate victim rule to native running requests."""
        if not candidates:
            return None
        views = [SimpleNamespace(
            pcb=self._pcb(r), priority=r.priority,
            prompt_tokens=r.num_prompt_tokens,
            generated_tokens=max(0, r.num_computed_tokens - r.num_prompt_tokens),
            computed_tokens=r.num_computed_tokens,
            is_prefill=r.num_computed_tokens < r.num_prompt_tokens)
            for r in candidates]
        index = self.policy.victim(views, time.time())
        if index is None:
            return None
        if not isinstance(index, int) or not 0 <= index < len(candidates):
            raise ValueError("Admission policy returned an invalid victim index")
        self.stats["victim_overrides"] = self.stats.get("victim_overrides", 0) + 1
        return candidates[index]

    def _passes(self, request, n_running, n_waiting, now) -> bool:
        if request.num_computed_tokens > 0:
            return True
        return bool(self.policy.admit(self._pcb(request), now,
                                      self._view(request, n_running, n_waiting)))

    def admit(self, request: "Request", n_running: int, n_waiting: int,
              idle: bool, others=()) -> bool:
        """True = let the scheduler try to fit `request` this step.
        `others`: the remaining waiting requests (for the idle guard)."""
        if request.num_computed_tokens > 0:
            return True  # resumed after preemption / async KV: never gated
        head = self._first_in_step
        self._first_in_step = False
        view = self._view(request, n_running, n_waiting)
        pcb = self._pcb(request)
        now = time.time()
        ok = bool(self.policy.admit(pcb, now, view))
        if not ok and idle and head:
            # Starvation guard (sim: `if not kept and not running and
            # n_inflight == 0: kept = [waiting[0]]`): admit the queue head
            # only when no other waiting request would pass this step.
            if not any(self._passes(r, n_running, n_waiting, now)
                       for r in others if r is not request):
                self.stats["admission_guard_admits"] += 1
                return True
        if ok:
            self.stats["admission_admits"] += 1
        else:
            self.stats["admission_holds"] += 1
            if self._log is not None:
                self._log.write(json.dumps({
                    "event": "hold", "ts": now, "program_id": pcb.program_id,
                    "request_id": request.request_id,
                    "kv_utilization": view.kv_utilization,
                    "kv_free_tokens": view.kv_free_tokens,
                    "kv_evictable_tokens": view.kv_evictable_tokens,
                    "prompt_tokens": view.prompt_tokens,
                    "cached_tokens": view.cached_tokens,
                    "n_running": n_running, "n_waiting": n_waiting}) + "\n")
        return ok
