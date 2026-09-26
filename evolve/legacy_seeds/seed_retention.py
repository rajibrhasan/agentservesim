"""Retention policy candidate for the automated policy search.

This file is the unit the search mutates. Only the region between the
EVOLVE-BLOCK markers changes; everything outside it is the fixed
contract. The class is loaded by the simulator as ``--retention evolved``
and drives the Retention Plane: at the end of every turn it decides what
happens to the program's KV cache across the tool gap that follows.

Contract (policies/base.py, RetentionPolicy):

  on_turn_complete(pcb, request_id, now) -> None | (action, deadline_ts)
                                          | (action, deadline_ts, info)
      Called when a turn's response finishes and the program enters a
      tool gap. Actions:
        "protect": keep this program's KV resident until deadline_ts
                   (seconds, absolute). Past the deadline the blocks stay
                   hit-able but are reclaimed first under memory pressure.
        "evict":   drop the KV now; the next turn re-prefills its context.
        None:      no action; the blocks compete in the ordinary LRU
                   free queue (stock vLLM behaviour).
      info, when given, must be a JSON-serializable dict of the inputs
      behind the decision; it is logged, never acted on.

  on_turn_arrival(pcb, now) -> "release" | None
      Called when the program's next turn arrives. "release" ends a
      standing protection (the blocks are about to be used anyway).

Observation boundary: the ONLY inputs are ``now`` (seconds) and the
Program Control Block fields below. The gap's actual duration is trace
knowledge and is never available; a policy must predict it from what it
has seen. Reading pcb.program_id or pcb.kv_request_id is forbidden, and
keeping any per-program table inside the policy is forbidden: program
state lives in the PCB, the policy holds only aggregate state.

  pcb.turn_idx            index of the turn that just completed (0-based)
  pcb.turns_completed     turns this program has completed so far
  pcb.arrival_ts          first arrival of the program (seconds)
  pcb.attained_service_s  total service time the program has received
  pcb.context_tokens      tokens of context the KV would keep resident
  pcb.tool_name           name of the tool the program is about to call
                          (None when unknown)
  pcb.in_gap, pcb.gap_started_ts, pcb.gap_elapsed_s(now)
                          tool-gap state (in on_turn_arrival, elapsed is
                          the gap that just ended)
  pcb.kv_protected, pcb.kv_deadline_ts, pcb.kv_instance
                          current KV residency state

Fitness: mean program JCT of the stock configuration divided by mean
program JCT under this policy, over several arrival rates and KV
budgets. 1.0 is parity with stock; higher is better. Protecting too
much starves other programs of KV and forces reclaims; protecting too
little re-prefills long contexts after every tool call.
"""

from harness.retention import RetentionPolicy


# System signal available at decision time (set by the executor before
# each call): ``self.signals.kv_utilization`` is the fraction of the KV
# pool in use (0..1) on the instance that served the turn, or None when
# the host did not provide it. Pressure-aware retention (e.g. shrinking
# the protection window as the pool fills) should read it from there.
# EVOLVE-BLOCK-START
class EvolvedRetention(RetentionPolicy):
    """Seed: fixed-horizon protection (the published TTL policy).

    Protect for TAU seconds after every turn, release on the next
    arrival. The horizon is a constant; it does not depend on the tool,
    the context size, or what the policy has observed so far.
    """

    TAU_S = 2.0

    def __init__(self):
        # Aggregate state only (no per-program tables): running
        # statistics of observed gaps by tool name are allowed here.
        self.gap_ema_by_tool = {}

    def on_turn_complete(self, pcb, request_id, now):
        return ("protect", now + self.TAU_S)

    def on_turn_arrival(self, pcb, now):
        gap = pcb.gap_elapsed_s(now)
        if gap is not None and pcb.tool_name is not None:
            prev = self.gap_ema_by_tool.get(pcb.tool_name)
            self.gap_ema_by_tool[pcb.tool_name] = (
                gap if prev is None else 0.8 * prev + 0.2 * gap)
        return "release"
# EVOLVE-BLOCK-END
