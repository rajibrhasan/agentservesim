"""Native Gate victim selection; run with the patched vLLM environment."""
from types import SimpleNamespace as NS

from policies.gate import EvolvedScheduling
from vllm.v1.core.sched.admission_gate import AdmissionGate


def test_gate_selects_cheapest_context_instead_of_lowest_priority():
    gate = AdmissionGate.__new__(AdmissionGate)
    gate.policy = EvolvedScheduling()
    gate.stats = {}
    gate.kvm = NS(_kv_retention_key=lambda r: r.request_id,
                  _protected_requests={})
    expensive = NS(request_id='a', arrival_time=1, num_prompt_tokens=1000,
                   num_computed_tokens=1100, priority=999)
    cheap = NS(request_id='b', arrival_time=2, num_prompt_tokens=100,
               num_computed_tokens=120, priority=1)
    assert gate.select_victim([expensive, cheap]) is cheap
    assert gate.stats['victim_overrides'] == 1
    gate.policy = NS(victim=lambda candidates, now: None)
    assert gate.select_victim([expensive, cheap]) is None
    assert gate.select_victim([]) is None
