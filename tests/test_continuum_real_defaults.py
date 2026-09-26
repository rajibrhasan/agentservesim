"""Real policy construction and provisional pin rollback without a GPU."""
import argparse
import asyncio
from unittest.mock import Mock

import pytest

from bench.core import policy_driver as driver_mod
from bench.core.runner import register_args
from policies.continuum import ContinuumKV
from policies.program import ProgramTable


@pytest.mark.parametrize("explicit,expected", [(None, 2.0), (2.0, 2.0), (60.0, 60.0)])
def test_cli_to_policy_window(explicit, expected):
    parser = argparse.ArgumentParser()
    register_args(parser)
    args = ["--model", "m", "--dataset", "d", "--output-dir", "o",
            "--retention", "continuum", "--scheduling", "continuum"]
    if explicit is not None:
        args += ["--retention-tau", str(explicit)]
    parsed = parser.parse_args(args)

    async def check():
        driver = driver_mod.PolicyDriver(
            driver_mod.TupleConfig(retention=parsed.retention,
                                   scheduling=parsed.scheduling,
                                   tau_s=parsed.retention_tau),
            [], asyncio.get_running_loop())
        try:
            policy = driver.retention_exec.policy
            assert policy.pin_s == expected
            assert policy.threshold_s == expected
        finally:
            driver._worker.shutdown(wait=True)
    asyncio.run(check())


def test_rejected_pin_is_cancelled_immediately_on_serving_instance(monkeypatch):
    monkeypatch.setenv("VLLM_KV_PIN_AT_FREE_TTL", "2")
    monkeypatch.setenv("VLLM_KV_RELEASE_AT_ARRIVAL", "1")
    calls = []

    async def check():
        driver = driver_mod.PolicyDriver(
            driver_mod.TupleConfig(retention="continuum", scheduling="continuum"),
            [], asyncio.get_running_loop())
        try:
            controls = []
            for instance in range(2):
                control = driver_mod.AsyncEngineKVControl(
                    Mock(), asyncio.get_running_loop(), 0)
                control._call = lambda method, *args, i=instance: (
                    calls.append((i, method, args)) or 4)
                controls.append(control)
            driver.kv._controls = controls
            driver._kv_utilization = lambda instance: 0.5
            table = driver.programs
            table.on_turn_release("p", 0, 0)
            table.on_turn_complete("p", 0, now=1, tool_name="slow")
            table.on_turn_release("p", 1, 11)
            driver._complete_sync("p", 1, "p:1", 1, 32, 1, 12, "slow")
            assert driver.retention_exec.decisions[-1].action == "none"
            assert calls == [(1, "kv_release", ("p:1",))]
            assert "p" not in driver_mod.DEFERRED_RELEASES
        finally:
            driver._worker.shutdown(wait=True)
    asyncio.run(check())


def test_no_tool_does_not_pin():
    table = ProgramTable()
    pcb = table.on_turn_complete("p", 0, now=1, tool_name=None)
    assert ContinuumKV().on_turn_complete(pcb, "p:0", 1) is None
