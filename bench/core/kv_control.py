"""Transports to a LIVE vLLM engine's KV protection utilities.

The engine (v0.19.0, branch agent-knobs) exposes kv_protect, kv_release,
kv_evict and kv_protection_stats through the generic EngineCore dispatch. These
two classes reach them -- in-process for an offline `LLM` object, over HTTP for
the routes `bench/serve_agent.py` adds to a running server.

Here rather than under `policies/` because a transport is not a policy: it
decides nothing, and a package of published decision rules should not carry the
code that opens a socket. The interface they implement lives with the policies
(`policies/utils/kv_control.py`), because that is what the executors write
against.
"""
import json
import urllib.request
from typing import Any, Optional

from policies.utils.kv_control import KVControl


class InProcessKVControl(KVControl):
    def __init__(self, llm) -> None:
        self._core = llm.llm_engine.engine_core

    def protect(self, request_id: str, deadline_ts: float) -> int:
        return self._core.call_utility("kv_protect", request_id, deadline_ts)

    def release(self, request_id: str) -> int:
        return self._core.call_utility("kv_release", request_id)

    def evict(self, request_id: str) -> int:
        return self._core.call_utility("kv_evict", request_id)

    def stats(self) -> dict[str, Any]:
        return self._core.call_utility("kv_protection_stats")


class HTTPKVControl(KVControl):
    def __init__(self, base_url: str, timeout_s: float = 10.0) -> None:
        self._base = base_url.rstrip("/")
        self._timeout = timeout_s

    def _post(self, path: str, payload: dict[str, Any]) -> Any:
        req = urllib.request.Request(
            self._base + path,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self._timeout) as resp:
            return json.loads(resp.read())

    def protect(self, request_id: str, deadline_ts: float) -> int:
        return self._post(
            "/kv/protect", {"request_id": request_id, "deadline_ts": deadline_ts}
        )["parked"]

    def release(self, request_id: str) -> int:
        return self._post("/kv/release", {"request_id": request_id})["released"]

    def evict(self, request_id: str) -> int:
        return self._post("/kv/evict", {"request_id": request_id})["evicted"]

    def stats(self) -> dict[str, Any]:
        with urllib.request.urlopen(
            self._base + "/kv/stats", timeout=self._timeout
        ) as resp:
            return json.loads(resp.read())
