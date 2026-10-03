
from typing import Any, Optional


class KVControl:
    """Interface: all methods refer to a completed request's blocks."""

    def protect(self, request_id: str, deadline_ts: float) -> int:
        raise NotImplementedError

    def release(self, request_id: str) -> int:
        raise NotImplementedError

    def evict(self, request_id: str) -> int:
        raise NotImplementedError

    def stats(self) -> dict[str, Any]:
        raise NotImplementedError


class RecordingKVControl(KVControl):
    def __init__(self, blocks_per_request: int = 4) -> None:
        self.calls: list[tuple] = []
        self.blocks_per_request = blocks_per_request

    def protect(self, request_id: str, deadline_ts: float) -> int:
        self.calls.append(("protect", request_id, deadline_ts))
        return self.blocks_per_request

    def release(self, request_id: str) -> int:
        self.calls.append(("release", request_id))
        return self.blocks_per_request

    def evict(self, request_id: str) -> int:
        self.calls.append(("evict", request_id))
        return self.blocks_per_request

    def stats(self) -> dict[str, Any]:
        self.calls.append(("stats",))
        return {}
