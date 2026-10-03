# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lifetime of Continuum's temporary waiting-queue protection.

An arriving successor suspends expiry until scheduling. If it is cancelled
before scheduling, the old pin resumes its original deadline, rather than
remaining held forever or losing the rest of its valid TTL. This records no
new physical references and cannot resurrect a reclaimed block.

Installed into vllm/v1/core/ by experiments/validation/prepare_policy_engine.py;
the copy under experiments/validation/engine_files/ is the source of truth.
"""

from dataclasses import dataclass, field


@dataclass
class HeldBlock:
    block_hash: object
    deadline: float
    tags: set = field(default_factory=set)


class QueueHolds:
    def __init__(self):
        self.counts = {}
        self.blocks = {}
        self.by_tag = {}

    def capture(self, manager, tag):
        self.counts[tag] = self.counts.get(tag, 0) + 1
        pool = manager.block_pool
        ids = self.by_tag.setdefault(tag, set())
        for bid in manager._protected_requests.get(tag, ()):
            deadline = pool._protected.get(bid)
            if deadline is None:
                continue
            block_hash = pool.blocks[bid].block_hash
            previous = self.blocks.get(bid)
            if previous is None or previous.block_hash != block_hash:
                previous = HeldBlock(block_hash, deadline)
                self.blocks[bid] = previous
            previous.tags.add(tag)
            ids.add(bid)

    def finish(self, manager, tag, cancelled=False):
        count = self.counts.get(tag, 0)
        if not count:
            return
        if cancelled and count > 1:
            self.counts[tag] = count - 1
            return
        self.counts.pop(tag)
        pool = manager.block_pool
        for bid in self.by_tag.pop(tag, ()):
            record = self.blocks.get(bid)
            if record is None:
                continue
            record.tags.discard(tag)
            if record.tags:
                continue
            del self.blocks[bid]
            if (
                cancelled
                and pool.blocks[bid].block_hash == record.block_hash
                and pool._protected.get(bid) == manager.KV_HOLD_DEADLINE
            ):
                pool.set_protection_deadline([bid], record.deadline)
