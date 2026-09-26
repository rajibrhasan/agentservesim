"""Require a complete scheduling sweep before declaring global idle."""


class IdleScheduleSweep:
    def __init__(self):
        self.failed = set()

    def reset(self):
        # Completion, dispatch or an outstanding batch invalidates older probes.
        self.failed.clear()

    def record_failure(self, instance_id, pending_instances):
        self.failed.add(instance_id)
        return set(pending_instances).issubset(self.failed)
