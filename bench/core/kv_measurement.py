"""Per-turn input KV reuse, separate from prefix-cache lookup hits."""


def merged(intervals):
    out = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if out and start <= out[-1][1]:
            out[-1] = (out[-1][0], max(end, out[-1][1]))
        else:
            out.append((start, end))
    return out


class TurnKV:
    def __init__(self, program, turn, prompt, history=0):
        self.program, self.turn, self.prompt = program, turn, prompt
        self.history = history
        self.computed, self.restored, self.prefix = [], [], []
        self.compute_total = self.recompute_total = 0
        self.complete = False

    def compute(self, start, end):
        start, end = max(0, start), min(self.prompt, end)
        if end > start:
            self.compute_total += end - start
            self.recompute_total += max(0, min(end, self.history) - start)
            self.computed = merged(self.computed + [(start, end)])

    def result(self):
        computed = merged(self.computed)
        restored = merged((max(0, a), min(b, self.prompt)) for a, b in self.restored)
        cpu = sum(b-a for a, b in restored)
        cpu -= sum(max(0, min(b, d)-max(a, c))
                   for a, b in restored for c, d in computed)
        reused = self.prompt - sum(b-a for a, b in computed)
        prefix = merged((max(0, a), min(b, self.prompt)) for a, b in self.prefix)
        return dict(program_id=self.program, turn_idx=self.turn,
                    input_toks=self.prompt, complete=self.complete,
                    kv_reused_tokens=reused, gpu_reused_tokens=reused-cpu,
                    cpu_restored_tokens=cpu, prefill_computed_tokens=self.compute_total,
                    recomputed_context_tokens=self.recompute_total,
                    cached_tokens=sum(b-a for a, b in prefix),
                    cache_measurement='input_kv_reuse')


def attach_turn_measurements(records, engine_metrics):
    measured = {}
    for engine in engine_metrics:
        for row in engine['turn_kv_measurements']:
            key = (row['program_id'], row['turn_idx'])
            if key in measured:
                raise ValueError(f'Duplicate KV measurement: {key}')
            measured[key] = row
    for record in records:
        key = (record['program_id'], record['turn_idx'])
        row = measured.pop(key)
        if not row['complete'] or row['input_toks'] != record['input_toks']:
            raise ValueError(f'KV measurement does not match completed turn: {key}')
        record.update({k: v for k, v in row.items()
                       if k not in ('program_id', 'turn_idx', 'input_toks', 'complete')})
    if measured:
        raise ValueError('Unmatched engine KV measurements')
