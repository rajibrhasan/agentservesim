"""Optional diagnostic JSONL; never enable for uninstrumented timing results."""
import gzip
import json
import os
from pathlib import Path


class ScheduleTrace:
    def __init__(self, directory, name):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f'{name}-{os.getpid()}.jsonl'
        self.compressed = os.environ.get('SCHEDULE_TRACE_GZIP') == '1'
        self.stream = (gzip.open(str(path) + '.gz', 'at', compresslevel=1)
                       if self.compressed else path.open('a', buffering=1))
        self.step = 0

    def write(self, **record):
        self.stream.write(json.dumps(dict(schema=1, step=self.step, **record)) + '\n')
        if self.compressed:
            self.stream.flush()
        self.step += 1


def simulator_state(scheduler):
    memory = scheduler.memory
    cache = memory.npu_prefix_cache
    return dict(requests=[dict(id=r.id, program=r.session_id, turn=r.sub_request_index,
                              arrival_ns=r.arrival, admitted=r.admit_seq,
                              computed=r.num_computed_tokens, cached=r.npu_cache_hit,
                              preemptions=r.n_preempted, reserved_bytes=r.kv_reserved,
                              prompt_tokens=r.submitted_input, recompute_end=r.original_input,
                              first_token_counted=r.first_token_counted)
                          for r in scheduler.request],
                kv_charged_bytes=memory.npu_used - memory.weight,
                kv_reserved_bytes=memory.npu_reserved,
                kv_capacity_bytes=memory.npu_mem - memory.weight,
                evictable_tokens=cache.evictable_size() if cache is not None else None,
                locked_tokens=cache.protected_size() if cache is not None else None)


_execution_traces = {}


def trace_execution(directory, instance, batch, rows):
    """Record the actual emitted compute charge without retaining large graphs."""
    key = (directory, instance)
    if key not in _execution_traces:
        _execution_traces[key] = ScheduleTrace(directory, f'execution-{instance}')
    compute = sum(int(float(row[1])) for row in rows if len(row) == 11)
    transfer_ns = sum(int(float(row[1])) for row in rows
                      if len(row) == 11 and row[0] == 'host_link_transfer')
    _execution_traces[key].write(plane='sim', event='emitted_execution',
        batch_id=batch.batch_id, time_ns=batch.batch_time,
        total_tokens=batch.total_len, prefill_queries=batch.prefill_q_list,
        prefill_contexts=batch.prefill_k_list, decode_contexts=batch.decode_k_list,
        emitted_compute_ns=compute, model_compute_ns=compute-transfer_ns,
        host_transfer_ns=transfer_ns, host_load_bytes=batch.load,
        host_store_bytes=batch.host_store_bytes, host_link_bytes_s=batch.host_link_bytes_s,
        requests=[dict(id=r.id, program=r.session_id, turn=r.sub_request_index)
                  for r in batch.requests])
