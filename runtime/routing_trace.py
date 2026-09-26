"""Optional observations actually used by each SAGA placement decision."""
import os
from .schedule_trace import ScheduleTrace

_traces = {}


def record_saga_route(plane, program, now, observed_at, home, workers, cached, destination):
    directory = os.environ.get('SAGA_ROUTING_TRACE_DIR')
    if not directory:
        return
    key = (directory, plane)
    if key not in _traces:
        _traces[key] = ScheduleTrace(directory, 'saga-routing-' + plane)
    _traces[key].write(event='route', plane=plane, program=str(program),
        ts=now, observed_at=observed_at, previous_home=home,
        cached_workers=sorted(cached), destination=destination,
        workers=[dict(instance=w.worker, load=w.load,
                      queued_sessions=list(w.queued_sessions),
                      empty_since=w.empty_since_s) for w in workers])
