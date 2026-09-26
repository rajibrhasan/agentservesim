"""Service safe control messages on the engine thread during worker execution.

The response reader only resolves the executor future. All scheduler and KV
operations stay on the engine owner thread. Never overtake an ADD, ABORT, or
unrecognized utility: its ordering may be required by a later retention call.
"""
from concurrent.futures import ThreadPoolExecutor, wait
from queue import Empty


class EngineControlWait:
    SAFE_METHODS = frozenset({'kv_protect', 'kv_protection_stats'})

    def __init__(self):
        self.reader = ThreadPoolExecutor(max_workers=1,
                                         thread_name_prefix='model-response')
        self.serviced = 0

    def result(self, future, engine, utility_type):
        response = self.reader.submit(future.result)
        while not response.done():
            # There is only one engine input consumer. Inspect the head under
            # Queue's mutex, but dispatch outside it (handlers may enqueue).
            queue = engine.input_queue
            with queue.mutex:
                head = queue.queue[0] if queue.queue else None
                eligible = (head is not None and head[0] == utility_type
                            and head[1][2] in self.SAFE_METHODS)
            if eligible:
                try:
                    request = queue.get_nowait()
                except Empty:
                    continue
                engine._handle_client_request(*request)
                self.serviced += 1
            else:
                # Wait without spinning; future.result(timeout) is unsupported
                # by vLLM's lazy FutureWrapper, hence the separate reader.
                wait([response], timeout=0.001)
        return response.result()

    def close(self):
        self.reader.shutdown(wait=True)
