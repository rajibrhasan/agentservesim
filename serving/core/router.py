import bisect
import json
import os
import random
from .logger import get_logger


#: Generated tokens whose ids the trace does not carry are numbered from here.
#: Above any real vocabulary id, and salted per (request, position), so the
#: blocks are unique to their request -- which is what makes them occupy the
#: pool without being falsely matchable by anyone else.
_SYNTH_ID_BASE = 1 << 40

#: Derive a turn's generated ids from the successor's prompt instead of
#: synthesising them. Off by default -- see derive_output_ids.
_DERIVE_FROM_SUCCESSOR = os.environ.get("SIM_DERIVE_OUTPUT_IDS", "0") != "0"


def derive_output_ids(cur, nxt, n_out, salt):
   
    if cur.get("output_tok_ids"):
        return list(cur["output_tok_ids"])
    if n_out <= 0:
        return []
    if _DERIVE_FROM_SUCCESSOR:
        cur_in = cur.get("input_tok_ids") or []
        nxt_in = (nxt or {}).get("input_tok_ids") or []
        if cur_in and nxt_in and len(nxt_in) > len(cur_in):
            taken = nxt_in[len(cur_in):len(cur_in) + n_out]
            if len(taken) == n_out:
                return list(taken)
    return [_SYNTH_ID_BASE + (salt * 1_000_003) + i for i in range(n_out)]


class Router:
    def __init__(
            self,
            num_instances,
            schedulers, req_num,
            routing_policy="RR",
            seed=42,
            policy_adapter=None
    ):
        # Unified serving policy adapter (unified_policy.py); when set it
        # owns instance selection, priority stamping, and the retention
        # events. None = stock behavior.
        self.policy_adapter = policy_adapter
        self.schedulers = schedulers
        self.num_instances = num_instances
        self.prefill_schedulers = [s for s in schedulers if s.pd_type != "decode"]
        self.prefill_instances = len(self.prefill_schedulers)
        self.decode_schedulers = [s for s in schedulers if s.pd_type == "decode"]
        self.decode_instances = len(self.decode_schedulers)
        self.req_num = req_num
        self.routing_policy = routing_policy.upper()
        self.seed = seed
        self._rnd = random.Random(seed) if seed is not None else random
        self.prefill_rr_counter = 0
        self.decode_rr_counter = 0

        # Pending requests (loaded but not yet routed)
        self._pending_requests = []
        self._pending_idx = 0
        self._enable_prefix_caching = False
        self._is_init = True

        # Agentic session dependency tracking (linear chain — backward compat)
        self._deferred_sessions = {}     # session_id -> session state dict
        self._request_to_session = {}    # request_id -> (session_id, sub_request_index)
        self._next_request_id = 0        # monotonic counter for unique request IDs

        # Multi-agent DAG workflow dependency tracking
        self._workflows = {}             # workflow_id -> workflow state dict
        self._request_to_node = {}       # request_id -> (workflow_id, node_id)
        self._workflow_metrics = []      # completed-workflow records (JCT etc.)

        if self.routing_policy == "RR":
            self._select_instance = self._rr_select
        elif self.routing_policy == "RAND":
            self._select_instance = self._rand_select
        elif self.routing_policy == "LOAD":
            self._select_instance = self._least_load_select
        elif self.routing_policy == "CUSTOM":
            self._select_instance = self._custom_select
        else:
            raise ValueError(f"Unknown routing_policy '{routing_policy}'. "
                             "Supported: RR, RAND, LOAD, CUSTOM")
        self.logger = get_logger(self.__class__)

    # -----------------------------------------------------------------------
    # Instance selection policies
    # -----------------------------------------------------------------------

    def _get_counter(self, role):
        return self.decode_rr_counter if role == "decode" else self.prefill_rr_counter

    def _set_counter(self, role, value):
        if role == "decode":
            self.decode_rr_counter = value
        else:
            self.prefill_rr_counter = value

    def _rr_select(self, schedulers, role):
        num_instances = len(schedulers)
        idx = self._get_counter(role) % num_instances
        self._set_counter(role, idx + 1)
        return idx

    def _rand_select(self, schedulers, role):
        return self._rnd.randrange(len(schedulers))

    def _least_load_select(self, schedulers, role):
        """vLLM-style least-loaded routing, normalized by instance capacity."""
        best_idx = 0
        best_score = float('inf')
        num_instances = len(schedulers)
        start = self._get_counter(role) % num_instances
        for offset in range(num_instances):
            idx = (start + offset) % num_instances
            sched = schedulers[idx]
            waiting = len(sched.request)
            running = sum(len(b.requests) for b in sched.inflight)
            raw_score = waiting * 4 + running
            capacity = getattr(sched, "max_num_seqs", 0)
            score = raw_score
            if capacity not in (0, float('inf')):
                score = raw_score / capacity
            if score < best_score:
                best_score = score
                best_idx = idx
        self._set_counter(role, (best_idx + 1) % num_instances)
        return best_idx

    def _custom_select(self, schedulers, role):
        raise NotImplementedError("Implement custom routing policy.")

    # -----------------------------------------------------------------------
    # Request loading and real-time routing
    # -----------------------------------------------------------------------

    def load_requests(self, path, enable_prefix_caching=False, is_init=True):
        """Load requests from dataset into pending queue (not yet routed).

        Supports two JSONL formats:
        - Flat: {"input_toks", "output_toks", "arrival_time_ns", ...}
        - Agentic session: {"session_id", "arrival_time_ns", "sub_requests": [...]}

        For agentic sessions, only the first sub-request is added to the
        pending queue. Subsequent sub-requests are released dynamically
        via notify_request_completed() when predecessors finish.
        """
        # The simulator chdir's into astra-sim/, so repo-relative dataset paths
        # are reached via '../'. Absolute paths (e.g. on /orange) are used as-is.
        if not os.path.isabs(path):
            path = f'../{path}'
        self._enable_prefix_caching = enable_prefix_caching
        self._is_init = is_init
        loaded_lines = 0

        with open(path) as f:
            for line in f:
                if self.req_num > 0 and loaded_lines >= self.req_num:
                    break
                row = json.loads(line)
                if 'nodes' in row:
                    self._load_dag_workflow(row, enable_prefix_caching)
                elif 'sub_requests' in row:
                    if self.policy_adapter is not None:
                        # SAGA's AFS needs the session's tenant and deadline,
                        # which only the raw row carries.
                        self.policy_adapter.register_session(row)
                    self._load_agentic_session(row, enable_prefix_caching)
                else:
                    self._load_flat_request(row, enable_prefix_caching)
                loaded_lines += 1

        # Sort pending requests by arrival time (agentic first sub-requests
        # may interleave with flat requests)
        self._pending_requests.sort(key=lambda r: r['arrival_time_ns'])

        self.logger.info("Loaded %d requests into pending queue "
                         "(%d agentic sessions, %d DAG workflows deferred)",
                         len(self._pending_requests),
                         len(self._deferred_sessions),
                         len(self._workflows))

    def _load_flat_request(self, row, enable_prefix_caching):
        """Load a single flat request into pending queue."""
        req_id = self._next_request_id
        self._next_request_id += 1
        req_data = {
            'index': req_id,
            'input_toks': int(row['input_toks']),
            'output_toks': int(row['input_toks'] + row['output_toks']),
            'arrival_time_ns': int(row['arrival_time_ns']),
        }
        if enable_prefix_caching:
            req_data['input_hash_ids'] = row.get('input_tok_ids', [])
            req_data['output_hash_ids'] = derive_output_ids(
                row, None, int(row['output_toks']), req_id)
        self._pending_requests.append(req_data)

    def _load_agentic_session(self, row, enable_prefix_caching):
        """Load an agentic session: first sub-request to pending, rest deferred."""
        sub_reqs = row['sub_requests']
        if not sub_reqs:
            return 0
        session_id = row.get('session_id', f'session_{self._next_request_id}')
        base_id = self._next_request_id
        self._next_request_id += len(sub_reqs)
        arrival_ns = int(row['arrival_time_ns'])

        # Store session state for dependency chain
        self._deferred_sessions[session_id] = {
            'sub_requests': sub_reqs,
            'next_index': 1,  # index 0 is being queued now
            'id_base': base_id,
            'arrival_ns': arrival_ns,  # for per-session JCT on completion
        }

        # Queue the first sub-request
        first = sub_reqs[0]
        req_data = {
            'index': base_id,
            'input_toks': int(first['input_toks']),
            'output_toks': int(first['input_toks'] + first['output_toks']),
            'arrival_time_ns': arrival_ns,
            'session_id': session_id,
            'sub_request_index': 0,
        }
        if enable_prefix_caching:
            req_data['input_hash_ids'] = first.get('input_tok_ids', [])
            req_data['output_hash_ids'] = derive_output_ids(
                first, sub_reqs[1] if len(sub_reqs) > 1 else None,
                int(first['output_toks']), base_id)
        self._pending_requests.append(req_data)
        self._request_to_session[base_id] = (session_id, 0)

        return len(sub_reqs)

    def _load_dag_workflow(self, row, enable_prefix_caching):
        """Load a multi-agent DAG workflow.

        Format:
            {"workflow_id": str, "arrival_time_ns": int,
             "nodes": [{"node_id", "input_toks", "output_toks", "model"?,
                        "input_tok_ids"?, "output_tok_ids"?}, ...],
             "edges": [{"src", "dst", "delay_ns"?, "message_bytes"?}, ...]}

        Each node is one agent LLM call. A node is released only when ALL its
        parents have completed; its release time is
        ``max(parent_completion + edge.delay_ns)`` over incoming edges — which
        is the fan-in / barrier wait. In-degree-0 nodes (roots) are queued at
        ``arrival_time_ns``; the rest are deferred and released dynamically in
        ``notify_request_completed``. The linear chain (``sub_requests``) is the
        path-graph special case of this.
        """
        nodes = row['nodes']
        if not nodes:
            return 0
        workflow_id = row.get('workflow_id', f'workflow_{self._next_request_id}')
        arrival_ns = int(row['arrival_time_ns'])
        base_id = self._next_request_id
        self._next_request_id += len(nodes)

        # Assign a unique request index to each node by position; map both ways.
        node_specs = {}          # node_id -> spec dict (toks, model, hashes)
        node_index = {}          # node_id -> request index
        for pos, node in enumerate(nodes):
            node_id = node.get('node_id', pos)
            node_specs[node_id] = node
            node_index[node_id] = base_id + pos

        # Build adjacency and in-degree from edges.
        children = {nid: [] for nid in node_specs}     # node_id -> [(child, delay_ns)]
        pending_parents = {nid: 0 for nid in node_specs}
        for edge in row.get('edges', []):
            src, dst = edge['src'], edge['dst']
            delay_ns = int(edge.get('delay_ns', 0))
            children[src].append((dst, delay_ns))
            pending_parents[dst] += 1

        self._workflows[workflow_id] = {
            'specs': node_specs,
            'index': node_index,
            'children': children,
            'pending_parents': pending_parents,
            'ready_time': {nid: arrival_ns for nid in node_specs},  # max(parent_end + delay)
            'released': set(),
            'completed': 0,
            'total': len(node_specs),
            'unreleased': len(node_specs),
            'arrival_ns': arrival_ns,
            'completions': {},               # node_id -> completion_time_ns
        }

        # Queue all root nodes (in-degree 0) at the workflow arrival time.
        for node_id in node_specs:
            if pending_parents[node_id] == 0:
                self._release_node(workflow_id, node_id, arrival_ns,
                                   to_pending_queue=True,
                                   enable_prefix_caching=enable_prefix_caching)

        return len(node_specs)

    def _release_node(self, workflow_id, node_id, release_time_ns,
                      to_pending_queue=False, enable_prefix_caching=None):
        """Build a request for a DAG node and add it to the pending queue."""
        if enable_prefix_caching is None:
            enable_prefix_caching = self._enable_prefix_caching
        wf = self._workflows[workflow_id]
        spec = wf['specs'][node_id]
        node_idx = wf['index'][node_id]
        req_data = {
            'index': node_idx,
            'input_toks': int(spec['input_toks']),
            'output_toks': int(spec['input_toks'] + spec['output_toks']),
            'arrival_time_ns': int(release_time_ns),
            'workflow_id': workflow_id,
            'node_id': node_id,
        }
        if enable_prefix_caching:
            req_data['input_hash_ids'] = spec.get('input_tok_ids', [])
            req_data['output_hash_ids'] = derive_output_ids(
                spec, None, int(spec.get('output_toks', 0)), req_data['index'])
        if to_pending_queue:
            # Loader path: appended now, sorted once after the full load.
            self._pending_requests.append(req_data)
        else:
            # Runtime release: keep the unconsumed queue sorted by arrival.
            self._insert_pending_sorted(req_data)
        self._request_to_node[node_idx] = (workflow_id, node_id)
        wf['released'].add(node_id)
        wf['unreleased'] -= 1

    def route_arrived_requests(self, current_time_ns):
        """Route requests that have arrived by current_time_ns to instances.

        Called at the start of each iteration in the main simulation loop.
        Returns the number of newly routed requests.
        """
        routed = 0
        if self.policy_adapter is not None:
            # SAGA: an idle instance may take a queued session from a
            # loaded one before this tick's arrivals are placed.
            self.policy_adapter.steal_tick(current_time_ns)
            # SAGA: recompute a paused session's context just before its
            # tool result is due, so its successor hits a warm prefix.
            self.policy_adapter.prefetch_tick(current_time_ns)
        while self._pending_idx < len(self._pending_requests):
            req_data = self._pending_requests[self._pending_idx]
            if req_data['arrival_time_ns'] > current_time_ns:
                break

            if self.policy_adapter is not None:
                instance_id = self.policy_adapter.select_instance(
                    req_data,
                    lambda: self._select_instance(self.prefill_schedulers, "prefill"),
                    current_time_ns)
                priority = self.policy_adapter.on_turn_routed(req_data)
            else:
                instance_id = self._select_instance(self.prefill_schedulers, "prefill")
                priority = None
            sched = self.prefill_schedulers[instance_id]

            wf_id = req_data.get('workflow_id')
            node_id = req_data.get('node_id')
            ident = dict(session_id=req_data.get('session_id'),
                         sub_request_index=req_data.get('sub_request_index'))
            if sched.enable_prefix_caching:
                sched.add_request([
                    req_data['index'], sched.model,
                    req_data['input_toks'], req_data['output_toks'],
                    req_data['arrival_time_ns'], sched.instance_id,
                    req_data.get('input_hash_ids', []), req_data.get('output_hash_ids', []),
                ], is_init=self._is_init, workflow_id=wf_id, node_id=node_id,
                   priority=priority, **ident)
            else:
                sched.add_request([
                    req_data['index'], sched.model,
                    req_data['input_toks'], req_data['output_toks'],
                    req_data['arrival_time_ns'], sched.instance_id,
                ], is_init=self._is_init, workflow_id=wf_id, node_id=node_id,
                   priority=priority, **ident)

            self._pending_idx += 1
            routed += 1

        return routed

    def has_pending_requests(self):
        """Check if there are unrouted requests remaining."""
        return self._pending_idx < len(self._pending_requests)

    def get_first_arrival_time(self):
        """Return the first request's arrival time in ns, or 1 if no requests."""
        if self._pending_requests:
            return max(1, self._pending_requests[0]['arrival_time_ns'])
        return 1

    # -----------------------------------------------------------------------
    # Agentic dependency chain management
    # -----------------------------------------------------------------------

    def notify_request_completed(self, request_id, completion_time_ns,
                                 req_obj=None, memory=None):
        """Called when a request finishes. Releases any successor work whose
        dependencies are now satisfied.

        Dispatches by request kind:
        - DAG workflow node -> release child nodes whose last parent just finished
        - agentic session sub-request -> release the next link in the chain
        - flat request -> no-op

        req_obj/memory (the finished Request and its instance's
        MemoryModel) feed the unified-policy adapter's turn-complete
        event when one is configured.
        """
        node_info = self._request_to_node.pop(request_id, None)
        if node_info is not None:
            if self.policy_adapter is not None:
                wf_id, nid = node_info
                spec = self._workflows[wf_id]['specs'][nid] if wf_id in self._workflows else {}
                self.policy_adapter.on_turn_complete(
                    req_obj, wf_id, nid, None,
                    int(spec.get('input_toks', 0)) + int(spec.get('output_toks', 0)),
                    memory, completion_time_ns)
            self._notify_dag_node_completed(node_info, completion_time_ns)
            return

        session_info = self._request_to_session.pop(request_id, None)
        if session_info is None:
            if self.policy_adapter is not None and req_obj is not None:
                # Flat request: only the routing in-flight count updates.
                self.policy_adapter.on_turn_complete(
                    req_obj, None, None, None, 0, memory, completion_time_ns)
            return
        session_id, completed_idx = session_info
        session = self._deferred_sessions.get(session_id)
        if session is None:
            return

        sub_reqs = session['sub_requests']
        next_idx = session['next_index']
        base_id = session['id_base']

        completed = sub_reqs[completed_idx]
        if self.policy_adapter is not None:
            # Traces without tool identity still enter a real gap when a
            # successor turn exists; the placeholder name keeps the PCB's
            # gap record (in_gap/gap_started_ts) live so policies can
            # measure gap durations at the next arrival.
            tool = completed.get('tool') or completed.get('tool_name')
            if tool is None and session['next_index'] < len(sub_reqs):
                tool = 'tool'
            self.policy_adapter.on_turn_complete(
                req_obj, session_id, completed_idx, tool,
                int(completed['input_toks']) + int(completed['output_toks']),
                memory, completion_time_ns)

        # Get tool duration from the completed sub-request
        tool_duration_ns = int(completed.get('tool_duration_ns', 0))
        release_time_ns = completion_time_ns + tool_duration_ns

        if next_idx < len(sub_reqs):
            # Release next sub-request
            next_sub = sub_reqs[next_idx]
            next_id = base_id + next_idx
            req_data = {
                'index': next_id,
                'input_toks': int(next_sub['input_toks']),
                'output_toks': int(next_sub['input_toks'] + next_sub['output_toks']),
                'arrival_time_ns': release_time_ns,
                'completed_tool_duration_s': tool_duration_ns / 1e9,
                'session_id': session_id,
                'sub_request_index': next_idx,
            }
            if self._enable_prefix_caching:
                req_data['input_hash_ids'] = next_sub.get('input_tok_ids', [])
                req_data['output_hash_ids'] = derive_output_ids(
                    next_sub,
                    sub_reqs[next_idx + 1] if next_idx + 1 < len(sub_reqs) else None,
                    int(next_sub['output_toks']), next_id)
            # Insert in sorted position after _pending_idx
            self._insert_pending_sorted(req_data)
            self._request_to_session[next_id] = (session_id, next_idx)
            session['next_index'] = next_idx + 1
        else:
            # Session complete — all sub-requests have been released and the
            # last one just finished. Record per-session JCT (arrival -> last
            # completion) so agentic runs emit the same <output>_workflows.csv
            # the DAG path does, keyed by session_id (num_nodes = turn count).
            self._workflow_metrics.append({
                'workflow_id': session_id,
                'arrival_ns': session['arrival_ns'],
                'end_ns': completion_time_ns,
                'jct_ns': completion_time_ns - session['arrival_ns'],
                'num_nodes': len(sub_reqs),
            })
            del self._deferred_sessions[session_id]

    def _notify_dag_node_completed(self, node_info, completion_time_ns):
        """A DAG node finished: advance its children's dependency counters and
        release any child whose final parent just completed."""
        workflow_id, node_id = node_info
        wf = self._workflows.get(workflow_id)
        if wf is None:
            return

        wf['completions'][node_id] = completion_time_ns

        for child_id, delay_ns in wf['children'][node_id]:
            if child_id in wf['released']:
                continue
            # A child's ready time is the max over its parents of
            # (parent completion + edge delay) — this is the barrier wait.
            ready = completion_time_ns + delay_ns
            if ready > wf['ready_time'][child_id]:
                wf['ready_time'][child_id] = ready
            wf['pending_parents'][child_id] -= 1
            if wf['pending_parents'][child_id] == 0:
                self._release_node(workflow_id, child_id, wf['ready_time'][child_id])

        wf['completed'] += 1
        if wf['completed'] >= wf['total']:
            # Every node has completed — record JCT and drop the state.
            self._record_workflow_metric(workflow_id, wf)
            del self._workflows[workflow_id]

    def _record_workflow_metric(self, workflow_id, wf):
        """JCT and shape of a fully completed workflow. JCT = last node's
        completion minus the workflow arrival (the whole DAG's wall time)."""
        arrival = wf['arrival_ns']
        end = max(wf['completions'].values())
        self._workflow_metrics.append({
            'workflow_id': workflow_id,
            'arrival_ns': arrival,
            'end_ns': end,
            'jct_ns': end - arrival,
            'num_nodes': wf['total'],
        })

    def _insert_pending_sorted(self, req_data):
        """Insert a request into _pending_requests maintaining arrival-time
        sort order for the not-yet-consumed portion (from _pending_idx onward)."""
        arrival = req_data['arrival_time_ns']
        # Binary search in the unconsumed portion
        lo = self._pending_idx
        hi = len(self._pending_requests)
        while lo < hi:
            mid = (lo + hi) // 2
            if self._pending_requests[mid]['arrival_time_ns'] <= arrival:
                lo = mid + 1
            else:
                hi = mid
        self._pending_requests.insert(lo, req_data)

    def has_deferred_sessions(self):
        """Check if there is deferred dependency work not yet released:
        agentic-session chains or DAG-workflow nodes still waiting on parents.
        Gates simulation exit so we don't stop with successors unreleased."""
        if self._deferred_sessions:
            return True
        return any(wf['unreleased'] > 0 for wf in self._workflows.values())

    # -----------------------------------------------------------------------
    # Multi-agent metrics (per-workflow JCT + tail)
    # -----------------------------------------------------------------------

    def has_workflow_metrics(self):
        return bool(self._workflow_metrics)

    @staticmethod
    def _percentile(sorted_vals, p):
        """Linear-interpolated percentile of a pre-sorted list."""
        if not sorted_vals:
            return 0.0
        k = (len(sorted_vals) - 1) * (p / 100.0)
        lo = int(k)
        hi = min(lo + 1, len(sorted_vals) - 1)
        if lo == hi:
            return float(sorted_vals[lo])
        return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)

    def workflow_metrics_summary(self):
        """Aggregate JCT (mean + P50/P90/P99 tail) and workflow throughput
        across all completed workflows. Returns a dict (ns for latencies)."""
        if not self._workflow_metrics:
            return None
        jcts = sorted(m['jct_ns'] for m in self._workflow_metrics)
        n = len(jcts)
        first_arrival = min(m['arrival_ns'] for m in self._workflow_metrics)
        last_end = max(m['end_ns'] for m in self._workflow_metrics)
        makespan_ns = max(1, last_end - first_arrival)
        return {
            'num_workflows': n,
            'jct_mean_ns': sum(jcts) / n,
            'jct_p50_ns': self._percentile(jcts, 50),
            'jct_p90_ns': self._percentile(jcts, 90),
            'jct_p99_ns': self._percentile(jcts, 99),
            'jct_min_ns': jcts[0],
            'jct_max_ns': jcts[-1],
            'makespan_ns': makespan_ns,
            'workflow_throughput_per_s': n / (makespan_ns / 1e9),
        }

    def save_workflow_metrics(self, path):
        """Write one row per completed workflow (workflow_id, arrival, end,
        jct, num_nodes), sorted by arrival time. Mirrors Scheduler.save_output
        path handling: relative paths resolve from the repo root (the simulator
        runs with cwd=astra-sim/)."""
        import os
        if not os.path.isabs(path):
            path = f'../{path}'
        out_dir = os.path.dirname(path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        rows = sorted(self._workflow_metrics, key=lambda m: m['arrival_ns'])
        with open(path, 'w') as f:
            f.write("workflow_id,arrival_ns,end_ns,jct_ns,num_nodes\n")
            for m in rows:
                f.write(f"{m['workflow_id']},{m['arrival_ns']},{m['end_ns']},"
                        f"{m['jct_ns']},{m['num_nodes']}\n")

    def get_next_pending_arrival(self):
        """Return the next pending request's arrival time, or None."""
        if self._pending_idx < len(self._pending_requests):
            return self._pending_requests[self._pending_idx]['arrival_time_ns']
        return None

    # -----------------------------------------------------------------------
    # Legacy: upfront routing (kept for backward compat)
    # -----------------------------------------------------------------------

    def generate(self, path, enable_prefix_caching=False, is_init=True):
        """Load and immediately route all requests (legacy behavior)."""
        self.load_requests(path, enable_prefix_caching, is_init)
        # Route all at once (arrival time ignored)
        self.route_arrived_requests(float('inf'))
        for scheduler in self.schedulers:
            self.logger.info(
                "Added %d requests to scheduler[%d] (%s type)",
                len(scheduler.request),
                scheduler.instance_id,
                scheduler.pd_type
            )

    def transfer_prefill_request(self, requests):
        for req in requests:
            instance_id = self._select_instance(self.decode_schedulers, "decode")
            self.decode_schedulers[instance_id].add_decode(req)
