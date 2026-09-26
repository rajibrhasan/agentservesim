"""Did the policy a run claims to have executed actually do anything?

Split out of evolve/simrun.py on 2026-09-12.

A policy whose defining event never fired produced a number for some OTHER
policy -- the engine default -- and that number is indistinguishable from a
good one by inspection. `mechanism_check` refuses such a run rather than
returning its JCT. The faithful-Continuum leaderboard results (Sep 2026) were
exactly this failure: release-at-admission never fired on session traces, and
every number looked plausible.

This is the simulator-side twin of the arena's own `mechanism_check`; the two
guard the same thing at different distances from the run.
"""

import ast
import csv
import json
import os
import re

def _read_workflows(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return [int(r["jct_ns"]) / 1e9 for r in rows]


def _decision_stats(log_dir):
    counts = {"protect": 0, "evict": 0, "release": 0, "swap": 0, "none": 0}
    p = os.path.join(log_dir, "retention.jsonl")
    if not os.path.exists(p):
        return counts
    with open(p) as f:
        for line in f:
            try:
                a = json.loads(line).get("action")
            except ValueError:
                continue
            if a in counts:
                counts[a] += 1
    return counts


BACKEND_DEATH = "ASTRA-Sim backend terminated"

_STATS_RE = re.compile(r"KV protection stats:\s*(\{.*?\})")


def _mechanism_stats(stdout, log_dir):
    """Adapter event counters printed at the end of a run, plus the number
    of priority stamps in the scheduling decision log."""
    out = {}
    m = _STATS_RE.search(stdout or "")
    if m:
        try:
            out.update(ast.literal_eval(m.group(1)))
        except (ValueError, SyntaxError):
            pass
    p = os.path.join(log_dir, "scheduling.jsonl")
    if os.path.exists(p):
        with open(p) as f:
            out["stamps"] = sum(1 for _ in f)
    return out


def mechanism_check(flags, mech):
    """Reasons a run's counters contradict the policy it claims to have run.
    A policy whose defining event never fired produced a number for some
    OTHER policy (the engine default), and that number must not be
    reported: the faithful-Continuum lb results (Sep 2026) were exactly
    this, release-at-admission never firing on session traces."""
    flags = list(flags or [])
    def val(k):
        return flags[flags.index(k) + 1] if k in flags and flags.index(k) + 1 < len(flags) else None
    ret, sch = val("--retention"), val("--scheduling")
    need = []
    # Evolved candidates are exempt: never protecting / never stamping is a
    # legitimate decision rule for a search candidate, not a broken mechanism.
    if ret in ("ttl", "saga-ttl", "oracle-ttl", "gate"):
        need += [("protected", "retention never protected a context"),
                 ("released", "retention never released a protection")]
    if ret == "min-waste":
        # InferCept's outcomes are preserve (protect + release), swap or
        # discard; a run in which it only ever discarded ran the engine
        # default. Either protect or swap shows the policy acted.
        need += [(("protected", "swapped_out"),
                  "min-waste never preserved or swapped a context")]
    if ret == "continuum":
        need += [("protected", "continuum never pinned"),
                 ("admission_releases", "continuum never released a pin at admission")]
    if sch in ("program-fcfs", "plas", "continuum", "oracle-srpt", "gate"):
        need += [("stamps", "priority scheduling never stamped a turn")]
    if not mech:
        return ["no mechanism counters found in simulator output"] if need else []
    def fired(k):
        keys = k if isinstance(k, tuple) else (k,)
        return any(mech.get(x) for x in keys)
    return [why for k, why in need if not fired(k)]

