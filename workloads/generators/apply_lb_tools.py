"""Verify the tool-name leaderboard traces equal the old ones except for the
`tool` field, then replace the impact lb subsets (backup kept as .notools.bak)."""
import json, collections, os, shutil, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
from runtime.paths import MAS       # noqa: E402
LB = os.path.join(MAS, "workloads", "leaderboard")
IMP = os.path.join(MAS, "workloads", "impact")
def load(p): return [json.loads(l) for l in open(p)]
old = load(f"{LB}/swelb_jps0.1_n500.jsonl"); new = load(f"{LB}/tools_staging/swelb_tools_jps0.1_n500.jsonl")
assert len(old) == len(new) == 500
diff = 0; tools = collections.Counter()
for a, b in zip(old, new):
    assert a["session_id"] == b["session_id"] and a["arrival_time_ns"] == b["arrival_time_ns"]
    assert len(a["sub_requests"]) == len(b["sub_requests"])
    for sa, sb in zip(a["sub_requests"], b["sub_requests"]):
        for k in ("input_toks", "output_toks", "tool_duration_ns", "input_tok_ids"):
            if sa.get(k) != sb.get(k): diff += 1
        tools[sb.get("tool")] += 1
print("field mismatches (excluding tool):", diff, "| turns", sum(tools.values()), "| tool None", tools[None], "| distinct", len(tools))
print("top labels", tools.most_common(12))
assert diff == 0
del old
new05 = load(f"{LB}/tools_staging/swelb_tools_jps0.05_n500.jsonl")
for n, src, name in ((20, new, "swelb20_jps0.1"), (40, new, "swelb40_jps0.1"), (60, new, "swelb60_jps0.1"),
                     (20, new05, "swelb20_jps0.05"), (40, new05, "swelb40_jps0.05")):
    oldp = f"{IMP}/{name}.jsonl"; oldrows = load(oldp)
    assert [r["session_id"] for r in oldrows] == [r["session_id"] for r in src[:n]], name
    assert [r["arrival_time_ns"] for r in oldrows] == [r["arrival_time_ns"] for r in src[:n]], name
    shutil.copy(oldp, oldp + ".notools.bak")
    with open(oldp, "w") as f:
        for r in src[:n]: f.write(json.dumps(r) + "\n")
    print("replaced", name, "programs", n)
print("APPLY_DONE")
