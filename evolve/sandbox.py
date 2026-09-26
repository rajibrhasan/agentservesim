"""Static observation-boundary check for retention candidates.

Runs before any simulation. A candidate that reads outside the PCB,
reaches for the filesystem, the clock, or randomness, or keys state by
program identity is rejected here with a reason the search can read.
The check is syntactic (AST), so it is cheap and deterministic; the
boundary it enforces is the one the paper states: the policy sees the
Program Control Block and the current time, nothing else.
"""

import ast
import os

ALLOWED_MODULES = {
    "math", "statistics", "collections", "typing", "dataclasses",
    # The contract, under the name the policies package uses now.
    "policies.base", "policies.program", "policies.utils.waste_model",
    # The legacy axis names. Kept because every champion already evolved --
    # and every best_program.py under evolve/results/ -- was written against
    # them, and those are records of runs that happened. Removing them would
    # not make an old candidate wrong, it would make it unimportable, which is
    # a different and worse thing.
    "harness.retention", "harness.program", "harness.waste_model",
    "harness.scheduling",
}
FORBIDDEN_CALLS = {
    "open", "exec", "eval", "compile", "__import__", "globals", "locals",
    "getattr", "setattr", "delattr", "vars", "input", "breakpoint",
}
# Identity fields: a policy decides from program STATE, never from which
# program it is. Dunder access closes the reflection routes around that.
FORBIDDEN_ATTRS = {"program_id", "kv_request_id", "__dict__", "__class__",
                   "__globals__", "__subclasses__", "__code__", "__builtins__"}
# One class per plane, several per file -- the convention `policies/continuum.py`
# and every recorded champion follow. A candidate must define at least one of
# them; whatever it leaves out falls back to the engine's own rule, which is a
# legal policy and not an error.
#
# This was three axis-specific contracts selected by EVOLVE_AXIS, which forced
# the search to be told which plane it was searching -- a fact already implied
# by the classes a candidate defines.
PLANE_CLASSES = {
    "retention": ("EvolvedRetention", {"on_turn_complete", "on_turn_arrival"}),
    "scheduling": ("EvolvedScheduling", {"priority", "victim", "admit"}),
    "routing": ("EvolvedRouting", {"route"}),
}
#: Kept for callers that ask what a candidate may define.
REQUIRED_CLASS = "EvolvedRetention"
REQUIRED_METHODS = set().union(*(m for _, m in PLANE_CLASSES.values()))


def check_source(src):
    """Return (ok, reasons). ok is False when any reason is present."""
    reasons = []
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        return False, [f"syntax error: {e}"]

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name not in ALLOWED_MODULES:
                    reasons.append(f"import of '{a.name}' is not allowed")
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if mod not in ALLOWED_MODULES:
                reasons.append(f"import from '{mod}' is not allowed")
        elif isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Name) and f.id in FORBIDDEN_CALLS:
                # getattr with a literal, non-forbidden, non-dunder name is the
                # defensive form of plain attribute access (candidates write
                # getattr(self.signals, "kv_utilization", None)); only the
                # dynamic form can reach identity fields.
                lit = (f.id == "getattr" and len(node.args) >= 2
                       and isinstance(node.args[1], ast.Constant)
                       and isinstance(node.args[1].value, str)
                       and node.args[1].value not in FORBIDDEN_ATTRS
                       and not node.args[1].value.startswith("__"))
                if not lit:
                    reasons.append(f"call to '{f.id}' is not allowed")
        elif isinstance(node, ast.Attribute):
            if node.attr in FORBIDDEN_ATTRS:
                reasons.append(f"access to '.{node.attr}' is not allowed")
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            reasons.append("global/nonlocal state is not allowed")

    defined = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)}
    decided = []
    for plane, (cls_name, methods) in PLANE_CLASSES.items():
        node = defined.get(cls_name)
        if node is None:
            continue
        have = {n.name for n in node.body
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
        if have & methods:
            decided.append(plane)
        else:
            reasons.append(
                f"{cls_name} defines none of {sorted(methods)}")
        for n in node.body:
            if isinstance(n, ast.AsyncFunctionDef):
                reasons.append("async methods are not allowed")
    if not decided and not reasons:
        reasons.append(
            "no policy class: define at least one of "
            f"{sorted(c for c, _ in PLANE_CLASSES.values())}")

    # De-duplicate while preserving order.
    seen = set()
    reasons = [r for r in reasons if not (r in seen or seen.add(r))]
    return (not reasons), reasons


def check_file(path):
    with open(path) as f:
        return check_source(f.read())


if __name__ == "__main__":
    import sys
    ok, reasons = check_file(sys.argv[1])
    print("OK" if ok else "REJECTED")
    for r in reasons:
        print(" -", r)
    sys.exit(0 if ok else 1)
