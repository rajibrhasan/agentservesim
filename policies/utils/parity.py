

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Iterable, Optional

KNOB_FIELDS = {
    "retention": ("program_id", "turn_idx", "action", "request_id"),
    "scheduling": ("program_id", "turn_idx", "priority"),
    "routing": ("program_id", "turn_idx", "instance"),
}

# info keys compared numerically with a relative tolerance when both
# sides carry them (harness floats vs mirror floats).
RETENTION_INFO_KEYS = ("w_preserve", "w_discard", "gap_pred_s", "context_tokens")
REL_TOL = 1e-9


@dataclass
class ParityReport:
    knob: str
    n_real: int
    n_sim: int
    mismatches: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.mismatches and self.n_real == self.n_sim

    def summary(self) -> str:
        status = "PARITY" if self.ok else "DIVERGED"
        lines = [f"{status} knob={self.knob} real={self.n_real} sim={self.n_sim}"]
        lines += self.mismatches[:20]
        if len(self.mismatches) > 20:
            lines.append(f"... {len(self.mismatches) - 20} more")
        return "\n".join(lines)


def _load(path_or_lines) -> list[dict]:
    if isinstance(path_or_lines, str):
        with open(path_or_lines) as f:
            return [json.loads(x) for x in f if x.strip()]
    return [json.loads(x) for x in path_or_lines if x.strip()]


def _num_close(a, b) -> bool:
    if a == b:
        return True
    try:
        a, b = float(a), float(b)
    except (TypeError, ValueError):
        return False
    return abs(a - b) <= REL_TOL * max(abs(a), abs(b), 1.0)


def check_parity(knob: str, real, sim) -> ParityReport:
    """Compare two decision logs (paths or iterables of JSONL lines)."""
    if knob not in KNOB_FIELDS:
        raise ValueError(f"unknown knob: {knob}")
    fields = KNOB_FIELDS[knob]
    real_recs, sim_recs = _load(real), _load(sim)
    rep = ParityReport(knob, len(real_recs), len(sim_recs))
    if len(real_recs) != len(sim_recs):
        rep.mismatches.append(
            f"length: real has {len(real_recs)} decisions, sim {len(sim_recs)}"
        )
    for i, (r, s) in enumerate(zip(real_recs, sim_recs)):
        for f in fields:
            if r.get(f) != s.get(f):
                rep.mismatches.append(
                    f"seq {i} {f}: real={r.get(f)!r} sim={s.get(f)!r}"
                )
        if knob == "retention":
            if r.get("blocks") is not None and s.get("blocks") is not None:
                if r["blocks"] != s["blocks"]:
                    rep.mismatches.append(
                        f"seq {i} blocks: real={r['blocks']} sim={s['blocks']}"
                    )
            ri, si = r.get("info"), s.get("info")
            if ri and si:
                for k in RETENTION_INFO_KEYS:
                    if k in ri and k in si and not _num_close(ri[k], si[k]):
                        rep.mismatches.append(
                            f"seq {i} info.{k}: real={ri[k]} sim={si[k]}"
                        )
        if knob == "routing":
            rf = (r.get("info") or {}).get("fallback", False)
            sf = (s.get("info") or {}).get("fallback", False)
            if rf != sf:
                rep.mismatches.append(
                    f"seq {i} fallback: real={rf} sim={sf}"
                )
    return rep


def main(argv: Optional[Iterable[str]] = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("knob", choices=sorted(KNOB_FIELDS))
    ap.add_argument("real_log")
    ap.add_argument("sim_log")
    args = ap.parse_args(argv)
    rep = check_parity(args.knob, args.real_log, args.sim_log)
    print(rep.summary())
    return 0 if rep.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
