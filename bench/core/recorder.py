

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


META_SCHEMA_VERSION = 1


def write_meta(output_dir: Path, **fields: Any) -> None:
    """Write meta.json. Required fields: model, vllm_version, engine_kwargs,
    dataset_path, dataset_hash, started_at, finished_at, num_requests."""
    payload = {"schema_version": META_SCHEMA_VERSION, **fields}
    (output_dir / "meta.json").write_text(json.dumps(payload, indent=2))


def write_requests(output_dir: Path, records: list[dict]) -> None:
    """Write requests.jsonl. Each record::"""
    with (output_dir / "requests.jsonl").open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def write_timeseries(output_dir: Path, header: list[str], rows: list[list]) -> None:
    """Write timeseries.csv. Default header::

        ["t", "prompt_throughput", "gen_throughput",
         "running", "waiting", "kv_cache_pct"]
    """
    import csv
    with (output_dir / "timeseries.csv").open("w") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
