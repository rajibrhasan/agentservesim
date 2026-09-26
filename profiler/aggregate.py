"""Aggregate complete replica CSVs by identical shape, retaining dispersion."""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics


def aggregate_tables(paths):
    tables = []
    schema = None
    for path in paths:
        with Path(path).open() as f:
            reader = csv.DictReader(f)
            fields = reader.fieldnames
            if not fields or "time_us" not in fields:
                raise ValueError(f"Not a layer timing table: {path}")
            if schema is not None and fields != schema:
                raise ValueError(f"Schema mismatch: {path}")
            schema = fields
            keys = [k for k in fields if k != "time_us"]
            table = {}
            for row in reader:
                key = tuple(row[k] for k in keys)
                value = float(row["time_us"])
                if key in table or not math.isfinite(value) or value <= 0:
                    raise ValueError(f"Duplicate or invalid measurement in {path}: {key}")
                table[key] = value
            if not table:
                raise ValueError(f"Empty profile: {path}")
            tables.append(table)
    if len(tables) < 2 or any(t.keys() != tables[0].keys() for t in tables[1:]):
        raise ValueError("At least two replicas with identical shape coverage are required")
    output, dispersion = [], []
    for key in sorted(tables[0]):
        values = [t[key] for t in tables]
        mean = statistics.mean(values)
        row = dict(zip(keys, key))
        output.append({**row, "time_us": mean})
        dispersion.append({**row, "mean_us": mean, "median_us": statistics.median(values),
                           "min_us": min(values), "max_us": max(values),
                           "cv": statistics.stdev(values) / mean, "replicas": len(values),
                           "samples_us": values})
    return schema, output, dispersion


def main():
    p = argparse.ArgumentParser()
    p.add_argument("roots", nargs="+", type=Path)
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()
    import yaml
    metas = [yaml.safe_load((r / "meta.yaml").read_text()) for r in a.roots]
    for field in ("model", "hardware", "variant", "architecture_sha256", "tp_degrees",
                  "measurement_iterations", "vllm_version", "cuda_version", "attention_grid"):
        if any(m[field] != metas[0][field] for m in metas[1:]):
            raise ValueError(f"Incompatible replica metadata: {field}")
    files = [{p.relative_to(r) for p in r.glob('tp*/*.csv')
              if p.name in {'dense.csv', 'per_sequence.csv', 'attention.csv', 'moe.csv'}}
             for r in a.roots]
    if not files[0] or any(s != files[0] for s in files[1:]):
        raise ValueError("Replica file coverage differs")
    prepared = []
    for rel in sorted(files[0]):
        prepared.append((rel, aggregate_tables([r / rel for r in a.roots])))
    a.out.mkdir(parents=True, exist_ok=False)
    for rel, (fields, rows, dispersion) in prepared:
        target = a.out / rel
        target.parent.mkdir(exist_ok=True)
        with target.open('w') as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        target.with_suffix('.dispersion.json').write_text(json.dumps(dispersion, indent=2))
    # Refit skew from pooled raw measurements, rather than averaging fitted alpha.
    for tp in metas[0]['tp_degrees']:
        paths = [r / f'tp{tp}/skew.csv' for r in a.roots]
        if any(p.exists() for p in paths):
            if not all(p.exists() for p in paths):
                raise ValueError('Missing skew replica')
            import pandas as pd
            pd.concat([pd.read_csv(p) for p in paths], ignore_index=True).to_csv(
                a.out / f'tp{tp}/skew.csv', index=False)
    from profiler.core.writer import _skew_fit_block
    meta = metas[0]
    meta['skew_fit'] = _skew_fit_block(a.out, meta['tp_degrees'])
    meta['replica_roots'] = [str(r.resolve()) for r in a.roots]
    meta['aggregation'] = 'arithmetic mean of latency; raw replicas retained'
    (a.out / 'meta.yaml').write_text(yaml.safe_dump(meta, sort_keys=False))


if __name__ == '__main__':
    main()
