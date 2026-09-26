# Anonymous submission artifact

This repository contains the serving simulator, program-aware policies, replay driver, profiling pipeline, and source changes for the reference serving engine.

## Contents

- `serving/`: simulator and scheduling, routing, and KV management integration.
- `policies/`, `runtime/`, `harness/`, `evolve/`: policy interfaces, implementations, and search support.
- `bench/`: real-serving replay and measurement.
- `profiler/`: profiling code and selected B200/RTX PRO 6000 performance tables.
- `configs/`: model and deployment descriptions.
- `astra-sim/`: vendored analytical backend and its initialized source dependencies.
- `integration/`: Python integration patch for upstream vLLM v0.19.0.
- `tests/`: source regression tests; some require dependencies or experiment fixtures not included here.

## Simulator setup

With Docker available, from this directory:

```sh
bash scripts/docker-sim.sh
# Inside the container:
bash scripts/compile.sh
python3 -m serving --help
```

The Docker build downloads dependencies and requires network access. The analytical ASTRA-Sim backend is included; ns-3 and htsim backends are outside this export. Consult the command-line help for the cluster and workload options. Workload conversion utilities are in `workloads/generators/`.

## Reference serving

See `integration/README.md` for applying the engine patch. Real replay requires GPUs, model weights obtained under their respective licenses, and the patched vLLM installation. Profiling is a separate preparation step; the included tables do not cover every deployment in `configs/`.

## Submission scope

This is a source artifact, not a complete result-reproduction bundle. It excludes private credentials, original Git history, cluster job scripts, installed environments, raw experiment logs, model weights, and workload datasets. Site-specific launch paths have been replaced with placeholders; supply local paths when using those tools. Fresh container builds, GPU replay, and the complete test suite have not been validated on this exported copy.

Upstream licenses and required copyright notices are retained. These attributions identify dependencies and must remain with redistributed source. The fresh repository uses an anonymous commit identity. `FILE_MANIFEST.json` records exported file checksums.
