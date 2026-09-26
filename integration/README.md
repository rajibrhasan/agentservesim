# vLLM integration

`vllm-v0.19.0.patch` is a source diff against the upstream v0.19.0 tag, covering Python files changed or added in the engine used by the real replay. Apply it to a vLLM v0.19.0 source checkout with `git apply /path/to/integration/vllm-v0.19.0.patch`, then build/install that checkout using its upstream instructions. Make this artifact repository importable alongside vLLM because the integration calls `bench` and `policies` modules.

The patch was extracted from the current engine source; it does not include binaries, CUDA build artifacts, model weights, credentials, or an installed Python environment. A fresh GPU build and end-to-end validation of this exported package remain required before claiming turnkey reproducibility.
