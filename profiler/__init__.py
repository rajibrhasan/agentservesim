"""Layerwise profiler for LLMServingSim."""

# ---------------------------------------------------------------------------
# _typeshed shim — runs FIRST, before anything else imports.
# ---------------------------------------------------------------------------
#
# vLLM has an import path that references ``_typeshed.DataclassInstance``
# at runtime. ``_typeshed`` is a typing-only stub module and isn't
# available as a real module, so the import fails with "No module named
# '_typeshed'". The fix is a tiny shim: register a fake ``_typeshed``
# module containing the bare attribute vLLM happens to touch.
#
# This runs in every process that imports ``profiler`` — the host
# (before spin_up() constructs vllm.LLM) and every vLLM worker
# process (before the Extension class is loaded via
# worker_extension_cls). Placing it in ``profiler/__init__.py``
# covers both paths automatically.

import sys as _sys
import types as _types

if "_typeshed" not in _sys.modules:
    _shim = _types.ModuleType("_typeshed")
    _shim.DataclassInstance = object  # type: ignore[attr-defined]
    _sys.modules["_typeshed"] = _shim
    del _shim

del _sys, _types


__version__ = "1.0.0"
