"""Compatibility shim. The policies live in `policies/`.

Still here for one concrete reason: `evolve/sandbox.py` allows an evolved
candidate to import `harness.retention`, `harness.program`,
`harness.waste_model` and `harness.scheduling`, and every champion already
evolved -- plus every `best_program.py` under `evolve/results/` -- was written
against those names. Those are records of runs that happened. Removing the
names would not make an old candidate wrong, it would make it unimportable,
which is a different and worse thing.

`harness.X` and `policies.X` resolve to the SAME module object, not to two
copies. That matters: a champion importing `harness.evolved_joint` while the
staged shim imports `policies.evolved_joint` would otherwise get two instances
of the candidate, with separate module state, and a search would score a
policy that is not the one it ran.

Delete this package once the sandbox allowlist, the seeds and the recorded
champions have all moved.
"""
import importlib
import importlib.abc
import importlib.util
import sys as _sys

import policies as _policies

#: old flat name -> where it lives under `policies/`. Three helpers moved into
#: `utils/`, the three axis modules dissolved into the paper modules (their
#: classes are all bound on the package), and `serve_agent` left entirely: it
#: launches a vLLM server, which was never a policy.
_MAP = {
    "retention": "", "scheduling": "", "routing": "",   # "" = the package
    "program": "program", "base": "base", "executors": "executors",
    "oracle": "oracle", "example_unified": "example_unified",
    "stock": "stock", "continuum": "continuum", "saga": "saga",
    "autellix": "autellix", "infercept": "infercept", "generic": "generic",
    "utils": "utils", "policies": "",
    "waste_model": "utils.waste_model", "kv_control": "utils.kv_control",
    "parity": "utils.parity",
}


class _AliasLoader(importlib.abc.Loader):
    """Returns the real module rather than executing a second copy."""

    def __init__(self, target):
        self._target = target

    def create_module(self, spec):
        return (_policies if not self._target
                else importlib.import_module(f"policies.{self._target}"))

    def exec_module(self, module):
        return None                      # already executed under its real name


class _AliasFinder(importlib.abc.MetaPathFinder):
    """Resolves `harness.X` to the module `policies.X` already is.

    A finder rather than a table of `sys.modules` entries, because candidates
    are staged into `policies/` at search time under names this file cannot
    know in advance (`evolved_joint`, `evolved_retention`, ...).
    """

    def find_spec(self, fullname, path=None, target=None):
        if not fullname.startswith(__name__ + "."):
            return None
        sub = fullname[len(__name__) + 1:]
        where = _MAP.get(sub, sub)
        if where:
            try:
                importlib.import_module(f"policies.{where}")
            except ImportError:
                return None
        return importlib.util.spec_from_loader(fullname, _AliasLoader(where))


_sys.meta_path.insert(0, _AliasFinder())

# `from harness import retention` reads an attribute off this package, which
# the finder never sees -- so bind the ones that exist today eagerly.
for _name, _where in _MAP.items():
    try:
        globals()[_name] = (_policies if not _where
                            else importlib.import_module(f"policies.{_where}"))
    except ImportError:
        pass


def __getattr__(name):
    return getattr(_policies, name)
