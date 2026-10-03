

import getpass
import os

HERE = os.path.dirname(os.path.abspath(__file__))

#: The repository root. Since the 2026-09-12 unification this single directory
#: holds both the engine (serving/, astra-sim/, bench/) and everything built on
#: it (harness/, runtime/, evolve/) -- they used to be two checkouts.
REPO = os.environ.get("EVOLVE_SIM_REPO", os.path.dirname(HERE))

#: Kept as a name because a great many call sites say AS_ROOT, but it is now
#: the same directory as REPO. It was `agentservesim/` when the policy contract
#: lived in a separate repository from the engine.
AS_ROOT = REPO

#: Container image and the python dependencies bind-mounted into it. The
#: simulator is never run from the host interpreter: ASTRA-Sim is a compiled
#: backend and the image is what pins it.
def _site(name, default):
    """A deployment-specific path, without putting anyone's username in the source.
    Resolution order: environment, then an OPTIONAL and gitignored
    `runtime/site.py`, then a sibling of the repository."""
    env = os.environ.get("EVOLVE_" + name)
    if env:
        return env
    try:
        from . import site as _s            # gitignored, optional
        v = getattr(_s, name, None)
        if v:
            return v
    except ImportError:
        pass
    return default


#: Scratch root: container image, python deps, and the large trace sets.
#: Override with EVOLVE_MAS, or create runtime/site.py with MAS = "...".
MAS = _site("MAS", os.path.join(os.path.dirname(REPO), "masservingsim"))
SIF = os.path.join(MAS, "sifs", "sim.sif")
PYDEPS = os.path.join(MAS, "sim_pydeps")

#: Run scratch. /dev/shm by default: ASTRA-Sim churns .et files fast enough to
#: exhaust the home inode quota, which crashes jobs hours in.
SCRATCH = os.environ.get(
    "EVOLVE_SCRATCH", "/dev/shm/evolve_" + getpass.getuser())

#: Engine settings held constant across every run. Changing one of these
#: changes what every recorded number means, so they are here rather than at
#: a call site.
ENGINE = ["--dtype", "bfloat16", "--block-size", "16",
          "--max-num-seqs", "128", "--max-num-batched-tokens", "16384",
          "--num-reqs", "0", "--log-level", "WARNING"]
