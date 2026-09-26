"""The benchmark driver and engine gate must load the same staged candidate."""
import pathlib
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]


class StagedPolicyTests(unittest.TestCase):
    def test_real_gate_candidate_initializes_without_an_engine(self):
        with tempfile.TemporaryDirectory() as scratch:
            staged = pathlib.Path(scratch) / 'harness'
            shutil.copytree(ROOT / 'harness', staged)
            shutil.copyfile(ROOT / 'evolve/gate_only_scheduling.py',
                            staged / 'evolved_scheduling.py')
            result = subprocess.run([sys.executable, '-c', '''
import asyncio, sys
from bench.core.policy_driver import PolicyDriver, TupleConfig, engine_flags
async def check():
    cfg = TupleConfig(retention='ttl', tau_s=2.0, scheduling='evolved',
                      harness_root=sys.argv[1])
    driver = PolicyDriver(cfg, [], asyncio.get_running_loop())
    from harness.evolved_scheduling import EvolvedScheduling
    assert type(driver.scheduling_exec.policy) is EvolvedScheduling
    assert engine_flags('ttl', 'evolved')['scheduling_policy'] == 'priority'
    assert driver.retention_exec.policy.tau_s == 2.0
    driver.retention_exec.finish()
    driver._worker.shutdown(wait=True)
asyncio.run(check())
''', scratch], cwd=ROOT, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main()
