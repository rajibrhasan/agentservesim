"""Engine subprocess output must survive the startup capture context."""
import os
import subprocess
import sys

from bench.core.logger import capture_stdio


def test_child_output_after_startup_is_persisted(tmp_path):
    path = tmp_path / 'engine.log'
    with capture_stdio(str(path)):
        child = subprocess.Popen([sys.executable, '-c',
            "import sys; sys.stdin.readline(); print('late engine error', file=sys.stderr, flush=True)"],
            stdin=subprocess.PIPE)
        os.write(1, b'startup output\n')
    child.communicate(b'go\n', timeout=10)
    assert child.returncode == 0
    assert 'startup output' in path.read_text()
    assert 'late engine error' in path.read_text()
