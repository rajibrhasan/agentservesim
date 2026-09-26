import csv
import tempfile
import unittest
from pathlib import Path

from profiler.aggregate import aggregate_tables
from profiler.core.provenance import check_resume
from profiler.core.hooks.timings import _match_slice


class ProfileReplicasTest(unittest.TestCase):
    def test_arithmetic_latency_mean_and_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = [Path(tmp) / f'{i}.csv' for i in range(2)]
            for path, value in zip(paths, [10, 30]):
                path.write_text(f'layer,tokens,time_us\nattention,16,{value}\n')
            _, rows, stats = aggregate_tables(paths)
            self.assertEqual(rows[0]['time_us'], 20)
            self.assertEqual(stats[0]['samples_us'], [10, 30])
            paths[1].write_text('layer,tokens,time_us\nattention,32,30\n')
            with self.assertRaises(ValueError):
                aggregate_tables(paths)

    def test_reject_duplicate_and_nonfinite(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'a.csv'
            for content in ['attention,16,10\nattention,16,20', 'attention,16,nan']:
                p.write_text('layer,tokens,time_us\n' + content + '\n')
                with self.assertRaises(ValueError):
                    aggregate_tables([p, p])

    def test_resume_requires_same_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            check_resume(tmp, {'gpu_uuid': 'A'})
            check_resume(tmp, {'gpu_uuid': 'A'})
            with self.assertRaises(ValueError):
                check_resume(tmp, {'gpu_uuid': 'B'})

    def test_phi_rotary_alias(self):
        bindings = {'rotary': {'vllm': ['RotaryEmbedding', 'Phi3LongRoPEScaledRotaryEmbedding']}}
        self.assertEqual(_match_slice('Phi3LongRoPEScaledRotaryEmbedding', [], bindings), 'rotary')
        self.assertIsNone(_match_slice('Unknown', [], bindings))


if __name__ == '__main__':
    unittest.main()


def test_tp_may_omit_one_when_tp1_is_already_on_disk():
    """A tp1 sweep that already cost 16 h should not have to be repeated to
    add tp2: job 42633906 was killed after finishing tp1 and before starting
    tp2, and --tp must include 1 forced a full 40 h redo.

    The parser no longer forbids it. The real dependency -- the tp1/ FOLDER
    that tp_stable replication and the MoE table are read from -- is enforced
    in run_full, after any --resume-from copy has landed.
    """
    import pathlib

    # Source-level assertions: profiler/__main__.py uses 3.10 union syntax and
    # profiler.core.runner pulls in vLLM, neither of which this 3.9 test venv
    # can import.
    main_src = pathlib.Path("profiler/__main__.py").read_text()
    assert '--tp must include 1' not in main_src, "the blanket tp1 rule is gone"
    assert 'def _parse_tp' in main_src

    src = pathlib.Path("profiler/core/runner.py").read_text()
    assert "1 not in args.tp_degrees" in src, "run_full must enforce tp1/"
    assert 'variant_root / "tp1"' in src


def test_resume_accepts_a_different_tp_set():
    """Resuming exists to add a degree to what a prior run measured, so the
    TP strings must be allowed to differ; model and hardware still decide
    what the copied CSVs mean and must still match."""
    import pathlib
    src = pathlib.Path("profiler/replicas.py").read_text()
    assert '{"model": a.model, "hardware": a.hardware}' in src
    assert '"tp": a.tp}' not in src.split("resume source arguments differ")[0][-400:]


def test_tp1_falls_back_to_the_installed_bundle():
    """Adding a degree to an ALREADY-INSTALLED model should need no resume
    source: tp1 can come from profiler/perf. A fresh run's output tree never
    has tp1 of its own, which is why the first Phi tp2 submission (42718354)
    would have failed the new check despite tp1 being installed."""
    import pathlib
    src = pathlib.Path("profiler/core/runner.py").read_text()
    assert "_INSTALLED_PERF" in src
    assert "_variant_root(_INSTALLED_PERF, args)" in src
    # Staged into the run output, because replicate_tp_stable reads
    # variant_root/tp1 and nothing else.
    assert "staged installed tp1" in src
    # All three sources named in the failure message, so the fix is obvious.
    assert "--resume-from" in src and "install that model's tp1 bundle first" in src
