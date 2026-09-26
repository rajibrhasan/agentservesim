import os
import subprocess
from time import time
from .request import *
from .logger import get_logger
from .run_paths import input_path

logger = get_logger("GraphGenerator")

# The Chakra LLM converter used to run as ``python -m
# chakra.src.converter.converter`` once per batch per iteration.  Measured on
# a warm node, that fork cost ~112 ms, of which ~109 ms was interpreter
# startup plus re-importing chakra's protobuf schema (et_def_pb2 alone is
# ~33 ms) and only ~3 ms was the conversion.  At the ~6.7 conversions/s this
# loop sustains, it was ~75% of total wall clock, with ASTRA-Sim blocked in
# pipe_read for all of it.  Doing the same work in-process is equivalent:
# LLMConverter keeps every piece of per-conversion state on the instance, so
# a fresh instance per call matches a fresh process, and it context-manages
# every file it opens.  The `cwd` the subprocess used was decorative --
# `chakra` resolves off PYTHONPATH, so both paths load the same module file.
_CONVERTER_CLS = None
_CONVERTER_UNAVAILABLE = False


def _llm_converter_cls():
    """Return chakra's LLMConverter, or None to fall back to the CLI."""
    global _CONVERTER_CLS, _CONVERTER_UNAVAILABLE
    if _CONVERTER_CLS is None and not _CONVERTER_UNAVAILABLE:
        try:
            from chakra.src.converter.llm_converter import LLMConverter
        except Exception as exc:
            _CONVERTER_UNAVAILABLE = True
            logger.warning(
                "in-process Chakra converter unavailable (%s); falling back "
                "to the converter CLI", exc)
        else:
            _CONVERTER_CLS = LLMConverter
    return _CONVERTER_CLS

def generate_graph(batch, hardware, num_npus, node_id=0, instance_id=0, npu_offset=0, enable_local_offloading=False, event=False, workload_name=None, inputs_root=None, cleanup_trace=True):

    cwd = os.getcwd()
    chakra = os.path.join(cwd, "extern/graph_frontend/chakra")
    if inputs_root is None:
        inputs_root = os.path.join(cwd, "inputs")

    if event:
        file_name = 'event_handler'
    else:
        file_name = f'{hardware}/{batch.model}/instance{instance_id}_batch{batch.batch_id}'

    # For DP groups, all instances write .et files to a shared workload folder
    output_name = workload_name if workload_name else file_name

    trace_path = input_path(inputs_root, "trace", f"{file_name}.txt")
    output_path = input_path(inputs_root, "workload", output_name, "llm")
    workload_dir = os.path.dirname(output_path)
    os.makedirs(workload_dir, exist_ok=True)

    cmd = [
        'python', '-m', 'chakra.src.converter.converter', 'LLM',
        '--input', trace_path,
        '--output', output_path,
        '--num-npus', str(num_npus),
        '--npu-offset', str(npu_offset),
    ]

    if enable_local_offloading:
        cmd.append('--local-offloading')

    logger.debug("Generating graph with command: %s", " ".join(cmd), extra={"node_id": node_id, "instance_id": instance_id})

    converter_cls = _llm_converter_cls()
    if converter_cls is not None:
        converter_cls(
            trace_path,
            output_path,
            num_npus,
            npu_offset,
            enable_local_offloading,
        ).convert()
    else:
        subprocess.run(cmd, cwd=chakra, text=True, check=True)
    if cleanup_trace:
        try:
            os.remove(trace_path)
        except FileNotFoundError:
            pass
    # Remember this batch's workload files so the scheduler can delete them
    # once every NPU has finished the batch (ASTRA-Sim reads them lazily
    # while the batch runs; nothing reads them afterwards). DP groups share
    # one workload folder across instances, so those are left to the
    # end-of-run cleanup.
    if not event and workload_name is None:
        import glob as _glob
        batch.et_files = sorted(_glob.glob(f"{output_path}.*.et"))
    return
