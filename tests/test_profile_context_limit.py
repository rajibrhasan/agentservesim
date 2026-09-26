import json
import ast
import argparse
import copy
import logging
from dataclasses import replace
from pathlib import Path

import pytest

from profiler.core.config import ProfileArgs, HOST_ENGINE_DEFAULTS, SHARD_FIELDS


def load_functions(path, names, namespace):
    # Test actual host-only configuration functions without importing vLLM
    # or initializing its GPU/platform discovery on the login node.
    tree = ast.parse(Path(path).read_text())
    tree.body = [ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)] + [
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    exec(compile(ast.fix_missing_locations(tree), path, 'exec'), namespace)
    return namespace


fuse_engine_kwargs = load_functions('profiler/core/engine.py',
    {'_deep_merge', '_profile_engine_overrides', 'fuse_engine_kwargs'},
    dict(copy=copy, HOST_ENGINE_DEFAULTS=HOST_ENGINE_DEFAULTS, SHARD_FIELDS=SHARD_FIELDS,
         log=logging.getLogger(__name__)))['fuse_engine_kwargs']
build_parser = load_functions('profiler/__main__.py', {'_add_common_flags', 'build_parser'},
    dict(argparse=argparse, logging=logging, Path=Path, PERF_DIR=Path('profiler/perf'),
         ARCH_DIR=Path('profiler/models'), MODEL_CONFIG_DIR=Path('configs/model')))['build_parser']


def test_explicit_phi_context_reaches_engine_and_preserves_tp_emulation():
    config = json.loads(Path('configs/model/microsoft/Phi-3.5-MoE-instruct.json').read_text())
    args = ProfileArgs(architecture='phimoe', model='microsoft/Phi-3.5-MoE-instruct',
                       hardware='test', tp_degrees=[1, 2], model_config=config,
                       max_model_len=131072, max_num_batched_tokens=16384, max_num_seqs=128)
    for tp in (1, 2):
        kwargs = fuse_engine_kwargs(args, tp)
        assert kwargs['max_model_len'] == 131072
        assert kwargs['tensor_parallel_size'] == 1
        assert kwargs['max_num_batched_tokens'] == 16384 + 128
    args = replace(args, max_model_len=0)
    with pytest.raises(ValueError, match='positive'):
        fuse_engine_kwargs(args, 1)


def test_cli_accepts_context_limit_and_user_alias():
    for flag in ('--max-model-len', '--max-len-tokens'):
        ns = build_parser().parse_args(['profile', 'microsoft/Phi-3.5-MoE-instruct',
                                       '--hardware', 'B200', flag, '131072'])
        assert ns.max_model_len == 131072
