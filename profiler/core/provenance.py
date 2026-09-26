"""Prevent profile resumption on different hardware or measurement settings."""
import dataclasses
import hashlib
import importlib.metadata
import json
from pathlib import Path
import socket


def check_resume(root, fingerprint, force=False):
    root = Path(root)
    path = root / 'provenance.json'
    if path.exists():
        previous = json.loads(path.read_text())
        if previous != fingerprint and not force:
            raise ValueError('Profile provenance differs; use a fresh output root')
    elif root.exists() and any(root.glob('tp*/*.csv')) and not force:
        raise ValueError('Existing profile has no provenance; use a fresh output root')
    root.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(fingerprint, indent=2, sort_keys=True))


def fingerprint(args, arch_path):
    import torch
    settings = dataclasses.asdict(args)
    for key in ('force', 'only_skew', 'skip_skew'):
        settings.pop(key, None)
    source = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(source.rglob('*.py')):
        if 'v0' not in path.parts:
            digest.update(str(path.relative_to(source)).encode())
            digest.update(path.read_bytes())
    return {'hostname': socket.gethostname(),
            'gpu_uuid': str(torch.cuda.get_device_properties(0).uuid),
            'cuda': torch.version.cuda,
            'versions': {n: importlib.metadata.version(n) for n in ('vllm', 'torch', 'transformers')},
            'architecture_sha256': hashlib.sha256(Path(arch_path).read_bytes()).hexdigest(),
            'profiler_sha256': digest.hexdigest(), 'settings': settings}
