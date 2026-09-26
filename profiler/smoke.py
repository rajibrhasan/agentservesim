"""Small GPU compatibility check before committing to a full profile sweep."""
import argparse
import json
import logging
from pathlib import Path


def main():
    from profiler.core import logger
    logger.configure(logging.DEBUG)
    from profiler.core.config import ProfileArgs, load_architecture
    from profiler.core.engine import spin_up, spin_down, probe_limits
    from profiler.core.categories import categories_for
    from profiler.core.hooks.batch import Shot

    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--tp", default="1")
    parser.add_argument("--output", required=True)
    cli = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    cfg = json.loads((root / "configs/model" / (cli.model + ".json")).read_text())
    arch = load_architecture(root / "profiler/models" / (cfg["model_type"] + ".yaml"))
    results = []
    for tp in map(int, cli.tp.split(",")):
        args = ProfileArgs(architecture=cfg["model_type"], model=cli.model,
                           hardware="smoke", model_config=cfg, dtype="bfloat16",
                           max_num_batched_tokens=512, max_num_seqs=8)
        llm, _, tmp = spin_up(args, tp)
        try:
            limits = probe_limits(llm)
            # The model's advertised context limit may exceed the engine's
            # accepted limit (for example, Phi's original LongRoPE window).
            # Leave one position for the token sampled after the forward.
            history = min(8192, limits.max_model_len - 256 - 1,
                          (limits.num_cache_tokens - 256 - 5 * 16) // 5)
            if history < 512:
                raise RuntimeError('Engine capacity is too small for the smoke test')
            logger.info('Smoke TP=%d: engine max_model_len=%d, mixed history=%d',
                        tp, limits.max_model_len, history)
            for category in categories_for(arch, tp):
                shots = {
                    "dense": [Shot.dense(16), Shot.dense(512)],
                    "per_sequence": [Shot.per_sequence(4)],
                    "attention": [Shot.attention(0, 0, 4, 512),
                                  Shot.attention(256, history, 4, history)],
                    "moe": [Shot.moe(16, cfg.get("num_experts_per_tok", 2)),
                            Shot.moe(512, cfg.get("num_local_experts", 16))],
                }[category.name]
                for shot in shots:
                    samples = llm.collective_rpc("fire", args=(
                        shot.as_dict(), category.catalog_slice(arch), category.name, 5))[0]
                    results.append({"tp": tp, "category": category.name,
                                    "shot": shot.as_dict(), "samples": samples})
        finally:
            spin_down(llm, tmp)
    Path(cli.output).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
