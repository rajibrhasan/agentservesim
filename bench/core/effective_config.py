"""Serialize resolved engine constraints, including each instance's KV pool."""


def effective_engine_configs(engines):
    result = []
    for instance, engine in enumerate(engines):
        cfg = engine.vllm_config
        cache, sched = cfg.cache_config, cfg.scheduler_config
        result.append({
            'instance': instance,
            'kv_cache_tokens': int(cache.num_gpu_blocks) * int(cache.block_size),
            'kv_block_size': int(cache.block_size),
            'max_model_len': int(cfg.model_config.max_model_len),
            'max_num_seqs': int(sched.max_num_seqs),
            'max_num_batched_tokens': int(sched.max_num_batched_tokens),
            'async_scheduling': bool(sched.async_scheduling),
            'scheduler_reserve_full_isl': bool(sched.scheduler_reserve_full_isl),
        })
    return result
