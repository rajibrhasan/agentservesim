
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class SwapBudget:
    load_blocks: int
    store_blocks: int
    new_blocks: int


def measured_swap_limit(forward_s, bytes_per_block, bandwidth_bytes_s):
    """Round the measured overlap window DOWN to complete physical blocks.

    Bandwidth is per rank, using the slowest participating rank and direction.
    Neither synthetic hardware constants nor unmeasured default rates are used.
    """
    if not math.isfinite(forward_s) or forward_s < 0:
        raise ValueError('forward duration must be finite and nonnegative')
    if not isinstance(bytes_per_block, int) or bytes_per_block <= 0:
        raise ValueError('physical block bytes must be a positive integer')
    if not math.isfinite(bandwidth_bytes_s) or bandwidth_bytes_s <= 0:
        raise ValueError('measured bandwidth must be finite and positive')
    return math.floor(forward_s * bandwidth_bytes_s / bytes_per_block)


def plan_swap_budget(limit, free_gpu, free_cpu, load_demand, store_demand, new_demand):
   
    values = (limit, free_gpu, free_cpu, load_demand, store_demand, new_demand)
    if any(not isinstance(v, int) or isinstance(v, bool) or v < 0 for v in values):
        raise ValueError('block counts must be nonnegative integers')
    store_budget = min(limit, max(0, (limit + new_demand - free_gpu) // 2))
    best = (0, 0, 0)
    # For each load count, the maximum feasible store count dominates smaller
    # counts: it can only increase free GPU capacity, our objective.
    for load in range(min(limit, load_demand) + 1):
        store = min(store_budget, store_demand, limit - load, free_cpu + load)
        capacity = free_gpu + store - load
        if capacity < 0:
            continue
        new = min(new_demand, capacity)
        best = max(best, (new, load, store))
    new, load, store = best
    return SwapBudget(load, store, new)
