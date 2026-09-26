from itertools import product

import pytest

from policies.infercept_budget import measured_swap_limit, plan_swap_budget


def test_budget_matches_exhaustive_feasible_allocations():
    # Independent enumeration catches constrained CPU/GPU pools, zero demand,
    # and cases where assigning the whole budget to one direction is inferior.
    for args in product(range(4), repeat=6):
        limit, gpu, cpu, loads, stores, demand = args
        feasible = [(new, load, store)
                    for load in range(loads + 1)
                    for store in range(stores + 1)
                    for new in range(demand + 1)
                    if load + store <= limit and store <= cpu + load
                    and store <= max(0, (limit + demand - gpu) // 2)
                    and load + new <= gpu + store]
        expected = max(feasible)
        result = plan_swap_budget(*args)
        assert (result.new_blocks, result.load_blocks, result.store_blocks) == expected


def test_free_gpu_capacity_suppresses_proactive_swap():
    assert plan_swap_budget(16, 100, 100, 0, 100, 8).store_blocks == 0
    assert plan_swap_budget(16, 24, 100, 0, 100, 8).store_blocks == 0
    assert plan_swap_budget(16, 8, 100, 0, 100, 8).store_blocks == 8
    assert plan_swap_budget(16, 0, 100, 0, 100, 8).store_blocks == 12


def test_free_gpu_capacity_still_allows_restores():
    result = plan_swap_budget(16, 100, 100, 6, 100, 8)
    assert result.load_blocks == 6
    assert result.store_blocks == 0


def test_full_host_requires_load_before_store_can_reuse_space():
    result = plan_swap_budget(4, 0, 0, 2, 2, 2)
    assert result.load_blocks == result.store_blocks == 2
    assert result.new_blocks == 0


def test_measured_budget_floors_incomplete_pages():
    assert measured_swap_limit(0.01, 1600, 319999) == 1
    assert measured_swap_limit(0.01, 1600, 320000) == 2
    assert measured_swap_limit(0, 1600, 320000) == 0
    with pytest.raises(ValueError):
        measured_swap_limit(1, 1600, float('nan'))
    with pytest.raises(ValueError):
        plan_swap_budget(1, 0, 0, -1, 0, 0)
