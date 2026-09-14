"""
HBM + CXL Memory Tiering Simulator for MoE Expert Placement
============================================================
An analytic memory-access-time model, applied to MoE expert routing traces.

WHAT THIS IS
------------
Each expert access is charged `latency + size/bandwidth` against whichever
tier that expert currently lives in. Tier parameters come from config.py:
HBM device latency is measured from our own DRAMSim3 HBM2 run, bandwidths
are published per-stack / per-link figures.

WHAT THIS IS NOT
----------------
This is NOT a cycle-accurate simulation. It does not model bank conflicts,
row-buffer state, refresh, or queueing. DRAMSim3 is used separately to
measure the HBM device parameters that feed this model -- it is not in this
loop. Say so plainly in the report.

TWO PROPERTIES YOU MUST DISCLOSE
--------------------------------
1. BANDWIDTH-DOMINATED. At 352 MB per expert the fixed latency term is
   < 0.01% of an HBM fetch and < 0.002% of a CXL fetch. The CXL/HBM ratio (12.5x) is exactly the inverse
   bandwidth ratio. CXL's +70 ns adder does not show up at this granularity.

2. TWO-VALUED. Every access costs either HBM_TIME_NS or CXL_TIME_NS, so
   average access time is an exact linear function of hit rate:

       avg_time = CXL_TIME - hit_rate * (CXL_TIME - HBM_TIME)

   Every latency curve in this project is therefore the hit-rate curve
   rescaled. Reporting both adds no information; we report hit rate as the
   primary metric and derive time from it.

   (The one exception is when MIGRATION_COST_FACTOR > 0: charging for the
   write-into-HBM on a miss breaks the linearity, which is exactly why that
   sensitivity matters.)

WORKLOAD REGIME
---------------
Charging a full expert fetch per token models BATCH-1 DECODE, where the
model is memory-bound and every routed expert's weights are streamed for
each token. At larger batch sizes the per-step expert union grows and this
model no longer applies -- see batch_sensitivity.py for where it breaks.
"""

import numpy as np
import pandas as pd

from config import (
    NUM_EXPERTS, TOP_K, EXPERT_SIZE_BYTES,
    HBM_LATENCY_NS, HBM_BANDWIDTH_GBPS, CXL_LATENCY_NS, CXL_BANDWIDTH_GBPS,
    HBM_TIME_NS, CXL_TIME_NS, MIGRATION_WRITE_NS, MIGRATION_COST_FACTOR,
    access_time_ns,
)

__all__ = [
    "expert_columns", "rank_experts", "assign_tiers",
    "time_from_hit_rate", "simulate_static", "simulate_lru",
    "simulate_hybrid", "simulate_periodic", "simulate_random_baseline",
    "access_sequence", "token_level_stats",
    "access_time_ns", "EXPERT_SIZE_BYTES",
    "HBM_LATENCY_NS", "HBM_BANDWIDTH_GBPS", "CXL_LATENCY_NS", "CXL_BANDWIDTH_GBPS",
    "HBM_TIME_NS", "CXL_TIME_NS",
]


# ----------------------------------------------------------------------
# Trace helpers
# ----------------------------------------------------------------------
def expert_columns(trace_df):
    """The routed-expert columns, in routing order (expert_1, expert_2, ...)."""
    return sorted(
        [c for c in trace_df.columns if c.startswith("expert_")],
        key=lambda c: int(c.split("_")[1]),
    )


def access_sequence(trace_df, top_k=None):
    """Flatten a trace into chronological access order.

    For top-2 routing, token t produces two accesses: expert_1 then expert_2.
    Returns a flat int array of length n_tokens * top_k.
    """
    cols = expert_columns(trace_df)
    if top_k is not None:
        cols = cols[:top_k]
    return trace_df[cols].to_numpy().reshape(-1)


def rank_experts(trace_df, num_experts=NUM_EXPERTS, top_k=None):
    """Rank experts hottest-first by TOTAL demand across all routed columns.

    BUGFIX: the original code ranked using only `expert_1.value_counts()`
    while the simulator charged for both expert_1 and expert_2. That chose
    the HBM resident set from half the actual demand, and systematically
    handicapped the static and periodic strategies against LRU. Rank on
    what you actually pay for.
    """
    seq = access_sequence(trace_df, top_k=top_k)
    counts = np.bincount(seq, minlength=num_experts)
    # Stable descending sort so ties break by expert id, deterministically.
    return list(np.argsort(-counts, kind="stable"))


def assign_tiers(ranked_experts, num_experts_in_hbm):
    """Top-k of a hottest-first ranking go to HBM, the rest to CXL."""
    return {
        e: ("HBM" if rank < num_experts_in_hbm else "CXL")
        for rank, e in enumerate(ranked_experts)
    }


def time_from_hit_rate(hit_rate):
    """Average access time implied by a hit rate, with no migration cost.

    Exposed deliberately: it makes the two-valued property checkable, and
    lets the dashboard derive latency from hit rate instead of storing a
    second, redundant column.
    """
    return CXL_TIME_NS - hit_rate * (CXL_TIME_NS - HBM_TIME_NS)


def _result(hits, misses, total_time_ns, migrations=0):
    total = hits + misses
    return {
        "hits": int(hits),
        "misses": int(misses),
        "accesses": int(total),
        "hit_rate": hits / total if total else 0.0,
        "hit_rate_pct": hits / total * 100 if total else 0.0,
        "migrations": int(migrations),
        "total_time_ns": float(total_time_ns),
        "avg_time_per_access_ns": float(total_time_ns / total) if total else 0.0,
    }


# ----------------------------------------------------------------------
# Strategy 1: static placement (frozen tier map)
# ----------------------------------------------------------------------
def simulate_static(trace_df, tier_map, top_k=None, num_experts=None):
    """Apply a fixed HBM/CXL assignment to every access. Vectorized."""
    seq = access_sequence(trace_df, top_k=top_k)
    num_experts = num_experts or (max(tier_map) + 1 if tier_map else NUM_EXPERTS)
    in_hbm = np.zeros(num_experts, dtype=bool)
    for e, tier in tier_map.items():
        if tier == "HBM":
            in_hbm[e] = True
    hit_mask = in_hbm[seq]
    hits = int(hit_mask.sum())
    misses = int(hit_mask.size - hits)
    return _result(hits, misses, hits * HBM_TIME_NS + misses * CXL_TIME_NS)


def simulate_random_baseline(trace_df, capacity_k, num_trials, rng, top_k=None,
                             num_experts=NUM_EXPERTS):
    """Control: place a RANDOM set of `capacity_k` experts in HBM.

    This is the experiment that proves the hot/cold heuristic does real work.
    Comparing tiered-vs-untiered only shows "less CXL is faster", which is
    true regardless of which experts you pick. Comparing smart-vs-random at
    the SAME capacity budget isolates the placement decision itself.
    """
    seq = access_sequence(trace_df, top_k=top_k)
    all_experts = np.arange(num_experts)
    times, hit_rates = [], []
    for _ in range(num_trials):
        chosen = rng.choice(all_experts, size=capacity_k, replace=False) if capacity_k else []
        in_hbm = np.zeros(num_experts, dtype=bool)
        in_hbm[list(chosen)] = True
        hit_mask = in_hbm[seq]
        hits = int(hit_mask.sum())
        misses = int(hit_mask.size - hits)
        times.append((hits * HBM_TIME_NS + misses * CXL_TIME_NS) / hit_mask.size)
        hit_rates.append(hits / hit_mask.size)
    return {
        "avg_time_ns_mean": float(np.mean(times)),
        "avg_time_ns_std": float(np.std(times)),
        "hit_rate_mean": float(np.mean(hit_rates)),
    }


# ----------------------------------------------------------------------
# Strategy 2: LRU (recency-adaptive)
# ----------------------------------------------------------------------
def simulate_lru(trace_df, capacity_k, top_k=None, migration_factor=None,
                 num_experts=NUM_EXPERTS):
    """HBM as a fixed-size LRU cache of experts, walked in chronological order.

    migration_factor: fraction of a full HBM write charged on each miss to
    install the expert. 0.0 = the original "migration is free" assumption.
    1.0 = the expert is fully written into HBM before use. The truth is in
    between (a real system overlaps some of it), which is why this is swept
    rather than fixed.
    """
    if migration_factor is None:
        migration_factor = MIGRATION_COST_FACTOR
    migrate_ns = MIGRATION_WRITE_NS * migration_factor

    seq = access_sequence(trace_df, top_k=top_k)
    if capacity_k >= num_experts:
        # Everything fits: treated as preloaded, like static placement at the
        # same budget, so every access is a hit.
        return _result(len(seq), 0, len(seq) * HBM_TIME_NS)
    if capacity_k <= 0:
        # No HBM budget at all: every access goes to CXL, nothing to install.
        return _result(0, len(seq), len(seq) * CXL_TIME_NS)

    cache = {}          # expert_id -> recency counter
    clock = 0
    hits = misses = 0
    total = 0.0
    for e in seq:
        clock += 1
        if e in cache:
            cache[e] = clock
            hits += 1
            total += HBM_TIME_NS
        else:
            misses += 1
            total += CXL_TIME_NS + migrate_ns
            if len(cache) >= capacity_k:
                victim = min(cache, key=cache.get)
                del cache[victim]
            cache[e] = clock
    return _result(hits, misses, total, migrations=misses)


# ----------------------------------------------------------------------
# Strategy 3: hybrid (reserved static slots + LRU remainder)
# ----------------------------------------------------------------------
def simulate_hybrid(trace_df, capacity_k, num_reserved, ranked_experts,
                    top_k=None, migration_factor=None):
    """Pin the hottest `num_reserved` experts; run LRU over the rest."""
    if migration_factor is None:
        migration_factor = MIGRATION_COST_FACTOR
    migrate_ns = MIGRATION_WRITE_NS * migration_factor

    reserved = set(ranked_experts[:num_reserved])
    lru_capacity = capacity_k - num_reserved
    seq = access_sequence(trace_df, top_k=top_k)

    cache = {}
    clock = 0
    hits = misses = migrations = 0
    total = 0.0
    for e in seq:
        clock += 1
        if e in reserved:
            hits += 1
            total += HBM_TIME_NS
        elif e in cache:
            cache[e] = clock
            hits += 1
            total += HBM_TIME_NS
        else:
            misses += 1
            total += CXL_TIME_NS
            if lru_capacity > 0:
                total += migrate_ns
                migrations += 1
                if len(cache) >= lru_capacity:
                    del cache[min(cache, key=cache.get)]
                cache[e] = clock
    return _result(hits, misses, total, migrations=migrations)


# ----------------------------------------------------------------------
# Strategy 4: periodic re-profiling
# ----------------------------------------------------------------------
def simulate_periodic(trace_df, capacity_k, reprofile_interval,
                      num_experts=NUM_EXPERTS, top_k=None):
    """Re-rank experts every `reprofile_interval` TOKENS from the window that
    just finished, and freeze that assignment for the next window.

    NO LOOK-AHEAD: window i is served by the ranking learned from window
    i-1. The first window uses a naive expert-id ordering (a true cold start).

    REPORT THE INTERVAL. The strategy's performance depends strongly on it,
    and short intervals converge toward recency-based behaviour while costing
    MORE bookkeeping than LRU, not less -- which undercuts the usual
    justification for periodic re-profiling. Never quote a periodic number
    without the interval attached.
    """
    cols = expert_columns(trace_df)
    if top_k is not None:
        cols = cols[:top_k]
    tokens = trace_df[cols].to_numpy()
    n_tokens = len(tokens)

    resident = np.zeros(num_experts, dtype=bool)
    resident[:capacity_k] = True  # cold start: no prior information

    hits = 0
    accesses = 0
    for start in range(0, n_tokens, reprofile_interval):
        window = tokens[start:start + reprofile_interval]
        flat = window.reshape(-1)
        hits += int(resident[flat].sum())
        accesses += flat.size

        counts = np.bincount(flat, minlength=num_experts)
        top = np.argsort(-counts, kind="stable")[:capacity_k]
        resident = np.zeros(num_experts, dtype=bool)
        resident[top] = True

    misses = accesses - hits
    return _result(hits, misses, hits * HBM_TIME_NS + misses * CXL_TIME_NS)


# ----------------------------------------------------------------------
# Token-level view (the metric that actually drives decode latency)
# ----------------------------------------------------------------------
def token_level_stats(trace_df, capacity_k, strategy="lru", tier_map=None,
                      migration_factor=None, num_experts=None):
    """Per-ACCESS hit rate flatters top-k routing.

    A token is only served entirely from HBM when ALL of its routed experts
    are resident. With top-2 and a 4-of-8 budget, a 60% per-access hit rate
    can still mean a majority of tokens take at least one CXL trip -- and it
    is the token, not the access, that the user waits for.

    Returns per-access hit rate plus:
      all_resident_pct  -- tokens served fully from HBM
      any_miss_pct      -- tokens that took >= 1 CXL trip
    """
    if migration_factor is None:
        migration_factor = MIGRATION_COST_FACTOR
    cols = expert_columns(trace_df)
    tokens = trace_df[cols].to_numpy()

    if strategy == "static":
        n = num_experts or (max(tier_map) + 1 if tier_map else NUM_EXPERTS)
        in_hbm = np.zeros(n, dtype=bool)
        for e, tier in (tier_map or {}).items():
            if tier == "HBM":
                in_hbm[e] = True
        hit_grid = in_hbm[tokens]
    elif strategy == "lru":
        hit_grid = np.zeros(tokens.shape, dtype=bool)
        if capacity_k <= 0:
            # No HBM budget: nothing is ever resident (and nothing to evict).
            tokens = tokens[:0]
        cache = {}
        clock = 0
        for i, row in enumerate(tokens):
            for j, e in enumerate(row):
                clock += 1
                if e in cache:
                    cache[e] = clock
                    hit_grid[i, j] = True
                else:
                    if len(cache) >= capacity_k:
                        del cache[min(cache, key=cache.get)]
                    cache[e] = clock
    else:
        raise ValueError(f"unknown strategy {strategy!r}")

    all_resident = hit_grid.all(axis=1)
    return {
        "hit_rate_pct": float(hit_grid.mean() * 100),
        "all_resident_pct": float(all_resident.mean() * 100),
        "any_miss_pct": float((~all_resident).mean() * 100),
    }


if __name__ == "__main__":
    import config
    config.summary()
    print("\nSelf-check: two-valued property")
    for hr in (0.0, 0.25, 0.5, 0.75, 1.0):
        print(f"  hit rate {hr:5.0%} -> {time_from_hit_rate(hr):>12,.1f} ns")
