"""
Hybrid Strategy: Static Reservation + LRU Cache
==================================================
Combines the two strategies tested so far into one:

  - A small number of HBM slots are PERMANENTLY reserved for the experts
    that are consistently hottest overall (the static approach) -- these
    never get evicted, no matter what happens recently.
  - The REMAINING HBM slots act as a dynamic LRU cache, shared among all
    other experts, adapting continuously to recent access patterns (the
    LRU approach).

Motivation: our two previous experiments showed static wins when the
workload is stable, and LRU wins once the workload shifts. A hybrid
should get the best of both -- permanently protecting the experts that
matter almost everywhere, while still adapting to short-term shifts
with whatever HBM budget is left over.

We test this hybrid two ways:
  1. On the STATIONARY trace (expert_trace.csv) -- sweeping how much of
     the HBM budget goes to the static reservation vs. the LRU portion,
     to find the best split.
  2. On the NON-STATIONARY trace (nonstationary_trace.csv) -- to see if
     a hybrid recovers static's Phase-0 advantage AND avoids static's
     Phase-1/2 collapse, beating both pure strategies overall.
"""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import OrderedDict

import paths
from tier_simulator import (
    access_time_ns, HBM_LATENCY_NS, HBM_BANDWIDTH_GBPS,
    CXL_LATENCY_NS, CXL_BANDWIDTH_GBPS, EXPERT_SIZE_BYTES,
)


def simulate_hybrid(trace_df, capacity_k, num_reserved, ranked_experts):
    """
    capacity_k    : total HBM budget (number of expert "slots")
    num_reserved  : how many of those slots are PERMANENTLY given to the
                    top-`num_reserved` hottest experts (by ranked_experts)
    ranked_experts: experts sorted hottest-first (used to decide the
                    reserved set)

    The remaining (capacity_k - num_reserved) slots form an LRU cache
    shared by every expert NOT in the reserved set.
    """
    reserved = set(ranked_experts[:num_reserved])
    lru_capacity = capacity_k - num_reserved

    cache = OrderedDict()  # LRU cache for the non-reserved experts only
    total_time_ns = 0.0
    hits = 0
    misses = 0

    for row in trace_df[["expert_1", "expert_2"]].itertuples(index=False):
        for expert_id in row:
            if expert_id in reserved:
                # Always an HBM hit -- permanently resident, never evicted
                t = access_time_ns(HBM_LATENCY_NS, HBM_BANDWIDTH_GBPS, EXPERT_SIZE_BYTES)
                hits += 1
            elif expert_id in cache:
                cache.move_to_end(expert_id)
                t = access_time_ns(HBM_LATENCY_NS, HBM_BANDWIDTH_GBPS, EXPERT_SIZE_BYTES)
                hits += 1
            else:
                t = access_time_ns(CXL_LATENCY_NS, CXL_BANDWIDTH_GBPS, EXPERT_SIZE_BYTES)
                misses += 1
                if lru_capacity > 0:
                    cache[expert_id] = True
                    if len(cache) > lru_capacity:
                        cache.popitem(last=False)

            total_time_ns += t

    total = hits + misses
    return {
        "avg_time_per_access_ns": total_time_ns / total,
        "hit_rate_pct": hits / total * 100,
    }


def sweep_hybrid_split(trace_df, num_experts, capacity_k):
    """
    For a fixed total HBM budget (capacity_k), try every possible split
    between "reserved static slots" and "LRU slots" -- from 0 reserved
    (pure LRU) to capacity_k reserved (pure static) -- and see which
    split performs best.
    """
    counts = trace_df["expert_1"].value_counts()
    ranked = counts.index.tolist()
    for e in range(num_experts):
        if e not in ranked:
            ranked.append(e)

    results = []
    for num_reserved in range(capacity_k + 1):
        stats = simulate_hybrid(trace_df, capacity_k, num_reserved, ranked)
        results.append({
            "num_reserved_static": num_reserved,
            "num_lru_slots": capacity_k - num_reserved,
            "avg_time_ns": stats["avg_time_per_access_ns"],
            "hit_rate_pct": stats["hit_rate_pct"],
        })
    return pd.DataFrame(results)


def simulate_hybrid_by_phase(trace_df, capacity_k, num_reserved, ranked_experts):
    """Same hybrid logic as simulate_hybrid, but keeps the cache state
    continuous across the whole (possibly multi-phase) trace, then
    reports average time PER PHASE for comparison against the earlier
    static/LRU non-stationary results."""
    reserved = set(ranked_experts[:num_reserved])
    lru_capacity = capacity_k - num_reserved
    cache = OrderedDict()
    per_row_times = []

    for row in trace_df[["expert_1", "expert_2"]].itertuples(index=False):
        for expert_id in row:
            if expert_id in reserved:
                t = access_time_ns(HBM_LATENCY_NS, HBM_BANDWIDTH_GBPS, EXPERT_SIZE_BYTES)
            elif expert_id in cache:
                cache.move_to_end(expert_id)
                t = access_time_ns(HBM_LATENCY_NS, HBM_BANDWIDTH_GBPS, EXPERT_SIZE_BYTES)
            else:
                t = access_time_ns(CXL_LATENCY_NS, CXL_BANDWIDTH_GBPS, EXPERT_SIZE_BYTES)
                if lru_capacity > 0:
                    cache[expert_id] = True
                    if len(cache) > lru_capacity:
                        cache.popitem(last=False)
            per_row_times.append(t)

    per_token_avg = np.array(per_row_times).reshape(-1, 2).mean(axis=1)
    temp_df = pd.DataFrame({"avg_time_ns": per_token_avg, "phase": trace_df["phase"].values})
    return temp_df.groupby("phase")["avg_time_ns"].mean().reset_index()


def plot_split_sweep(results_df, out_path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(results_df["num_reserved_static"], results_df["avg_time_ns"], marker="o", color="#8172B2")
    best_row = results_df.loc[results_df["avg_time_ns"].idxmin()]
    ax.scatter([best_row["num_reserved_static"]], [best_row["avg_time_ns"]],
               color="red", zorder=5, s=90, label=f"Best split: {int(best_row['num_reserved_static'])} reserved")
    ax.set_xlabel("Number of HBM slots permanently reserved (rest = LRU)")
    ax.set_ylabel("Avg access time (ns)")
    ax.set_title("Hybrid strategy: finding the best static/LRU split")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)


def plot_phase_three_way(static_df, lru_df, hybrid_df, out_path):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    width = 0.25
    x = static_df["phase"]

    ax.bar(x - width, static_df["avg_time_ns"], width, label="Pure Static", color="#4C72B0")
    ax.bar(x, lru_df["avg_time_ns"], width, label="Pure LRU", color="#55A868")
    ax.bar(x + width, hybrid_df["avg_time_ns"], width, label="Hybrid", color="#8172B2")

    ax.set_xlabel("Phase (hot experts change at each boundary)")
    ax.set_ylabel("Avg access time (ns)")
    ax.set_title("Static vs LRU vs Hybrid under a shifting workload")
    ax.set_xticks(x)
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)


if __name__ == "__main__":
    NUM_EXPERTS = 8
    CAPACITY_K = 4

    # --- Experiment 1: best static/LRU split on the STATIONARY trace ---
    stationary_df = pd.read_csv(paths.require_data("expert_trace.csv"))
    split_results = sweep_hybrid_split(stationary_df, NUM_EXPERTS, CAPACITY_K)
    split_results.to_csv(paths.result("hybrid_split_sweep.csv"), index=False)
    plot_split_sweep(split_results, paths.result("hybrid_split_sweep.png"))

    print("=== Hybrid split sweep (stationary trace, HBM budget = 4) ===")
    print(split_results.to_string(index=False))

    # --- Experiment 2: hybrid vs pure static vs pure LRU on the NON-STATIONARY trace ---
    nonstationary_df = pd.read_csv(paths.require_data("nonstationary_trace.csv"))
    counts = nonstationary_df[nonstationary_df["phase"] == 0]["expert_1"].value_counts()
    ranked_from_phase0 = counts.index.tolist()
    for e in range(NUM_EXPERTS):
        if e not in ranked_from_phase0:
            ranked_from_phase0.append(e)

    # Try a middling split: reserve half the budget statically, half as LRU
    HYBRID_RESERVED = CAPACITY_K // 2
    hybrid_phase_results = simulate_hybrid_by_phase(
        nonstationary_df, CAPACITY_K, HYBRID_RESERVED, ranked_from_phase0
    )

    # Re-run pure static and pure LRU on the same trace for a fair 3-way comparison
    from nonstationary_experiment import (
        static_placement_from_first_phase, simulate_static_by_phase, simulate_lru_by_phase
    )
    tier_map = static_placement_from_first_phase(nonstationary_df, NUM_EXPERTS, CAPACITY_K)
    static_phase_results = simulate_static_by_phase(nonstationary_df, tier_map)
    lru_phase_results = simulate_lru_by_phase(nonstationary_df, CAPACITY_K)

    print(f"\n=== Static vs LRU vs Hybrid ({HYBRID_RESERVED} reserved / {CAPACITY_K - HYBRID_RESERVED} LRU), non-stationary trace ===")
    combined = static_phase_results.merge(
        lru_phase_results, on="phase", suffixes=("_static", "_lru")
    ).merge(
        hybrid_phase_results.rename(columns={"avg_time_ns": "avg_time_ns_hybrid"}), on="phase"
    )
    print(combined.to_string(index=False))
    combined.to_csv(paths.result("hybrid_nonstationary_comparison.csv"), index=False)

    plot_phase_three_way(static_phase_results, lru_phase_results, hybrid_phase_results,
                          paths.result("hybrid_nonstationary_comparison.png"))

    print(f"\nWrote into {paths.RESULTS_DIR}:"
          f"\n       hybrid_split_sweep.csv/.png"
          f"\n       hybrid_nonstationary_comparison.csv/.png")
