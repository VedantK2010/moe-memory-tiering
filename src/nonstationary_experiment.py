"""
Non-Stationary Hot-Expert Experiment
======================================
Tests the hypothesis from our previous result: static (frequency-based)
placement wins over LRU (recency-based) ONLY because our earlier trace
had one fixed, unchanging popularity distribution throughout. Real
workloads may not be so stable -- e.g. a model serving different tasks
or topics over time could see WHICH experts are "hot" change partway
through usage.

This script:
  1. Generates a trace split into several PHASES, each with its own,
     different set of hot experts (simulating a shift in workload).
  2. Computes a "static" placement the way a real deployment realistically
     would -- by profiling usage ONCE (using only the first phase) and
     then freezing that decision for the rest of the run. This is the
     fair real-world framing: you don't get to see the future when you
     decide a static placement.
  3. Runs the SAME LRU cache logic from tier_simulator.py, which keeps
     adapting continuously and has no such blind spot.
  4. Compares average access time for both strategies, broken down BY
     PHASE, to see whether LRU starts winning once the distribution
     shifts away from what the static strategy was frozen on.
"""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import paths
from tier_simulator import (
    access_time_ns, assign_tiers, simulate, simulate_lru_cache,
    HBM_LATENCY_NS, HBM_BANDWIDTH_GBPS, CXL_LATENCY_NS, CXL_BANDWIDTH_GBPS,
    EXPERT_SIZE_BYTES,
)

NUM_EXPERTS = 8
TOP_K = 2
TOKENS_PER_PHASE = 20_000
NUM_PHASES = 3
P_REPEAT_TARGET = 0.27
SEED = 7


def generate_nonstationary_trace(num_experts, top_k, tokens_per_phase, num_phases,
                                  p_repeat_target, seed, skew_alpha=6):
    """
    Generate a trace made of several back-to-back phases, each with its
    OWN popularity distribution. Locality (temporal repeat behavior) is
    preserved within each phase using the same calibrated approach as
    generate_trace.py.

    IMPORTANT: rather than drawing an independent random distribution
    per phase (which can coincidentally keep the same expert "hot" by
    chance across phases -- not a real test of adaptation), we generate
    ONE skewed shape via Dirichlet and then ROTATE it by a fixed amount
    each phase. This guarantees the identity of the hot expert(s)
    actually changes at each phase boundary, which is the scenario we
    want to test.
    """
    rng = np.random.default_rng(seed)
    all_rows = []
    phase_labels = []
    phase_base_probs = []

    base_shape = rng.dirichlet(alpha=[skew_alpha] * num_experts)
    roll_amount = max(1, num_experts // num_phases)

    prev_first = None
    for phase in range(num_phases):
        base_probs = np.roll(base_shape, shift=phase * roll_amount)
        phase_base_probs.append(base_probs)
        collision_prob = float(np.sum(base_probs ** 2))
        p_forced = max(0.0, (p_repeat_target - collision_prob) / (1 - collision_prob))

        if prev_first is None:
            prev_first = int(rng.choice(num_experts, p=base_probs))

        for _ in range(tokens_per_phase):
            if rng.random() < p_forced:
                first = prev_first
            else:
                first = int(rng.choice(num_experts, p=base_probs))

            remaining = [e for e in range(num_experts) if e != first]
            remaining_probs = base_probs[remaining] / base_probs[remaining].sum()
            rest = rng.choice(remaining, size=top_k - 1, replace=False, p=remaining_probs)

            row = [first] + list(rest)
            all_rows.append(row)
            phase_labels.append(phase)
            prev_first = first

    df = pd.DataFrame(all_rows, columns=[f"expert_{i+1}" for i in range(top_k)])
    df.insert(0, "token_id", np.arange(len(df)))
    df["phase"] = phase_labels
    return df, phase_base_probs


def static_placement_from_first_phase(trace_df, num_experts, capacity_k):
    """
    Simulate a REALISTIC static deployment: profile usage using ONLY the
    first phase (as if this were done once, early, before the workload
    shifted), rank experts by that snapshot, and freeze that HBM/CXL
    assignment for the entire rest of the trace -- including phases the
    static strategy never got to see.
    """
    first_phase_df = trace_df[trace_df["phase"] == 0]
    counts = first_phase_df["expert_1"].value_counts()
    ranked = counts.index.tolist()
    for e in range(num_experts):
        if e not in ranked:
            ranked.append(e)

    tier_map = assign_tiers(ranked, capacity_k)
    return tier_map


def simulate_static_by_phase(trace_df, tier_map):
    """Run the static (frozen) tier assignment across the WHOLE trace, but
    report results broken down per phase so we can see where it starts
    to fail."""
    expert_cols = ["expert_1", "expert_2"]
    results = []
    for phase, group in trace_df.groupby("phase"):
        stats = simulate(group[expert_cols], tier_map)
        results.append({"phase": phase, "avg_time_ns": stats["avg_time_per_access_ns"]})
    return pd.DataFrame(results)


def simulate_lru_by_phase(trace_df, capacity_k):
    """Run the LRU cache continuously across the whole trace (state
    persists across phase boundaries, exactly as it would in a real
    deployment -- the cache doesn't know or care about our phase labels),
    but report results grouped by phase afterward for comparison."""
    from collections import OrderedDict

    cache = OrderedDict()
    per_row_times = []

    for row in trace_df[["expert_1", "expert_2"]].itertuples(index=False):
        for expert_id in row:
            if expert_id in cache:
                cache.move_to_end(expert_id)
                t = access_time_ns(HBM_LATENCY_NS, HBM_BANDWIDTH_GBPS, EXPERT_SIZE_BYTES)
            else:
                t = access_time_ns(CXL_LATENCY_NS, CXL_BANDWIDTH_GBPS, EXPERT_SIZE_BYTES)
                cache[expert_id] = True
                if len(cache) > capacity_k:
                    cache.popitem(last=False)
            per_row_times.append(t)

    # per_row_times has 2 entries per token (expert_1, expert_2); average
    # them per token, then attach phase labels and group
    per_token_avg = np.array(per_row_times).reshape(-1, 2).mean(axis=1)
    temp_df = pd.DataFrame({"avg_time_ns": per_token_avg, "phase": trace_df["phase"].values})
    return temp_df.groupby("phase")["avg_time_ns"].mean().reset_index()


def plot_phase_comparison(static_df, lru_df, capacity_k, out_path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    width = 0.35
    x = static_df["phase"]

    ax.bar(x - width / 2, static_df["avg_time_ns"], width,
           label="Static (frozen after Phase 0)", color="#4C72B0")
    ax.bar(x + width / 2, lru_df["avg_time_ns"], width,
           label="LRU (continuously adaptive)", color="#55A868")

    ax.set_xlabel("Phase (hot experts change at each boundary)")
    ax.set_ylabel("Avg access time (ns)")
    ax.set_title(f"Static vs LRU under a shifting workload (HBM capacity = {capacity_k} experts)")
    ax.set_xticks(x)
    ax.legend()
    ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)


if __name__ == "__main__":
    trace_df, phase_base_probs = generate_nonstationary_trace(
        NUM_EXPERTS, TOP_K, TOKENS_PER_PHASE, NUM_PHASES, P_REPEAT_TARGET, SEED
    )
    trace_df.to_csv(paths.data("nonstationary_trace.csv"), index=False)

    print("Hot-expert distribution per phase (target popularity):")
    for i, probs in enumerate(phase_base_probs):
        hottest = int(np.argmax(probs))
        print(f"  Phase {i}: hottest expert = {hottest}, distribution = {np.round(probs, 3).tolist()}")

    CAPACITY_K = 4  # test at a mid-range HBM budget, matches our earlier example

    tier_map = static_placement_from_first_phase(trace_df, NUM_EXPERTS, CAPACITY_K)
    print(f"\nStatic placement decided ONCE from Phase 0 only: {tier_map}")

    static_results = simulate_static_by_phase(trace_df, tier_map)
    lru_results = simulate_lru_by_phase(trace_df, CAPACITY_K)

    print("\n=== Static (frozen after Phase 0) -- avg time per phase ===")
    print(static_results.to_string(index=False))

    print("\n=== LRU (continuously adaptive) -- avg time per phase ===")
    print(lru_results.to_string(index=False))

    merged = static_results.merge(lru_results, on="phase", suffixes=("_static", "_lru"))
    merged["lru_advantage_pct"] = (
        (merged["avg_time_ns_static"] - merged["avg_time_ns_lru"]) / merged["avg_time_ns_static"] * 100
    )
    merged.to_csv(paths.result("nonstationary_comparison.csv"), index=False)
    print("\n=== LRU advantage over static, per phase ===")
    print(merged[["phase", "lru_advantage_pct"]].to_string(index=False))

    plot_phase_comparison(static_results, lru_results, CAPACITY_K,
                          paths.result("nonstationary_comparison.png"))
    print(f"\nWrote: {paths.data('nonstationary_trace.csv')}"
          f"\n       {paths.result('nonstationary_comparison.csv')}"
          f"\n       {paths.result('nonstationary_comparison.png')}")
