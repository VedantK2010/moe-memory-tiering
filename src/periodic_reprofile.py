"""
Periodic Re-Profiling Strategy
=================================
A middle ground between the two extremes we've already tested:

  - Pure Static: profile once, freeze forever (fails once workload shifts)
  - Pure LRU: adapt on EVERY single access (maximally reactive, but never
    benefits from "knowing" an expert is reliably hot -- it can just as
    easily evict something important as something unimportant)

Periodic re-profiling: every N tokens, look back at what was actually
used in the window that just finished, re-rank experts by that recent
usage, and use that ranking as a FROZEN static assignment until the
next re-profiling point. This is a realistic, deployable middle ground
-- a real system could plausibly re-run a cheap profiling pass every
few thousand requests without the overhead of a fully dynamic cache.

IMPORTANT (no look-ahead): the ranking used for window i is always
computed from window (i-1)'s actual observed accesses -- never from the
window it's being applied to. The very first window has no prior data
to learn from, so it uses a naive, uninformed default ordering (plain
expert-ID order) -- this reflects a real cold-start with zero prior
information, not a lucky guess.
"""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import paths
from tier_simulator import access_time_ns, assign_tiers, simulate, \
    HBM_LATENCY_NS, HBM_BANDWIDTH_GBPS, CXL_LATENCY_NS, CXL_BANDWIDTH_GBPS, EXPERT_SIZE_BYTES


def simulate_periodic_reprofile(trace_df, capacity_k, num_experts, reprofile_interval):
    """
    Walk the trace in chunks of `reprofile_interval` tokens. Each chunk
    uses a tier assignment based on the PREVIOUS chunk's observed usage
    (the first chunk uses a naive, uninformed default). Returns overall
    per-token times, tagged with phase (if present) for later grouping.
    """
    expert_cols = ["expert_1", "expert_2"]
    n_tokens = len(trace_df)

    # Cold start: no prior data, so no informed ranking is possible yet.
    current_ranking = list(range(num_experts))
    current_tier_map = assign_tiers(current_ranking, capacity_k)

    per_token_times = []

    for start in range(0, n_tokens, reprofile_interval):
        end = min(start + reprofile_interval, n_tokens)
        window = trace_df.iloc[start:end]

        # Apply the CURRENT tier map (decided from the previous window)
        # to every access in this window.
        for row in window[expert_cols].itertuples(index=False):
            row_times = []
            for expert_id in row:
                tier = current_tier_map[expert_id]
                if tier == "HBM":
                    t = access_time_ns(HBM_LATENCY_NS, HBM_BANDWIDTH_GBPS, EXPERT_SIZE_BYTES)
                else:
                    t = access_time_ns(CXL_LATENCY_NS, CXL_BANDWIDTH_GBPS, EXPERT_SIZE_BYTES)
                row_times.append(t)
            per_token_times.append(np.mean(row_times))

        # Now that this window is finished, profile IT, and use that
        # ranking for the NEXT window.
        counts = window["expert_1"].value_counts()
        new_ranking = counts.index.tolist()
        for e in range(num_experts):
            if e not in new_ranking:
                new_ranking.append(e)
        current_ranking = new_ranking
        current_tier_map = assign_tiers(current_ranking, capacity_k)

    result_df = pd.DataFrame({"avg_time_ns": per_token_times})
    if "phase" in trace_df.columns:
        result_df["phase"] = trace_df["phase"].values
    return result_df


def compare_reprofile_intervals(trace_df, capacity_k, num_experts, intervals):
    """Try several re-profiling frequencies and compare overall average time."""
    results = []
    for interval in intervals:
        per_token = simulate_periodic_reprofile(trace_df, capacity_k, num_experts, interval)
        results.append({
            "reprofile_interval": interval,
            "overall_avg_time_ns": per_token["avg_time_ns"].mean(),
        })
    return pd.DataFrame(results)


def plot_interval_comparison(results_df, static_baseline, lru_baseline, hybrid_baseline, out_path):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(results_df["reprofile_interval"], results_df["overall_avg_time_ns"],
            marker="o", color="#C44E52", label="Periodic re-profile")
    ax.axhline(static_baseline, color="#4C72B0", linestyle="--", label="Pure Static")
    ax.axhline(lru_baseline, color="#55A868", linestyle="--", label="Pure LRU")
    ax.axhline(hybrid_baseline, color="#8172B2", linestyle="--", label="Hybrid (static+LRU)")
    ax.set_xlabel("Re-profiling interval (tokens between re-ranks)")
    ax.set_ylabel("Overall avg access time (ns)")
    ax.set_title("Periodic re-profiling vs. the other three strategies")
    ax.set_xscale("log")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)


def plot_phase_four_way(static_df, lru_df, hybrid_df, reprofile_df, out_path):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    width = 0.2
    x = static_df["phase"]

    reprofile_by_phase = reprofile_df.groupby("phase")["avg_time_ns"].mean().reset_index()

    ax.bar(x - 1.5 * width, static_df["avg_time_ns"], width, label="Pure Static", color="#4C72B0")
    ax.bar(x - 0.5 * width, lru_df["avg_time_ns"], width, label="Pure LRU", color="#55A868")
    ax.bar(x + 0.5 * width, hybrid_df["avg_time_ns"], width, label="Hybrid", color="#8172B2")
    ax.bar(x + 1.5 * width, reprofile_by_phase["avg_time_ns"], width, label="Periodic Re-profile", color="#C44E52")

    ax.set_xlabel("Phase (hot experts change at each boundary)")
    ax.set_ylabel("Avg access time (ns)")
    ax.set_title("All four strategies under a shifting workload")
    ax.set_xticks(x)
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)


if __name__ == "__main__":
    NUM_EXPERTS = 8
    CAPACITY_K = 4

    nonstationary_df = pd.read_csv(paths.require_data("nonstationary_trace.csv"))

    # --- Try several re-profiling frequencies ---
    intervals = [500, 1000, 2000, 5000, 10000, 20000]
    interval_results = compare_reprofile_intervals(nonstationary_df, CAPACITY_K, NUM_EXPERTS, intervals)
    interval_results.to_csv(paths.result("reprofile_interval_sweep.csv"), index=False)
    print("=== Overall avg access time vs re-profiling interval ===")
    print(interval_results.to_string(index=False))

    # --- Load the three earlier strategies' results for a fair overall comparison ---
    prior = pd.read_csv(paths.RESULTS_DIR / "hybrid_nonstationary_comparison.csv")
    static_overall = prior["avg_time_ns_static"].mean()
    lru_overall = prior["avg_time_ns_lru"].mean()
    hybrid_overall = prior["avg_time_ns_hybrid"].mean()

    print(f"\nPure Static overall avg: {static_overall:.1f} ns")
    print(f"Pure LRU overall avg:    {lru_overall:.1f} ns")
    print(f"Hybrid overall avg:      {hybrid_overall:.1f} ns")

    plot_interval_comparison(interval_results, static_overall, lru_overall, hybrid_overall,
                              paths.result("reprofile_interval_sweep.png"))

    # --- Best interval: per-phase breakdown vs the other three strategies ---
    best_interval = int(interval_results.loc[interval_results["overall_avg_time_ns"].idxmin(), "reprofile_interval"])
    print(f"\nBest re-profiling interval: every {best_interval} tokens")

    reprofile_best = simulate_periodic_reprofile(nonstationary_df, CAPACITY_K, NUM_EXPERTS, best_interval)

    # Rebuild the static/LRU/hybrid per-phase dataframes to plot alongside
    static_phase = prior[["phase", "avg_time_ns_static"]].rename(columns={"avg_time_ns_static": "avg_time_ns"})
    lru_phase = prior[["phase", "avg_time_ns_lru"]].rename(columns={"avg_time_ns_lru": "avg_time_ns"})
    hybrid_phase = prior[["phase", "avg_time_ns_hybrid"]].rename(columns={"avg_time_ns_hybrid": "avg_time_ns"})

    plot_phase_four_way(static_phase, lru_phase, hybrid_phase, reprofile_best,
                         paths.result("four_strategy_comparison.png"))

    reprofile_best.groupby("phase")["avg_time_ns"].mean().reset_index().to_csv(
        paths.result("reprofile_best_by_phase.csv"), index=False
    )

    print(f"\nWrote into {paths.RESULTS_DIR}:"
          f"\n       reprofile_interval_sweep.csv/.png"
          f"\n       four_strategy_comparison.png"
          f"\n       reprofile_best_by_phase.csv")
