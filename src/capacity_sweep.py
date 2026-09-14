"""
Experiment 1: Capacity vs. Latency, and Does Placement Intelligence Matter?
===========================================================================
Three questions, on the stationary synthetic trace:

  1. How does hit rate / access time vary with the HBM capacity budget k?
  2. Does SMART (hot/cold) placement beat RANDOM placement at the SAME k?
  3. Does DYNAMIC (LRU) placement beat both?

Question 2 is the important one. "More HBM is faster" is true no matter
which experts you pick and proves nothing about the tiering policy.
Holding the capacity budget fixed and varying only WHICH experts are
resident isolates the placement decision itself.

Outputs:
  results/capacity_sweep.csv          (both figures below are drawn from it)
  results/capacity_sweep.png
  results/strategy_comparison.png
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (
    DATA_DIR, RESULTS_DIR, ENCODING, NUM_EXPERTS, SEED, NUM_RANDOM_TRIALS,
    HBM_TIME_NS, CXL_TIME_NS, EXPERT_SIZE_BYTES,
)
import tier_simulator as ts


def run(trace_df, num_experts=NUM_EXPERTS, num_trials=NUM_RANDOM_TRIALS, seed=SEED):
    rng = np.random.default_rng(seed)
    ranked = ts.rank_experts(trace_df, num_experts)

    rows = []
    for k in range(num_experts + 1):
        static = ts.simulate_static(trace_df, ts.assign_tiers(ranked, k))
        rand = ts.simulate_random_baseline(trace_df, k, num_trials, rng)
        lru = ts.simulate_lru(trace_df, k)

        smart_t = static["avg_time_per_access_ns"]
        rand_t = rand["avg_time_ns_mean"]
        rows.append({
            "num_experts_in_hbm": k,
            "hbm_budget_gb": k * EXPERT_SIZE_BYTES / 1e9,
            "static_hit_rate_pct": static["hit_rate_pct"],
            "static_avg_time_ns": smart_t,
            "random_hit_rate_pct": rand["hit_rate_mean"] * 100,
            "random_avg_time_ns_mean": rand_t,
            "random_avg_time_ns_std": rand["avg_time_ns_std"],
            "smart_advantage_pct": (rand_t - smart_t) / rand_t * 100 if rand_t else 0.0,
            "lru_hit_rate_pct": lru["hit_rate_pct"],
            "lru_avg_time_ns": lru["avg_time_per_access_ns"],
        })
    return pd.DataFrame(rows)


def plot(df, out_sweep, out_strategy, num_experts=NUM_EXPERTS):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(df["num_experts_in_hbm"], df["static_avg_time_ns"] / 1e6,
                 marker="o", color="#4C72B0")
    axes[0].set_xlabel("Experts kept in HBM (rest in CXL)")
    axes[0].set_ylabel("Avg access time (ms)")
    axes[0].set_title("Access time vs. HBM capacity")
    axes[0].grid(alpha=0.3)

    axes[1].plot(df["num_experts_in_hbm"], df["static_hit_rate_pct"],
                 marker="o", label="Static (frequency)", color="#4C72B0")
    axes[1].plot(df["num_experts_in_hbm"], df["lru_hit_rate_pct"],
                 marker="^", label="LRU (recency)", color="#55A868")
    axes[1].plot(df["num_experts_in_hbm"], df["random_hit_rate_pct"],
                 marker="s", label="Random", color="#DD8452")
    axes[1].set_xlabel("Experts kept in HBM")
    axes[1].set_ylabel("HBM hit rate (%)")
    axes[1].set_title("Hit rate vs. HBM capacity")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_sweep, dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(df["num_experts_in_hbm"], df["random_avg_time_ns_mean"] / 1e6,
                 marker="s", label="Random", color="#DD8452")
    axes[0].fill_between(
        df["num_experts_in_hbm"],
        (df["random_avg_time_ns_mean"] - df["random_avg_time_ns_std"]) / 1e6,
        (df["random_avg_time_ns_mean"] + df["random_avg_time_ns_std"]) / 1e6,
        color="#DD8452", alpha=0.2)
    axes[0].plot(df["num_experts_in_hbm"], df["static_avg_time_ns"] / 1e6,
                 marker="o", label="Static smart (frequency)", color="#4C72B0")
    axes[0].plot(df["num_experts_in_hbm"], df["lru_avg_time_ns"] / 1e6,
                 marker="^", label="LRU (recency)", color="#55A868")
    axes[0].set_xlabel("HBM capacity budget (experts)")
    axes[0].set_ylabel("Avg access time (ms)")
    axes[0].set_title("Three placement strategies, same budget")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].bar(df["num_experts_in_hbm"], df["smart_advantage_pct"], color="#55A868")
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set_xlabel("HBM capacity budget (experts)")
    axes[1].set_ylabel("Smart advantage over random (%)")
    axes[1].set_title("Does hot/cold placement actually help?")
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_strategy, dpi=150)
    plt.close(fig)


def main():
    trace = pd.read_csv(DATA_DIR / "expert_trace.csv")
    df = run(trace)
    df.to_csv(RESULTS_DIR / "capacity_sweep.csv", index=False, encoding=ENCODING)
    plot(df, RESULTS_DIR / "capacity_sweep.png", RESULTS_DIR / "strategy_comparison.png")

    print("=== Capacity sweep (stationary trace) ===")
    cols = ["num_experts_in_hbm", "static_hit_rate_pct", "random_hit_rate_pct",
            "lru_hit_rate_pct", "smart_advantage_pct"]
    print(df[cols].to_string(index=False, float_format=lambda v: f"{v:8.2f}"))
    best = df.loc[df["smart_advantage_pct"].idxmax()]
    print(f"\nSmart placement beats random by up to "
          f"{best['smart_advantage_pct']:.2f}% (at k={int(best['num_experts_in_hbm'])}).")
    print(f"HBM-only baseline: {HBM_TIME_NS/1e6:.3f} ms/access   "
          f"CXL-only: {CXL_TIME_NS/1e6:.3f} ms/access")


if __name__ == "__main__":
    main()
