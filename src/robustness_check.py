"""
Experiment 3: Does the Strategy Ranking Hold on a Second Dataset?
==================================================================
Every phased result rests on one trace with one seed and one skew strength.
This re-runs the identical four-strategy comparison on an independently
generated dataset (different seed AND stronger hot/cold skew) and asks
whether the ORDER of the strategies survives.

BOTH datasets are recomputed here. The original version hardcoded dataset 1's
numbers as literals, so the comparison could silently drift out of date the
moment anything upstream changed.

HONESTY NOTE: if the ranking changes, that IS the result. A policy ordering
that flips when you change the skew of the workload is direct evidence for
the "no single policy wins everywhere" thesis -- it is a finding, not a
failure, and it should be reported as one.

Outputs:
  results/robustness_check.csv
  results/robustness_check.png
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (
    DATA_DIR, RESULTS_DIR, ENCODING, NUM_EXPERTS, CAPACITY_K,
    DATASET2_SKEW_ALPHA, SYNTH_SKEW_ALPHA, REPROFILE_INTERVALS,
)
import tier_simulator as ts
from nonstationary_experiment import run_phase_comparison


def evaluate(trace_df, capacity_k, num_experts):
    interval_scores = {
        iv: ts.simulate_periodic(trace_df, capacity_k, iv, num_experts)["hit_rate_pct"]
        for iv in REPROFILE_INTERVALS
    }
    best_interval = max(interval_scores, key=interval_scores.get)
    _, overall, _ = run_phase_comparison(trace_df, capacity_k, num_experts, best_interval)
    return overall, best_interval


def main():
    ds1 = pd.read_csv(DATA_DIR / "nonstationary_trace.csv")
    ds2 = pd.read_csv(DATA_DIR / "dataset2_trace.csv")

    o1, iv1 = evaluate(ds1, CAPACITY_K, NUM_EXPERTS)
    o2, iv2 = evaluate(ds2, CAPACITY_K, NUM_EXPERTS)

    strategies = ["static", "lru", "hybrid", "periodic"]
    df = pd.DataFrame({
        "strategy": [s.capitalize() for s in strategies],
        "dataset1_hit_rate_pct": [o1[f"{s}_overall_pct"] for s in strategies],
        "dataset2_hit_rate_pct": [o2[f"{s}_overall_pct"] for s in strategies],
    })
    # Rank 1 = best (highest hit rate)
    df["dataset1_rank"] = df["dataset1_hit_rate_pct"].rank(ascending=False).astype(int)
    df["dataset2_rank"] = df["dataset2_hit_rate_pct"].rank(ascending=False).astype(int)
    df.to_csv(RESULTS_DIR / "robustness_check.csv", index=False, encoding=ENCODING)

    print(f"Dataset 1: alpha={SYNTH_SKEW_ALPHA} (milder skew), best periodic interval {iv1:,}")
    print(f"Dataset 2: alpha={DATASET2_SKEW_ALPHA} (stronger skew), best periodic interval {iv2:,}")
    print("\n=== Strategy ranking across two independent datasets ===")
    print(df.to_string(index=False, float_format=lambda v: f"{v:9.2f}"))

    identical = bool((df["dataset1_rank"] == df["dataset2_rank"]).all())
    if identical:
        print("\n--> Ranking is IDENTICAL across both datasets. The ordering is robust.")
    else:
        print("\n--> Ranking CHANGED between datasets. Report this: it is evidence that")
        print("    the best policy depends on workload skew, which is the central claim")
        print("    of this study. Movements:")
        for _, r in df.iterrows():
            if r["dataset1_rank"] != r["dataset2_rank"]:
                print(f"      {r['strategy']:<10} rank {r['dataset1_rank']} -> {r['dataset2_rank']}")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(df))
    width = 0.38
    ax.bar(x - width / 2, df["dataset1_hit_rate_pct"], width,
           label=f"Dataset 1 (alpha={SYNTH_SKEW_ALPHA})", color="#4C72B0")
    ax.bar(x + width / 2, df["dataset2_hit_rate_pct"], width,
           label=f"Dataset 2 (alpha={DATASET2_SKEW_ALPHA}, stronger skew)", color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels(df["strategy"])
    ax.set_ylabel("Overall HBM hit rate (%)")
    ax.set_title("Strategy ranking across two independent datasets")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "robustness_check.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
