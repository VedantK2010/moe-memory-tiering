"""
Tuning the Hybrid Split Ratio on Dataset 2
=============================================
The robustness check showed the hybrid strategy (fixed 50/50 split)
went from 2nd-best on Dataset 1 to WORST on Dataset 2. Before concluding
the hybrid approach itself is unreliable, we need to check whether that
was simply the wrong RATIO for Dataset 2's more extreme skew -- not a
flaw in the hybrid idea itself.

This sweeps every possible reserved/LRU split (0 reserved through fully
reserved) on Dataset 2's actual trace, and compares the BEST possible
hybrid score against Periodic Re-profile's score on the same dataset.
"""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import paths
from hybrid_strategy import simulate_hybrid_by_phase

NUM_EXPERTS = 8
CAPACITY_K = 4

if __name__ == "__main__":
    trace_df = pd.read_csv(paths.require_data("dataset2_trace.csv"))

    counts = trace_df[trace_df["phase"] == 0]["expert_1"].value_counts()
    ranked_from_phase0 = counts.index.tolist()
    for e in range(NUM_EXPERTS):
        if e not in ranked_from_phase0:
            ranked_from_phase0.append(e)

    results = []
    for num_reserved in range(CAPACITY_K + 1):
        phase_result = simulate_hybrid_by_phase(trace_df, CAPACITY_K, num_reserved, ranked_from_phase0)
        overall_avg = phase_result["avg_time_ns"].mean()
        results.append({
            "num_reserved_static": num_reserved,
            "num_lru_slots": CAPACITY_K - num_reserved,
            "overall_avg_time_ns": overall_avg,
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv(paths.result("dataset2_hybrid_split_sweep.csv"), index=False)

    print("=== Hybrid split sweep on Dataset 2 ===")
    print(results_df.to_string(index=False))

    best_row = results_df.loc[results_df["overall_avg_time_ns"].idxmin()]
    print(f"\nBest split on Dataset 2: {int(best_row['num_reserved_static'])} reserved / "
          f"{int(best_row['num_lru_slots'])} LRU  -->  {best_row['overall_avg_time_ns']:.1f} ns")

    # Reference points come from the robustness check's own output rather
    # than pasted literals, so they cannot drift out of step with it.
    summary_path = paths.RESULTS_DIR / "robustness_check_summary.csv"
    if not summary_path.exists():
        raise SystemExit(f"Run src/robustness_check.py first -- {summary_path.name} is missing.")
    summary = pd.read_csv(summary_path).set_index("strategy")["dataset2_overall_ns"]

    dataset2_static = summary["Pure Static"]
    dataset2_lru = summary["Pure LRU"]
    dataset2_hybrid_5050 = summary["Hybrid"]
    dataset2_reprofile = summary["Periodic Re-profile"]

    print(f"\nFor comparison, on the SAME Dataset 2:")
    print(f"  Pure Static:               {dataset2_static:.1f} ns")
    print(f"  Pure LRU:                  {dataset2_lru:.1f} ns")
    print(f"  Hybrid (fixed 50/50):      {dataset2_hybrid_5050:.1f} ns")
    print(f"  Hybrid (BEST tuned split): {best_row['overall_avg_time_ns']:.1f} ns")
    print(f"  Periodic Re-profile:       {dataset2_reprofile:.1f} ns")

    if best_row["num_reserved_static"] == 0:
        print("\n  The best 'hybrid' on this dataset reserves NOTHING -- it is pure LRU.")
        print("  Tuning the split does not rescue the hybrid here; it dissolves it.")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(results_df["num_reserved_static"], results_df["overall_avg_time_ns"],
            marker="o", color="#8172B2", label="Hybrid (varying split)")
    ax.axhline(dataset2_static, color="#4C72B0", linestyle="--", label="Pure Static")
    ax.axhline(dataset2_lru, color="#55A868", linestyle="--", label="Pure LRU")
    ax.axhline(dataset2_reprofile, color="#C44E52", linestyle="--", label="Periodic Re-profile")
    ax.scatter([best_row["num_reserved_static"]], [best_row["overall_avg_time_ns"]],
               color="red", zorder=5, s=90)
    ax.set_xlabel("Number of HBM slots permanently reserved (rest = LRU)")
    ax.set_ylabel("Overall avg access time (ns)")
    ax.set_title("Dataset 2: does tuning the hybrid split help?")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(paths.result("dataset2_hybrid_tuning.png"), dpi=150)

    print(f"\nWrote into {paths.RESULTS_DIR}:"
          f"\n       dataset2_hybrid_split_sweep.csv"
          f"\n       dataset2_hybrid_tuning.png")
