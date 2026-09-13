"""
Robustness Check: Does the Strategy Ranking Survive a Second Dataset?
======================================================================
Every result in this project rests on one random trace. This script
regenerates a completely independent dataset -- different seed AND a
different skew strength -- and re-runs the same four-strategy comparison,
to check whether the ranking of strategies holds up or was an artefact of
the first dataset's particular randomness.

TWO FIXES OVER THE FIRST VERSION
---------------------------------
1. Dataset 1's numbers were pasted in as literals, so the "comparison"
   silently compared fresh results against stale constants. Both datasets
   are now generated and scored in the same run, by the same code.

2. The trace generator was copy-pasted here with one value changed.
   nonstationary_experiment.generate_nonstationary_trace now takes
   skew_alpha directly, so this script calls it instead of carrying a
   near-identical duplicate that could drift out of sync.
"""

import numpy as np
import pandas as pd

import paths
from nonstationary_experiment import (
    generate_nonstationary_trace, static_placement_from_first_phase,
    simulate_static_by_phase, simulate_lru_by_phase,
)
from hybrid_strategy import simulate_hybrid_by_phase
from periodic_reprofile import simulate_periodic_reprofile

NUM_EXPERTS = 8
TOP_K = 2
CAPACITY_K = 4
TOKENS_PER_PHASE = 20_000
NUM_PHASES = 3
REPROFILE_INTERVAL = 2000   # the winning interval from the dataset 1 sweep

# Dataset 1 is the original: seed 7, mild skew, Mixtral's deep-layer locality.
# Dataset 2 deliberately differs on all three axes at once -- if the ranking
# is fragile, this is where it shows.
DATASETS = {
    "dataset1": dict(seed=7,  skew_alpha=6, p_repeat=0.27),
    "dataset2": dict(seed=99, skew_alpha=3, p_repeat=0.20),
}


def rank_experts_from_phase0(trace_df, num_experts):
    counts = trace_df[trace_df["phase"] == 0]["expert_1"].value_counts()
    ranked = counts.index.tolist()
    for e in range(num_experts):
        if e not in ranked:
            ranked.append(e)
    return ranked


def score_all_strategies(trace_df):
    """Overall average access time for each of the four strategies."""
    tier_map = static_placement_from_first_phase(trace_df, NUM_EXPERTS, CAPACITY_K)
    static_phase = simulate_static_by_phase(trace_df, tier_map)
    lru_phase = simulate_lru_by_phase(trace_df, CAPACITY_K)

    ranked = rank_experts_from_phase0(trace_df, NUM_EXPERTS)
    hybrid_phase = simulate_hybrid_by_phase(trace_df, CAPACITY_K, CAPACITY_K // 2, ranked)

    reprofile = simulate_periodic_reprofile(trace_df, CAPACITY_K, NUM_EXPERTS, REPROFILE_INTERVAL)
    reprofile_phase = reprofile.groupby("phase")["avg_time_ns"].mean().reset_index()

    return {
        "Pure Static": static_phase["avg_time_ns"].mean(),
        "Pure LRU": lru_phase["avg_time_ns"].mean(),
        "Hybrid": hybrid_phase["avg_time_ns"].mean(),
        "Periodic Re-profile": reprofile_phase["avg_time_ns"].mean(),
    }


def main():
    scores = {}
    for name, cfg in DATASETS.items():
        trace_df, phase_probs = generate_nonstationary_trace(
            NUM_EXPERTS, TOP_K, TOKENS_PER_PHASE, NUM_PHASES,
            cfg["p_repeat"], cfg["seed"], skew_alpha=cfg["skew_alpha"],
        )
        trace_df.to_csv(paths.data(f"{name}_trace.csv"), index=False)

        print(f"=== {name} (seed={cfg['seed']}, alpha={cfg['skew_alpha']}, "
              f"p_repeat={cfg['p_repeat']}) ===")
        for i, probs in enumerate(phase_probs):
            print(f"  Phase {i}: hottest = expert {int(np.argmax(probs))}, "
                  f"max share = {probs.max():.3f}")

        scores[name] = score_all_strategies(trace_df)
        print()

    summary = pd.DataFrame({
        "strategy": list(scores["dataset1"].keys()),
        "dataset1_overall_ns": list(scores["dataset1"].values()),
        "dataset2_overall_ns": list(scores["dataset2"].values()),
    })
    summary["dataset1_rank"] = summary["dataset1_overall_ns"].rank().astype(int)
    summary["dataset2_rank"] = summary["dataset2_overall_ns"].rank().astype(int)
    summary["rank_change"] = summary["dataset2_rank"] - summary["dataset1_rank"]

    out = paths.result("robustness_check_summary.csv")
    summary.to_csv(out, index=False)

    print("=== Strategy ranking across two independent datasets ===")
    print(summary.to_string(index=False))

    moved = summary[summary["rank_change"] != 0]
    if moved.empty:
        print("\n--> Ranking is IDENTICAL across both datasets. Result is robust.")
    else:
        print("\n--> Ranking CHANGED between datasets. Strategies that moved:")
        for _, r in moved.iterrows():
            print(f"      {r['strategy']}: #{r['dataset1_rank']} -> #{r['dataset2_rank']}")
        held = summary[summary["rank_change"] == 0]["strategy"].tolist()
        print(f"    Held rank: {', '.join(held) if held else 'none'}")
        print("\n    A strategy whose rank moves is tuned to one dataset's shape, not")
        print("    to the problem. Report it as such rather than dropping it.")

    print(f"\nWrote: {out}")


if __name__ == "__main__":
    main()
