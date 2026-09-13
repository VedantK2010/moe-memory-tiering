"""
Experiment 4: Real Mixtral 8x7B Routing Traces
===============================================
Runs every placement strategy against REAL recorded expert routing from
Mixtral 8x7B inference (HuggingFace allenai/analysis_mixtral), at two
network depths: layer 15 (middle) and layer 31 (deep).

THREE THINGS THIS DOES THAT THE ORIGINAL DID NOT
------------------------------------------------
1. TOP-2, not just top-1. Mixtral routes each token to two experts. The
   traces contain both, but the original benchmark read only expert_1 --
   evaluating half the workload while claiming the full "1.6 million
   routing decisions". Both are reported here, side by side, because the
   gap between them is itself a finding.

2. TOKEN-LEVEL residency. A token only avoids CXL entirely when ALL of its
   routed experts are resident. With top-2 and a 4-of-8 budget a healthy
   per-access hit rate can still leave most TOKENS taking a CXL trip -- and
   the token is what the user waits for.

3. THE FULL INTERVAL SWEEP for periodic re-profiling, with the interval
   reported next to every number. A periodic result quoted without its
   interval is not reproducible, and very short intervals win for reasons
   that undercut the strategy's own justification.

Outputs:
  results/real_trace_results.csv       one row per (layer, routing, strategy)
  results/real_periodic_sweep.csv      hit rate vs re-profiling interval
  results/real_trace_results.png
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (
    RESULTS_DIR, ENCODING, NUM_EXPERTS, CAPACITY_K, REAL_TRACES,
    REPROFILE_INTERVALS, TOP_K,
)
import tier_simulator as ts

WARMUP_TOKENS = 20_000


def static_oracle(trace_df, capacity_k, top_k):
    """Upper bound: rank on the ENTIRE trace. Not deployable (it sees the
    future); reported as the ceiling a perfect profiler could reach."""
    ranked = ts.rank_experts(trace_df, NUM_EXPERTS, top_k=top_k)
    return ts.simulate_static(trace_df, ts.assign_tiers(ranked, capacity_k), top_k=top_k)


def static_warmup(trace_df, capacity_k, top_k, warmup=WARMUP_TOKENS):
    """Deployable: profile on the first `warmup` tokens, freeze, then measure
    only on the remainder."""
    head, tail = trace_df.iloc[:warmup], trace_df.iloc[warmup:]
    ranked = ts.rank_experts(head, NUM_EXPERTS, top_k=top_k)
    return ts.simulate_static(tail, ts.assign_tiers(ranked, capacity_k), top_k=top_k)


def analyse_layer(name, path, capacity_k=CAPACITY_K):
    df = pd.read_csv(path)
    rows = []
    seq1 = df["expert_1"].to_numpy()
    repeat_rate = float((seq1[1:] == seq1[:-1]).mean())

    for routing, top_k in [("top-1", 1), ("top-2", TOP_K)]:
        oracle = static_oracle(df, capacity_k, top_k)
        warm = static_warmup(df, capacity_k, top_k)
        lru = ts.simulate_lru(df, capacity_k, top_k=top_k)

        per_interval = {
            iv: ts.simulate_periodic(df, capacity_k, iv, NUM_EXPERTS, top_k=top_k)
            for iv in REPROFILE_INTERVALS
        }
        best_iv = max(per_interval, key=lambda k: per_interval[k]["hit_rate_pct"])

        for strategy, res, extra in [
            ("Static (oracle)", oracle, ""),
            ("Static (profiled on first 20k)", warm, ""),
            ("LRU", lru, ""),
            (f"Periodic (every {best_iv:,})", per_interval[best_iv], f"interval={best_iv}"),
        ]:
            rows.append({
                "layer": name, "routing": routing, "strategy": strategy,
                "hit_rate_pct": res["hit_rate_pct"],
                "avg_time_ns": res["avg_time_per_access_ns"],
                "note": extra,
            })

    # Token-level residency only means something for top-k > 1.
    tok_lru = ts.token_level_stats(df, capacity_k, "lru")
    ranked = ts.rank_experts(df, NUM_EXPERTS)
    tok_static = ts.token_level_stats(
        df, capacity_k, "static", tier_map=ts.assign_tiers(ranked, capacity_k))

    sweep = pd.DataFrame([
        {"layer": name, "routing": routing, "interval": iv,
         "hit_rate_pct": ts.simulate_periodic(
             df, capacity_k, iv, NUM_EXPERTS, top_k=tk)["hit_rate_pct"]}
        for routing, tk in [("top-1", 1), ("top-2", TOP_K)]
        for iv in REPROFILE_INTERVALS
    ])

    meta = {
        "layer": name, "tokens": len(df), "repeat_rate": repeat_rate,
        "lru_all_resident_pct": tok_lru["all_resident_pct"],
        "lru_any_miss_pct": tok_lru["any_miss_pct"],
        "static_all_resident_pct": tok_static["all_resident_pct"],
    }
    return pd.DataFrame(rows), sweep, meta


def main():
    all_rows, all_sweeps, metas = [], [], []
    for name, path in REAL_TRACES.items():
        if not path.exists():
            print(f"  !! missing {path.name} -- skipping {name}")
            continue
        rows, sweep, meta = analyse_layer(name, path)
        all_rows.append(rows)
        all_sweeps.append(sweep)
        metas.append(meta)

    if not all_rows:
        print("No real traces found in data/. Skipping.")
        return

    results = pd.concat(all_rows, ignore_index=True)
    sweeps = pd.concat(all_sweeps, ignore_index=True)
    results.to_csv(RESULTS_DIR / "real_trace_results.csv", index=False, encoding=ENCODING)
    sweeps.to_csv(RESULTS_DIR / "real_periodic_sweep.csv", index=False, encoding=ENCODING)

    for m in metas:
        print(f"\n=== {m['layer']} ({m['tokens']:,} tokens, "
              f"consecutive-repeat rate {m['repeat_rate']:.4f}) ===")
        sub = results[results["layer"] == m["layer"]]
        for routing in ("top-1", "top-2"):
            print(f"  {routing}:")
            for _, r in sub[sub["routing"] == routing].iterrows():
                print(f"    {r['strategy']:<32} {r['hit_rate_pct']:6.2f}%")
        print(f"  TOKEN-LEVEL (top-2, LRU): both experts resident "
              f"{m['lru_all_resident_pct']:.2f}%  |  "
              f"at least one CXL trip {m['lru_any_miss_pct']:.2f}%")

    print("\n" + "=" * 70)
    print("Headline: per-access hit rate overstates what tiering buys under")
    print("top-2 routing. Report token-level residency alongside it.")
    print("=" * 70)

    # Normalise the periodic label (it carries its interval) so the same four
    # strategies line up across both panels.
    plot_df = results.copy()
    plot_df["strategy_key"] = plot_df["strategy"].str.replace(
        r"Periodic \(every [\d,]+\)", "Periodic (best interval)", regex=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    colours = ["#4C72B0", "#8FA8CE", "#55A868", "#C44E52"]
    for ax, routing in zip(axes, ("top-1", "top-2")):
        sub = plot_df[plot_df["routing"] == routing]
        pivot = sub.pivot_table(index="layer", columns="strategy_key",
                                values="hit_rate_pct", sort=False)
        layers = list(pivot.index)
        strategies = list(pivot.columns)
        width = 0.8 / len(strategies)
        for i, strat in enumerate(strategies):
            ax.bar(np.arange(len(layers)) + (i - len(strategies) / 2 + 0.5) * width,
                   pivot[strat].to_numpy(), width, label=strat,
                   color=colours[i % len(colours)])
        ax.set_xticks(range(len(layers)))
        ax.set_xticklabels(layers)
        ax.set_title(f"Real Mixtral routing, {routing}")
        ax.grid(alpha=0.3, axis="y")
    axes[0].set_ylabel("HBM hit rate (%)")
    axes[1].legend(fontsize=7, loc="lower right")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "real_trace_results.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
