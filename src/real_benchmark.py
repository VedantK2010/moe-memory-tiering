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

ONE EVALUATION WINDOW FOR EVERY STRATEGY
----------------------------------------
The first WARMUP_TOKENS tokens are a profiling prefix and are never scored.
Every strategy -- static, LRU, periodic, and the token-level statistics --
is scored on the same remainder, so none of them is measured on tokens it
was fitted to or on a different span of the workload. Deployable static
placement ranks experts on the prefix only; the static ORACLE ranks on the
evaluation window itself (it sees the future) and is reported as a ceiling,
labelled as such.

Outputs:
  results/real_trace_results.csv       one row per (layer, routing, strategy)
  results/real_periodic_sweep.csv      hit rate vs re-profiling interval
  results/real_token_residency.csv     per layer: tokens with ALL experts resident
  results/real_trace_profile.csv       per-expert selection shares (dashboard replay)
  results/real_trace_results.png
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (
    RESULTS_DIR, ENCODING, NUM_EXPERTS, CAPACITY_K, REAL_TRACES,
    REPROFILE_INTERVALS, TOP_K, REAL_WARMUP_TOKENS as WARMUP_TOKENS,
)
import tier_simulator as ts


def static_oracle(eval_df, capacity_k, top_k):
    """Upper bound: rank on the evaluation window itself. Not deployable (it
    sees the future); reported as the ceiling a perfect profiler could reach."""
    ranked = ts.rank_experts(eval_df, NUM_EXPERTS, top_k=top_k)
    return ts.simulate_static(eval_df, ts.assign_tiers(ranked, capacity_k), top_k=top_k)


def static_warmup(prefix_df, eval_df, capacity_k, top_k):
    """Deployable: profile on the prefix, freeze, then measure on the
    evaluation window only."""
    ranked = ts.rank_experts(prefix_df, NUM_EXPERTS, top_k=top_k)
    return ts.simulate_static(eval_df, ts.assign_tiers(ranked, capacity_k), top_k=top_k)


def analyse_layer(name, path, capacity_k=CAPACITY_K):
    full = pd.read_csv(path)
    # Profiling prefix vs the evaluation window every strategy is scored on.
    prefix = full.iloc[:WARMUP_TOKENS]
    df = full.iloc[WARMUP_TOKENS:].reset_index(drop=True)
    rows = []
    seq1 = df["expert_1"].to_numpy()
    repeat_rate = float((seq1[1:] == seq1[:-1]).mean())

    for routing, top_k in [("top-1", 1), ("top-2", TOP_K)]:
        oracle = static_oracle(df, capacity_k, top_k)
        warm = static_warmup(prefix, df, capacity_k, top_k)
        lru = ts.simulate_lru(df, capacity_k, top_k=top_k)

        per_interval = {
            iv: ts.simulate_periodic(df, capacity_k, iv, NUM_EXPERTS, top_k=top_k)
            for iv in REPROFILE_INTERVALS
        }
        best_iv = max(per_interval, key=lambda k: per_interval[k]["hit_rate_pct"])

        for strategy, res, extra in [
            ("Static (oracle)", oracle, ""),
            (f"Static (profiled on first {WARMUP_TOKENS // 1000}k)", warm, ""),
            ("LRU", lru, ""),
            (f"Periodic (every {best_iv:,})", per_interval[best_iv], f"interval={best_iv}"),
        ]:
            rows.append({
                "layer": name, "routing": routing, "strategy": strategy,
                "hit_rate_pct": res["hit_rate_pct"],
                "avg_time_ns": res["avg_time_per_access_ns"],
                "note": extra,
            })

    # Token-level residency only means something for top-k > 1. Static uses
    # the deployable ranking (profiled on the prefix), like the rows above.
    tok_lru = ts.token_level_stats(df, capacity_k, "lru")
    ranked = ts.rank_experts(prefix, NUM_EXPERTS)
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
        "layer": name, "tokens": len(full), "eval_tokens": len(df),
        "repeat_rate": repeat_rate,
        "lru_all_resident_pct": tok_lru["all_resident_pct"],
        "lru_any_miss_pct": tok_lru["any_miss_pct"],
        "static_all_resident_pct": tok_static["all_resident_pct"],
    }

    # Per-expert selection shares over the whole trace. The dashboard's live
    # replay is calibrated to these, so its illustration traces to the data.
    cols = ts.expert_columns(full)
    first = np.bincount(full[cols[0]].to_numpy(), minlength=NUM_EXPERTS)
    second = (np.bincount(full[cols[1]].to_numpy(), minlength=NUM_EXPERTS)
              if len(cols) > 1 else np.zeros(NUM_EXPERTS, dtype=int))
    profile = pd.DataFrame({
        "layer": name, "expert": np.arange(NUM_EXPERTS),
        "first_choice_share_pct": first / first.sum() * 100,
        "second_choice_share_pct": second / max(1, second.sum()) * 100,
    })
    return pd.DataFrame(rows), sweep, meta, profile


def main():
    all_rows, all_sweeps, metas, profiles = [], [], [], []
    for name, path in REAL_TRACES.items():
        if not path.exists():
            print(f"  !! missing {path.name} -- skipping {name}")
            continue
        rows, sweep, meta, profile = analyse_layer(name, path)
        all_rows.append(rows)
        all_sweeps.append(sweep)
        metas.append(meta)
        profiles.append(profile)

    if not all_rows:
        print("No real traces found in data/. Skipping.")
        return

    results = pd.concat(all_rows, ignore_index=True)
    sweeps = pd.concat(all_sweeps, ignore_index=True)
    results.to_csv(RESULTS_DIR / "real_trace_results.csv", index=False, encoding=ENCODING)
    sweeps.to_csv(RESULTS_DIR / "real_periodic_sweep.csv", index=False, encoding=ENCODING)
    pd.DataFrame(metas).to_csv(RESULTS_DIR / "real_token_residency.csv",
                               index=False, encoding=ENCODING)
    pd.concat(profiles, ignore_index=True).to_csv(
        RESULTS_DIR / "real_trace_profile.csv", index=False, encoding=ENCODING)

    for m in metas:
        print(f"\n=== {m['layer']} ({m['tokens']:,} tokens, {m['eval_tokens']:,} scored; "
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
