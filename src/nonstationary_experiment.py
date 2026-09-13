"""
Experiment 2: Four Strategies Under a Shifting Workload
========================================================
The stationary trace rewards frequency-based placement, because the hot set
never changes. Real serving does not look like that -- a model handling
different tasks or topics over time sees WHICH experts are hot shift.

This runs all four strategies on the phased trace:

  Pure Static  -- profiled ONCE on phase 0, then frozen. This is the fair
                  real-world framing: you do not get to see the future when
                  you pick a static placement.
  Pure LRU     -- adapts on every access.
  Hybrid       -- half the budget pinned to phase-0 hot experts, half LRU.
  Periodic     -- re-ranks every N tokens from the window just finished.

Also sweeps the hybrid static/LRU split to find the best division of a
fixed budget.

Outputs:
  results/nonstationary_by_phase.csv
  results/hybrid_split_sweep.csv
  results/nonstationary_by_phase.png
  results/hybrid_split_sweep.png
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (
    DATA_DIR, RESULTS_DIR, ENCODING, NUM_EXPERTS, CAPACITY_K, REPROFILE_INTERVALS,
)
import tier_simulator as ts


def static_from_first_phase(trace_df, num_experts, capacity_k):
    """Profile on phase 0 only, freeze for the whole run."""
    phase0 = trace_df[trace_df["phase"] == 0]
    ranked = ts.rank_experts(phase0, num_experts)
    return ts.assign_tiers(ranked, capacity_k), ranked


def by_phase(trace_df, fn):
    """Apply a per-access simulator phase by phase, keeping cache state
    continuous across boundaries (a real cache does not know about our
    phase labels)."""
    return [fn(g) for _, g in trace_df.groupby("phase")]


def hit_grid_lru(trace_df, capacity_k):
    """LRU over the whole trace, returning a per-access hit mask so results
    can be grouped by phase afterwards without resetting the cache."""
    cols = ts.expert_columns(trace_df)
    seq = trace_df[cols].to_numpy().reshape(-1)
    hits = np.zeros(seq.size, dtype=bool)
    cache, clock = {}, 0
    for i, e in enumerate(seq):
        clock += 1
        if e in cache:
            cache[e] = clock
            hits[i] = True
        else:
            if len(cache) >= capacity_k:
                del cache[min(cache, key=cache.get)]
            cache[e] = clock
    return hits.reshape(len(trace_df), len(cols))


def hit_grid_hybrid(trace_df, capacity_k, num_reserved, ranked):
    cols = ts.expert_columns(trace_df)
    seq = trace_df[cols].to_numpy().reshape(-1)
    reserved = set(ranked[:num_reserved])
    lru_cap = capacity_k - num_reserved
    hits = np.zeros(seq.size, dtype=bool)
    cache, clock = {}, 0
    for i, e in enumerate(seq):
        clock += 1
        if e in reserved:
            hits[i] = True
        elif e in cache:
            cache[e] = clock
            hits[i] = True
        elif lru_cap > 0:
            if len(cache) >= lru_cap:
                del cache[min(cache, key=cache.get)]
            cache[e] = clock
    return hits.reshape(len(trace_df), len(cols))


def hit_grid_periodic(trace_df, capacity_k, interval, num_experts):
    cols = ts.expert_columns(trace_df)
    tokens = trace_df[cols].to_numpy()
    hits = np.zeros(tokens.shape, dtype=bool)
    resident = np.zeros(num_experts, dtype=bool)
    resident[:capacity_k] = True
    for start in range(0, len(tokens), interval):
        window = tokens[start:start + interval]
        hits[start:start + interval] = resident[window]
        counts = np.bincount(window.reshape(-1), minlength=num_experts)
        resident = np.zeros(num_experts, dtype=bool)
        resident[np.argsort(-counts, kind="stable")[:capacity_k]] = True
    return hits


def grid_to_phase_table(grid, trace_df, label):
    per_token = grid.mean(axis=1)
    df = pd.DataFrame({"phase": trace_df["phase"].to_numpy(), "hit_rate": per_token})
    out = df.groupby("phase")["hit_rate"].mean().reset_index()
    out[f"hit_rate_pct_{label}"] = out["hit_rate"] * 100
    return out[["phase", f"hit_rate_pct_{label}"]]


def run_phase_comparison(trace_df, capacity_k, num_experts, best_interval):
    tier_map, ranked_p0 = static_from_first_phase(trace_df, num_experts, capacity_k)

    in_hbm = np.zeros(num_experts, dtype=bool)
    for e, t in tier_map.items():
        if t == "HBM":
            in_hbm[e] = True
    cols = ts.expert_columns(trace_df)
    static_grid = in_hbm[trace_df[cols].to_numpy()]

    grids = {
        "static": static_grid,
        "lru": hit_grid_lru(trace_df, capacity_k),
        "hybrid": hit_grid_hybrid(trace_df, capacity_k, capacity_k // 2, ranked_p0),
        "periodic": hit_grid_periodic(trace_df, capacity_k, best_interval, num_experts),
    }

    table = None
    for label, grid in grids.items():
        part = grid_to_phase_table(grid, trace_df, label)
        table = part if table is None else table.merge(part, on="phase")

    overall = {f"{k}_overall_pct": float(g.mean() * 100) for k, g in grids.items()}
    return table, overall, grids


def hybrid_split_sweep(trace_df, capacity_k, num_experts):
    ranked = ts.rank_experts(trace_df, num_experts)
    rows = []
    for reserved in range(capacity_k + 1):
        r = ts.simulate_hybrid(trace_df, capacity_k, reserved, ranked)
        rows.append({
            "num_reserved_static": reserved,
            "num_lru_slots": capacity_k - reserved,
            "hit_rate_pct": r["hit_rate_pct"],
            "avg_time_ns": r["avg_time_per_access_ns"],
        })
    return pd.DataFrame(rows)


def main():
    trace = pd.read_csv(DATA_DIR / "nonstationary_trace.csv")

    # Pick the periodic interval on its own merits, and report which won.
    interval_scores = {
        iv: ts.simulate_periodic(trace, CAPACITY_K, iv, NUM_EXPERTS)["hit_rate_pct"]
        for iv in REPROFILE_INTERVALS
    }
    best_interval = max(interval_scores, key=interval_scores.get)

    table, overall, _ = run_phase_comparison(trace, CAPACITY_K, NUM_EXPERTS, best_interval)
    table.to_csv(RESULTS_DIR / "nonstationary_by_phase.csv", index=False, encoding=ENCODING)

    split = hybrid_split_sweep(trace, CAPACITY_K, NUM_EXPERTS)
    split.to_csv(RESULTS_DIR / "hybrid_split_sweep.csv", index=False, encoding=ENCODING)

    print("=== Periodic interval selection (phased trace) ===")
    for iv, hr in interval_scores.items():
        mark = "  <-- best" if iv == best_interval else ""
        print(f"  every {iv:>6,} tokens : {hr:6.2f}%{mark}")
    print(f"\nNOTE: the winning interval is {best_interval:,} tokens. Always quote "
          f"periodic results\n      with the interval attached -- short intervals "
          f"approach recency behaviour\n      while costing MORE bookkeeping than LRU, not less.\n")

    print("=== HBM hit rate by phase (hot experts rotate each phase) ===")
    print(table.to_string(index=False, float_format=lambda v: f"{v:8.2f}"))
    print("\nOverall: " + "  ".join(f"{k.replace('_overall_pct','')}={v:.2f}%"
                                    for k, v in overall.items()))

    print("\n=== Hybrid split sweep (budget = 4 experts) ===")
    print(split.to_string(index=False, float_format=lambda v: f"{v:10.2f}"))
    best = split.loc[split["hit_rate_pct"].idxmax()]
    print(f"Best split: {int(best['num_reserved_static'])} pinned / "
          f"{int(best['num_lru_slots'])} LRU  ->  {best['hit_rate_pct']:.2f}%")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = table["phase"].to_numpy()
    width = 0.2
    for i, (label, colour) in enumerate([
        ("static", "#4C72B0"), ("lru", "#55A868"),
        ("hybrid", "#8172B2"), ("periodic", "#C44E52")
    ]):
        ax.bar(x + (i - 1.5) * width, table[f"hit_rate_pct_{label}"], width,
               label=label.capitalize(), color=colour)
    ax.set_xlabel("Phase (hot experts rotate at each boundary)")
    ax.set_ylabel("HBM hit rate (%)")
    ax.set_title(f"Four strategies under a shifting workload "
                 f"(periodic interval = {best_interval:,})")
    ax.set_xticks(x)
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "nonstationary_by_phase.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(split["num_reserved_static"], split["hit_rate_pct"], marker="o", color="#8172B2")
    ax.scatter([best["num_reserved_static"]], [best["hit_rate_pct"]],
               color="red", zorder=5, s=90,
               label=f"Best: {int(best['num_reserved_static'])} pinned")
    ax.set_xlabel("HBM slots permanently pinned (rest = LRU)")
    ax.set_ylabel("HBM hit rate (%)")
    ax.set_title("Hybrid: best static/LRU split of a fixed budget")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "hybrid_split_sweep.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
