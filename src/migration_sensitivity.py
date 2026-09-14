"""
Experiment 7: What Happens When Migration Isn't Free?
======================================================
THE OBVIOUS OBJECTION
---------------------
Our LRU result assumes that installing an expert into HBM after a miss costs
nothing. It does not. Every miss writes 352 MB into HBM, and at a 40-45%
miss rate that is a great deal of write traffic that the model currently
ignores.

Static placement never migrates -- its resident set is fixed. LRU migrates
on every miss. So charging for migration penalises LRU and leaves static
untouched, and there must be some migration cost at which LRU's advantage
disappears entirely. This finds it.

THE SWEEP
---------
MIGRATION_COST_FACTOR scales the write:
    0.0  the original assumption -- migration is free
    1.0  the full expert is written into HBM at HBM bandwidth before use
Reality sits in between: a real system overlaps part of the install with
compute, and may stream the expert directly to the compute units while
writing it. We report the whole curve rather than picking a number.

WHY THIS MATTERS BEYOND THE SENSITIVITY
---------------------------------------
If LRU only wins when migration is free, then the honest recommendation is
static or periodic placement, not LRU. Finding the crossover tells you which
recommendation to make.

Outputs:
  results/migration_sensitivity.csv
  results/migration_sensitivity.png
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (
    RESULTS_DIR, ENCODING, NUM_EXPERTS, CAPACITY_K, REAL_TRACES,
    MIGRATION_WRITE_NS, HBM_TIME_NS, CXL_TIME_NS, REAL_WARMUP_TOKENS,
)
import tier_simulator as ts

FACTORS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0]


def run_layer(name, path, capacity_k=CAPACITY_K):
    full = pd.read_csv(path)
    # Same protocol as real_benchmark.py: static is profiled on the prefix,
    # and both policies are scored on the same remainder.
    prefix = full.iloc[:REAL_WARMUP_TOKENS]
    df = full.iloc[REAL_WARMUP_TOKENS:].reset_index(drop=True)

    # Static never migrates, so it is a flat reference line.
    ranked = ts.rank_experts(prefix, NUM_EXPERTS)
    static = ts.simulate_static(df, ts.assign_tiers(ranked, capacity_k))
    static_time = static["avg_time_per_access_ns"]

    # LRU's time is linear in the migration factor (every miss pays
    # factor x a full write), so the break-even factor is exact rather than
    # read off the sweep grid:  static = lru_0 + f* x write x miss_rate.
    lru0 = ts.simulate_lru(df, capacity_k, migration_factor=0.0)
    miss_rate = lru0["misses"] / lru0["accesses"]
    breakeven = (static_time - lru0["avg_time_per_access_ns"]) / (MIGRATION_WRITE_NS * miss_rate)

    rows = []
    for f in FACTORS:
        lru = ts.simulate_lru(df, capacity_k, migration_factor=f)
        rows.append({
            "breakeven_factor": breakeven,
            "layer": name,
            "migration_factor": f,
            "migration_cost_us": MIGRATION_WRITE_NS * f / 1e3,
            "lru_hit_rate_pct": lru["hit_rate_pct"],
            "lru_avg_time_ns": lru["avg_time_per_access_ns"],
            "static_avg_time_ns": static_time,
            "static_hit_rate_pct": static["hit_rate_pct"],
            "lru_advantage_pct": (static_time - lru["avg_time_per_access_ns"])
                                 / static_time * 100,
            "migrations": lru["migrations"],
        })
    return pd.DataFrame(rows)


def main():
    parts = []
    for name, path in REAL_TRACES.items():
        if path.exists():
            parts.append(run_layer(name, path))
    if not parts:
        print("No real traces found -- skipping.")
        return

    df = pd.concat(parts, ignore_index=True)
    df.to_csv(RESULTS_DIR / "migration_sensitivity.csv", index=False, encoding=ENCODING)

    print("=== Migration cost sensitivity (LRU vs static, real Mixtral traces) ===")
    print(f"A full migration writes {MIGRATION_WRITE_NS/1e3:,.0f} us of expert into HBM.")
    print(f"For reference: one HBM access is {HBM_TIME_NS/1e3:,.0f} us, "
          f"one CXL access {CXL_TIME_NS/1e3:,.0f} us.\n")

    for layer in df["layer"].unique():
        sub = df[df["layer"] == layer]
        print(f"--- {layer} ---")
        print(sub[["migration_factor", "migration_cost_us", "lru_hit_rate_pct",
                   "lru_advantage_pct"]].to_string(index=False,
                                                    float_format=lambda v: f"{v:10.2f}"))
        x = float(sub.iloc[0]["breakeven_factor"])
        base = sub.iloc[0]["lru_advantage_pct"]
        if x > 1:
            print(f"  LRU beats static across the whole sweep "
                  f"(advantage {base:.2f}% -> {sub.iloc[-1]['lru_advantage_pct']:.2f}%); "
                  f"break-even would need {x:.0%} of a full write.\n")
        else:
            print(f"  LRU's {base:.2f}% advantage vanishes once migration costs "
                  f"{x:.1%} of a full HBM write\n  "
                  f"({MIGRATION_WRITE_NS*x/1e3:,.1f} us per miss). Beyond that, "
                  f"static is the better policy.\n")

    fig, axes = plt.subplots(1, len(df["layer"].unique()), figsize=(11, 4.3), squeeze=False)
    for ax, layer in zip(axes[0], df["layer"].unique()):
        sub = df[df["layer"] == layer]
        ax.plot(sub["migration_factor"], sub["lru_advantage_pct"],
                marker="o", color="#55A868")
        ax.axhline(0, color="#C44E52", ls="--", lw=1, label="Static placement")
        ax.fill_between(sub["migration_factor"], 0, sub["lru_advantage_pct"],
                        where=sub["lru_advantage_pct"] > 0, alpha=0.15, color="#55A868")
        ax.fill_between(sub["migration_factor"], 0, sub["lru_advantage_pct"],
                        where=sub["lru_advantage_pct"] <= 0, alpha=0.15, color="#C44E52")
        ax.set_xlabel("Migration cost (fraction of a full HBM write)")
        ax.set_ylabel("LRU advantage over static (%)")
        ax.set_title(layer)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("Does LRU still win once you pay to install the expert?", y=1.02)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "migration_sensitivity.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
