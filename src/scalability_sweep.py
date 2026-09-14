"""
Experiment 8: Does Tiering Scale With Expert Count?
====================================================
WHY THIS EXISTS
---------------
Everything else here uses Mixtral's 8 experts, which is the LEAST favourable
case for expert tiering. Modern MoE models are far more granular:

    Mixtral 8x7B      8 experts,  top-2   (k/N = 25%)
    DeepSeek-V2     160 experts,  top-6   (k/N = 3.8%)
    DeepSeek-V3     256 experts,  top-8   (k/N = 3.1%)

The intuition going in was that finer granularity helps: each token touches
a smaller FRACTION of experts, so a hot/cold split should have more room to
work. The measurement does NOT support that intuition, and the reason it
fails is the interesting part -- see the conclusion below.

WHAT IS HELD FIXED
------------------
The HBM budget is expressed as a FRACTION of all experts, not an absolute
count, so the comparison is about granularity rather than about giving
larger models more memory. Skew (Dirichlet alpha) and temporal locality are
held constant across N so that only granularity changes.

TWO ROUTING REGIMES
-------------------
  fixed-k   : top-2 regardless of N (isolates the effect of granularity)
  scaled-k  : k grows as N/32, roughly tracking real granular-MoE designs

Outputs:
  results/scalability_sweep.csv
  results/scalability_sweep.png
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (
    RESULTS_DIR, ENCODING, SEED, SYNTH_SKEW_ALPHA, SYNTH_P_REPEAT,
    EXPERT_SIZE_BYTES, NUM_LAYERS, NUM_EXPERTS,
)
import tier_simulator as ts
from generate_trace import _solve_forced_repeat

# Up to 256 experts so the scaled-k regime reaches DeepSeek-V3's topology
# (256 experts, top-8 = 256/32).
EXPERT_COUNTS = [8, 16, 32, 64, 128, 256]
BUDGET_FRACTIONS = [0.125, 0.25, 0.5]
TOKENS = 40_000


def synth_trace(num_experts, top_k, num_tokens=TOKENS, alpha=SYNTH_SKEW_ALPHA,
                p_repeat=SYNTH_P_REPEAT, seed=SEED):
    """Same generator family as generate_trace.py, parameterised by N and k.

    Total expert weight is held constant as N grows: a 128-expert model has
    experts 1/16th the size of an 8-expert model, matching how granular MoE
    designs actually split the same FFN capacity into more, smaller pieces.
    """
    rng = np.random.default_rng(seed + num_experts)
    base = rng.dirichlet(alpha=[alpha] * num_experts)
    p_forced = _solve_forced_repeat(base, p_repeat)

    rows, prev = [], None
    all_e = np.arange(num_experts)
    for _ in range(num_tokens):
        if prev is not None and rng.random() < p_forced:
            first = prev
        else:
            first = int(rng.choice(num_experts, p=base))
        mask = all_e != first
        probs = base[mask] / base[mask].sum()
        rest = rng.choice(all_e[mask], size=top_k - 1, replace=False, p=probs)
        rows.append([first, *rest])
        prev = first

    return pd.DataFrame(rows, columns=[f"expert_{i+1}" for i in range(top_k)])


def main():
    rows = []
    for regime, k_of in [("fixed-k (top-2)", lambda n: 2),
                         ("scaled-k (N/32)", lambda n: max(2, n // 32))]:
        for n in EXPERT_COUNTS:
            k = k_of(n)
            trace = synth_trace(n, k)
            ranked = ts.rank_experts(trace, n)
            # Each expert is 1/N of Mixtral's per-layer expert weight.
            expert_bytes = EXPERT_SIZE_BYTES * NUM_EXPERTS / n

            for frac in BUDGET_FRACTIONS:
                cap = max(1, int(round(n * frac)))
                tier_map = ts.assign_tiers(ranked, cap)
                static = ts.simulate_static(trace, tier_map, num_experts=n)
                lru = ts.simulate_lru(trace, cap, num_experts=n)
                tok = ts.token_level_stats(trace, cap, "static",
                                           tier_map=tier_map, num_experts=n)
                rows.append({
                    "regime": regime,
                    "num_experts": n,
                    "top_k": k,
                    "routed_fraction_pct": k / n * 100,
                    "budget_fraction": frac,
                    "experts_in_hbm": cap,
                    "hbm_gb_per_layer": cap * expert_bytes / 1e9,
                    "cxl_offload_gb_all_layers": (n - cap) * expert_bytes * NUM_LAYERS / 1e9,
                    "static_hit_rate_pct": static["hit_rate_pct"],
                    "lru_hit_rate_pct": lru["hit_rate_pct"],
                    "tokens_fully_resident_pct": tok["all_resident_pct"],
                })

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "scalability_sweep.csv", index=False, encoding=ENCODING)

    for regime in df["regime"].unique():
        print(f"\n=== {regime} ===")
        sub = df[df["regime"] == regime]
        print(sub[["num_experts", "top_k", "routed_fraction_pct", "budget_fraction",
                   "static_hit_rate_pct", "lru_hit_rate_pct",
                   "tokens_fully_resident_pct", "cxl_offload_gb_all_layers"]]
              .to_string(index=False, float_format=lambda v: f"{v:9.2f}"))

    half = df[(df["budget_fraction"] == 0.5) & (df["regime"] == "scaled-k (N/32)")]
    lo, hi = half.iloc[0], half.iloc[-1]
    print(f"\nAt a 50% HBM budget, scaled-k: token-level residency goes from "
          f"{lo['tokens_fully_resident_pct']:.1f}% at N={int(lo['num_experts'])} "
          f"to {hi['tokens_fully_resident_pct']:.1f}% at N={int(hi['num_experts'])}.")
    print(f"CXL offload at N={int(hi['num_experts'])}: "
          f"{hi['cxl_offload_gb_all_layers']:.1f} GB across {NUM_LAYERS} layers.")
    fixed = df[(df["budget_fraction"] == 0.5) & (df["regime"] == "fixed-k (top-2)")]
    flo, fhi = fixed.iloc[0], fixed.iloc[-1]
    print(f"\nAt a 50% HBM budget, fixed top-2: {flo['tokens_fully_resident_pct']:.1f}% "
          f"at N={int(flo['num_experts'])} -> {fhi['tokens_fully_resident_pct']:.1f}% "
          f"at N={int(fhi['num_experts'])}.")
    print("\n" + "=" * 68)
    print("FINDING: top_k dominates, not N.")
    print("=" * 68)
    print("Holding top-2 fixed, raising N from 8 to 128 changes token-level")
    print("residency only modestly. But letting k scale with N halves it, because")
    print("a token is only served from HBM when ALL k of its experts are resident,")
    print("and that probability decays geometrically in k -- roughly p^k, not p.")
    print("")
    print("So granular MoE is NOT automatically a better fit for expert tiering.")
    print("A DeepSeek-style top-8 model is HARDER to tier than Mixtral's top-2,")
    print("despite routing to a far smaller fraction of its experts. The design")
    print("axis that matters for tiering is top_k, and that is the recommendation")
    print("to put in the report.")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    for ax, regime in zip(axes, df["regime"].unique()):
        sub = df[df["regime"] == regime]
        for frac, colour in zip(BUDGET_FRACTIONS, ["#C44E52", "#DD8452", "#4C72B0"]):
            s = sub[sub["budget_fraction"] == frac]
            ax.plot(s["num_experts"], s["tokens_fully_resident_pct"], marker="o",
                    color=colour, label=f"HBM budget = {frac:.0%} of experts")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Number of experts per layer")
        ax.set_title(regime)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Tokens fully served from HBM (%)")
    axes[1].legend(fontsize=8)
    fig.suptitle("Tiering headroom vs. expert granularity", y=1.02)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "scalability_sweep.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
