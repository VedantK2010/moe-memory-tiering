"""
Experiment 6: Where Expert Tiering Stops Paying (Batch Size)
=============================================================
THE OBJECTION THIS ANSWERS
--------------------------
Everything else in this study assumes batch-1 decode, where each token
streams its own routed experts and tiering has real leverage. The obvious
challenge from anyone who knows MoE serving is:

    "At batch size 32, nearly every expert is touched every step. Your
     hot/cold split collapses. Why does any of this matter?"

That challenge is correct, and the right response is to quantify exactly
where it becomes true rather than to avoid the topic.

THE MODEL
---------
At batch size B, one decode step processes B independent sequences. Each
token routes to TOP_K experts, so the step needs the UNION of experts
across all B tokens. Two things happen as B grows, pulling in opposite
directions:

  - The per-step working set grows toward all NUM_EXPERTS, so a fixed HBM
    budget covers a smaller fraction of it. Tiering helps less.
  - An expert fetched once serves every token in the batch that routed to
    it, so bytes-per-token FALLS. Memory pressure per token drops.

Both matter. The first is why tiering stops paying; the second is why
large-batch serving is memory-efficient in the first place.

SAMPLING
--------
Decode batches are B tokens from B *independent* sequences, not B
consecutive tokens from one. So each simulated step samples B token
positions uniformly at random from the trace. Using consecutive tokens
would inherit the trace's temporal locality and understate the union size,
flattering our own result.

An analytic reference is included: under independent uniform routing the
expected union size is

    E[|U|] = N * (1 - (1 - K/N)^B)

The real traces sit below this because routing is skewed and correlated.

Outputs:
  results/batch_sensitivity.csv
  results/batch_sensitivity.png
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (
    RESULTS_DIR, ENCODING, NUM_EXPERTS, TOP_K, CAPACITY_K, SEED,
    EXPERT_SIZE_BYTES, HBM_TIME_NS, CXL_TIME_NS, REAL_TRACES,
)
import tier_simulator as ts

BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128]
STEPS_PER_BATCH_SIZE = 3000


def analytic_union(batch_size, num_experts=NUM_EXPERTS, top_k=TOP_K):
    """Expected distinct experts per step under independent uniform routing."""
    return num_experts * (1 - (1 - top_k / num_experts) ** batch_size)


def measure(trace_df, capacity_k=CAPACITY_K, num_experts=NUM_EXPERTS,
            batch_sizes=BATCH_SIZES, steps=STEPS_PER_BATCH_SIZE, seed=SEED):
    rng = np.random.default_rng(seed)
    cols = ts.expert_columns(trace_df)
    tokens = trace_df[cols].to_numpy()
    n = len(tokens)

    # Static hot/cold split, chosen on total demand (the best case for tiering).
    ranked = ts.rank_experts(trace_df, num_experts)
    resident = np.zeros(num_experts, dtype=bool)
    resident[ranked[:capacity_k]] = True

    rows = []
    for B in batch_sizes:
        union_sizes = np.empty(steps)
        hbm_hits = np.empty(steps)
        for s in range(steps):
            idx = rng.integers(0, n, size=B)
            needed = np.unique(tokens[idx].reshape(-1))
            union_sizes[s] = needed.size
            hbm_hits[s] = resident[needed].sum()

        misses = union_sizes - hbm_hits
        # One step fetches each distinct expert once, from its own tier.
        step_time_tiered = hbm_hits * HBM_TIME_NS + misses * CXL_TIME_NS
        step_time_hbm_only = union_sizes * HBM_TIME_NS

        rows.append({
            "batch_size": B,
            "mean_union_size": union_sizes.mean(),
            "union_pct_of_all_experts": union_sizes.mean() / num_experts * 100,
            "analytic_uniform_union": analytic_union(B, num_experts, TOP_K),
            # Two different questions, both worth reporting:
            #   ..._fits_budget   -- COULD a k-expert budget hold this step's
            #                        working set at all? A capacity-only bound.
            #   ..._fully_resident-- does the ACTUAL placement hold all of it,
            #                        so the step never touches CXL? This is the
            #                        one that determines step latency.
            "pct_steps_union_fits_budget": float((union_sizes <= capacity_k).mean() * 100),
            "pct_steps_fully_resident": float((hbm_hits == union_sizes).mean() * 100),
            "hbm_hit_rate_pct": float(hbm_hits.sum() / union_sizes.sum() * 100),
            "bytes_per_token_mb": union_sizes.mean() * EXPERT_SIZE_BYTES / B / 1e6,
            "time_per_token_tiered_us": step_time_tiered.mean() / B / 1e3,
            "time_per_token_hbm_only_us": step_time_hbm_only.mean() / B / 1e3,
            "tiering_penalty_x": float(step_time_tiered.mean() / step_time_hbm_only.mean()),
        })
    return pd.DataFrame(rows)


def main():
    path = REAL_TRACES.get("Layer 15")
    if path is None or not path.exists():
        print("Real trace not found -- skipping batch sensitivity.")
        return
    trace = pd.read_csv(path)

    df = measure(trace)
    df.to_csv(RESULTS_DIR / "batch_sensitivity.csv", index=False, encoding=ENCODING)

    print(f"=== Batch-size sensitivity (Layer 15, HBM budget = {CAPACITY_K}/"
          f"{NUM_EXPERTS} experts) ===\n")
    show = df[["batch_size", "mean_union_size", "union_pct_of_all_experts",
               "pct_steps_union_fits_budget", "pct_steps_fully_resident",
               "hbm_hit_rate_pct", "bytes_per_token_mb", "tiering_penalty_x"]]
    print(show.to_string(index=False, float_format=lambda v: f"{v:10.2f}"))

    b1 = df.iloc[0]
    fits_gone = df[df["pct_steps_fully_resident"] < 1.0]
    print(f"\nAt batch 1: {b1['mean_union_size']:.2f} experts per step; the budget "
          f"COULD hold the working set for {b1['pct_steps_union_fits_budget']:.1f}% of "
          f"steps,\n           but the actual placement serves only "
          f"{b1['pct_steps_fully_resident']:.1f}% of steps without touching CXL.")
    if len(fits_gone):
        B = int(fits_gone.iloc[0]["batch_size"])
        print(f"By batch {B}: essentially NO step fits in HBM -- every step "
              f"touches CXL.")
    big = df.iloc[-1]
    print(f"At batch {int(big['batch_size'])}: {big['union_pct_of_all_experts']:.1f}% "
          f"of all experts touched per step, so a {CAPACITY_K}/{NUM_EXPERTS} budget "
          f"covers {big['hbm_hit_rate_pct']:.1f}% of them.")
    print(f"\nThe other half of the story: bytes fetched per token falls from "
          f"{b1['bytes_per_token_mb']:,.0f} MB at batch 1 to "
          f"{big['bytes_per_token_mb']:,.1f} MB at batch {int(big['batch_size'])} "
          f"({b1['bytes_per_token_mb']/big['bytes_per_token_mb']:.0f}x less), because "
          f"one fetch\nserves every token in the batch that routed to that expert.")
    print("\nCONCLUSION FOR THE REPORT: expert tiering is a SMALL-BATCH, "
          "latency-sensitive\ntechnique. State the regime explicitly rather than "
          "letting a reader find the edge.")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))

    axes[0].plot(df["batch_size"], df["mean_union_size"], marker="o",
                 color="#4C72B0", label="Measured (real routing)")
    axes[0].plot(df["batch_size"], df["analytic_uniform_union"], ls="--",
                 color="gray", label="Uniform-routing model")
    axes[0].axhline(CAPACITY_K, color="#C44E52", ls=":", label=f"HBM budget ({CAPACITY_K})")
    axes[0].set_xscale("log", base=2)
    axes[0].set_xlabel("Batch size")
    axes[0].set_ylabel("Distinct experts per step")
    axes[0].set_title("Per-step expert working set")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].plot(df["batch_size"], df["pct_steps_union_fits_budget"], marker="s",
                 ls="--", color="#BBBBBB", label="Budget could hold it")
    axes[1].plot(df["batch_size"], df["pct_steps_fully_resident"], marker="o",
                 color="#55A868", label="Actually all resident")
    axes[1].legend(fontsize=8)
    axes[1].set_xscale("log", base=2)
    axes[1].set_xlabel("Batch size")
    axes[1].set_ylabel("% of steps served entirely from HBM")
    axes[1].set_title("Where tiering stops paying")
    axes[1].grid(alpha=0.3)

    axes[2].plot(df["batch_size"], df["bytes_per_token_mb"], marker="o",
                 color="#8172B2")
    axes[2].set_xscale("log", base=2)
    axes[2].set_yscale("log")
    axes[2].set_xlabel("Batch size")
    axes[2].set_ylabel("Expert bytes fetched per token (MB)")
    axes[2].set_title("Batching amortises the fetch")
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "batch_sensitivity.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
