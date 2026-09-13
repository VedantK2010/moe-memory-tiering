"""
Experiment 9: CXL Memory Expansion and Pooling
===============================================
The one named deliverable with no coverage anywhere else in this project,
and the one closest to what CXL is actually sold to do.

THE SETUP
---------
A serving cluster runs R replicas of the same MoE model. Each replica needs
its hot experts in local HBM (private, per-replica, because they are being
read at HBM bandwidth continuously). The cold experts have to live
somewhere, and there are two ways to arrange that:

  PRIVATE CXL : each replica gets its own CXL device holding its own copy
                of the cold experts. Capacity scales as R x cold_bytes.

  POOLED CXL  : one CXL pool holds a SINGLE copy of the cold experts, shared
                read-only by all R replicas. Capacity is cold_bytes,
                independent of R.

Expert weights during inference are strictly read-only, which is what makes
the single shared copy legitimate rather than a trick.

THE TRADE-OFF (do not report the capacity win alone)
----------------------------------------------------
Pooling saves capacity but concentrates bandwidth. R replicas now contend
for one pool's links, so per-replica CXL bandwidth falls as 1/R and the
effective cold-fetch time rises. The honest result is the crossover: pooling
wins on cost until bandwidth contention makes the latency unacceptable, and
where that lands depends on how often you actually touch CXL -- which is
exactly the hit rate the rest of this study measures.

A pool with L links serves aggregate L x CXL_BANDWIDTH. We report the number
of links needed to keep per-replica bandwidth whole, since that is the real
provisioning question.

Outputs:
  results/pooling_study.csv
  results/pooling_study.png
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (
    RESULTS_DIR, ENCODING, NUM_EXPERTS, NUM_LAYERS, CAPACITY_K, TOP_K,
    EXPERT_SIZE_BYTES, EXPERT_BYTES_ALL_LAYERS,
    HBM_LATENCY_NS, HBM_BANDWIDTH_GBPS, CXL_LATENCY_NS, CXL_BANDWIDTH_GBPS,
    access_time_ns,
)

REPLICAS = [1, 2, 4, 8, 16, 32]
# Representative HBM hit rate, from the real Mixtral top-2 measurements.
HIT_RATE = 0.595


def analyse(capacity_k=CAPACITY_K, num_experts=NUM_EXPERTS, hit_rate=HIT_RATE):
    hot_bytes = EXPERT_SIZE_BYTES * capacity_k * NUM_LAYERS
    cold_bytes = EXPERT_SIZE_BYTES * (num_experts - capacity_k) * NUM_LAYERS

    rows = []
    for r in REPLICAS:
        private_total = r * (hot_bytes + cold_bytes)
        pooled_total = r * hot_bytes + cold_bytes
        saved = private_total - pooled_total

        # Bandwidth contention: R replicas share one pool's links.
        for links in (1, 2, 4, 8):
            per_replica_bw = CXL_BANDWIDTH_GBPS * links / r
            cxl_time_pooled = access_time_ns(CXL_LATENCY_NS, max(per_replica_bw, 1e-9),
                                             EXPERT_SIZE_BYTES)
            cxl_time_private = access_time_ns(CXL_LATENCY_NS, CXL_BANDWIDTH_GBPS,
                                              EXPERT_SIZE_BYTES)
            hbm_time = access_time_ns(HBM_LATENCY_NS, HBM_BANDWIDTH_GBPS,
                                      EXPERT_SIZE_BYTES)
            t_pooled = hit_rate * hbm_time + (1 - hit_rate) * cxl_time_pooled
            t_private = hit_rate * hbm_time + (1 - hit_rate) * cxl_time_private

            rows.append({
                "replicas": r,
                "pool_links": links,
                "private_cxl_gb": r * cold_bytes / 1e9,
                "pooled_cxl_gb": cold_bytes / 1e9,
                "capacity_saved_gb": saved / 1e9,
                "consolidation_ratio": private_total / pooled_total,
                "per_replica_pool_bw_gbps": per_replica_bw,
                "access_time_private_us": t_private / 1e3,
                "access_time_pooled_us": t_pooled / 1e3,
                "pooled_slowdown_x": t_pooled / t_private,
            })
    return pd.DataFrame(rows)


def main():
    df = analyse()
    df.to_csv(RESULTS_DIR / "pooling_study.csv", index=False, encoding=ENCODING)

    hot_gb = EXPERT_SIZE_BYTES * CAPACITY_K * NUM_LAYERS / 1e9
    cold_gb = EXPERT_SIZE_BYTES * (NUM_EXPERTS - CAPACITY_K) * NUM_LAYERS / 1e9
    print("=== CXL pooling: capacity ===")
    print(f"Per replica: {hot_gb:.1f} GB hot (private HBM) + {cold_gb:.1f} GB cold")
    print(f"Total expert weights per model: {EXPERT_BYTES_ALL_LAYERS/1e9:.1f} GB\n")

    cap = df[df["pool_links"] == 1][
        ["replicas", "private_cxl_gb", "pooled_cxl_gb", "capacity_saved_gb",
         "consolidation_ratio"]]
    print(cap.to_string(index=False, float_format=lambda v: f"{v:10.2f}"))

    big = cap.iloc[-1]
    print(f"\nAt {int(big['replicas'])} replicas, pooling removes "
          f"{big['capacity_saved_gb']:.0f} GB of duplicated cold-expert capacity "
          f"\n({big['private_cxl_gb']:.0f} GB private -> {big['pooled_cxl_gb']:.0f} GB "
          f"shared), a {big['consolidation_ratio']:.2f}x consolidation on total "
          f"memory.")

    print("\n=== The catch: bandwidth contention ===")
    print(f"(assuming a {HIT_RATE:.1%} HBM hit rate, our measured top-2 figure)\n")
    pivot = df.pivot_table(index="replicas", columns="pool_links",
                           values="pooled_slowdown_x")
    pivot.columns = [f"{c} link(s)" for c in pivot.columns]
    print(pivot.to_string(float_format=lambda v: f"{v:8.2f}x"))
    print("\nEach cell is the per-access slowdown versus giving every replica its")
    print("own private CXL device. 1.00x means pooling costs nothing.")

    ok = df[(df["pooled_slowdown_x"] <= 1.05)].groupby("replicas")["pool_links"].min()
    print("\nLinks needed to keep the pooling penalty under 5%:")
    for r, l in ok.items():
        print(f"  {r:>3} replicas -> {l} link(s)  "
              f"({CXL_BANDWIDTH_GBPS * l / r:.1f} GB/s per replica)")
    print("\nRECOMMENDATION: pooling is a capacity and cost play, not a performance")
    print("one. It pays when the HBM hit rate is high enough that CXL is touched")
    print("rarely -- which is precisely what the placement policy controls. Good")
    print("placement is what makes pooling affordable.")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    c = df[df["pool_links"] == 1]
    axes[0].plot(c["replicas"], c["private_cxl_gb"], marker="o",
                 color="#C44E52", label="Private CXL per replica")
    axes[0].plot(c["replicas"], c["pooled_cxl_gb"], marker="s",
                 color="#55A868", label="Pooled (one shared copy)")
    axes[0].set_xscale("log", base=2)
    axes[0].set_xlabel("Model replicas")
    axes[0].set_ylabel("Cold-expert capacity (GB)")
    axes[0].set_title("Pooling removes duplicated capacity")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    for links, colour in zip((1, 2, 4, 8), ["#C44E52", "#DD8452", "#8172B2", "#4C72B0"]):
        s = df[df["pool_links"] == links]
        axes[1].plot(s["replicas"], s["pooled_slowdown_x"], marker="o",
                     color=colour, label=f"{links} link(s)")
    axes[1].axhline(1.0, color="black", lw=0.8, ls="--")
    axes[1].set_xscale("log", base=2)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Model replicas sharing the pool")
    axes[1].set_ylabel("Access-time penalty vs private CXL")
    axes[1].set_title("...but concentrates bandwidth")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "pooling_study.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
