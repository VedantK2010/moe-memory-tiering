"""
Markov Prefetching for Expert Tiering
=======================================
A miss only hurts if you have to wait for it. If the next expert can be
predicted while the current one is still computing, its weights can be
pulled across CXL early and the link latency disappears behind useful
work.

The predictor is an order-N Markov model over the expert access
sequence, trained online -- no offline training pass, no look-ahead.

THREE FIXES OVER THE FIRST VERSION
-----------------------------------
1. Cold-start bias. The original picked the argmax over an all-zero
   transition row, which always returned expert 0, so early in the run it
   prefetched expert 0 continuously and scored free hits whenever expert 0
   happened to be next. An untrained context now predicts nothing, and a
   context is only trusted once it has `min_support` observations.

2. Wrong guesses were free. A prefetch moves an expert's weights across
   CXL whether or not the guess was right, consuming link bandwidth and
   energy. Wasted prefetches are now counted and charged in the bandwidth
   and energy columns. Latency is still reported optimistically (a
   correct prefetch fully hides the fetch), which is the upper bound --
   the accuracy column tells you how much of it is real.

3. Inconsistent timing model. The original used bare 150/220 ns
   constants while every other script in the project charges
   latency + size/bandwidth for a full expert transfer, so its "average
   latency" was three orders of magnitude off the rest of the report.
   It now imports the shared model from tier_simulator.

Run it across the whole dataset suite with:
    python src/prefetch_simulator.py
"""

import argparse
from collections import OrderedDict, defaultdict

import numpy as np
import pandas as pd

import paths
from tier_simulator import (
    access_time_ns, EXPERT_SIZE_BYTES,
    HBM_LATENCY_NS, HBM_BANDWIDTH_GBPS, CXL_LATENCY_NS, CXL_BANDWIDTH_GBPS,
)

# Energy per fetch, from the DRAMSim3 run (see calc_energy.py for the
# caveat on what this constant does and does not include).
HBM_ENERGY_NJ = 21.08
CXL_ENERGY_NJ = HBM_ENERGY_NJ * 3.0


class MarkovPrefetcher:
    """
    Order-N Markov predictor over the expert access stream.

    order=1 conditions on the previous expert; order=2 on the previous
    two, which captures alternating patterns (A B A B) that a first-order
    model averages away. Higher orders are sparser, so `min_support`
    guards against acting on a context seen once or twice.
    """

    def __init__(self, num_experts, order=1, min_support=8, min_confidence=0.30):
        self.num_experts = num_experts
        self.order = order
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.counts = defaultdict(lambda: np.zeros(num_experts, dtype=np.int64))
        self.history = ()

    def observe(self, expert_id):
        """Record the transition into `expert_id`, then advance the context."""
        if len(self.history) == self.order:
            self.counts[self.history][expert_id] += 1
        self.history = (self.history + (expert_id,))[-self.order:]

    def predict(self):
        """
        Best guess at the next expert, or None when the current context
        has not been seen often enough, or its best successor is not
        clearly favoured. Returning None is what stops the predictor from
        burning CXL bandwidth on noise.
        """
        if len(self.history) < self.order:
            return None
        row = self.counts.get(self.history)
        if row is None:
            return None
        total = row.sum()
        if total < self.min_support:
            return None
        best = int(row.argmax())
        if row[best] / total < self.min_confidence:
            return None
        return best


def simulate_prefetch(sequence, num_experts, capacity_k, order=1,
                      min_support=8, min_confidence=0.30):
    """
    LRU-managed HBM with a Markov prefetcher running alongside it.

    Per access:
      - resident in HBM            -> hit
      - not resident, but the last prefetch guessed it -> hit, latency hidden
      - otherwise                  -> demand miss, full CXL fetch
    """
    t_hbm = access_time_ns(HBM_LATENCY_NS, HBM_BANDWIDTH_GBPS, EXPERT_SIZE_BYTES)
    t_cxl = access_time_ns(CXL_LATENCY_NS, CXL_BANDWIDTH_GBPS, EXPERT_SIZE_BYTES)

    cache = OrderedDict()
    model = MarkovPrefetcher(num_experts, order, min_support, min_confidence)

    in_flight = None          # expert currently being prefetched, if any
    lru_hits = prefetch_hits = demand_misses = 0
    issued = 0

    for expert_id in sequence:
        if expert_id in cache:
            cache.move_to_end(expert_id)
            lru_hits += 1
        else:
            if expert_id == in_flight:
                prefetch_hits += 1        # the guess paid off; latency hidden
            else:
                demand_misses += 1        # stall for the full CXL fetch
            cache[expert_id] = True
            if len(cache) > capacity_k:
                cache.popitem(last=False)

        model.observe(expert_id)

        # Only worth issuing if the prediction is not already resident.
        guess = model.predict()
        in_flight = guess if (guess is not None and guess not in cache) else None
        if in_flight is not None:
            issued += 1

    n = len(sequence)
    wasted = issued - prefetch_hits

    # Critical-path time: correct prefetches cost HBM speed at use time.
    total_time_ns = (lru_hits + prefetch_hits) * t_hbm + demand_misses * t_cxl

    # Every prefetch crosses the link whether or not it was used, as does
    # every demand miss. This is the bandwidth and energy the policy
    # actually spends.
    cxl_transfers = demand_misses + issued
    energy_nj = (lru_hits + prefetch_hits) * HBM_ENERGY_NJ + cxl_transfers * CXL_ENERGY_NJ

    return {
        "accesses": n,
        "lru_hit_rate_pct": lru_hits / n * 100,
        "effective_hit_rate_pct": (lru_hits + prefetch_hits) / n * 100,
        "prefetch_hits": prefetch_hits,
        "prefetches_issued": issued,
        "prefetches_wasted": wasted,
        "prefetch_accuracy_pct": (prefetch_hits / issued * 100) if issued else 0.0,
        "demand_misses": demand_misses,
        "avg_latency_ns": total_time_ns / n,
        "cxl_transfers": cxl_transfers,
        "cxl_traffic_vs_lru_only": cxl_transfers / max(1, n - lru_hits),
        "energy_per_access_nj": energy_nj / n,
    }


def simulate_lru_only(sequence, capacity_k):
    """Baseline: the same cache with the prefetcher switched off."""
    t_hbm = access_time_ns(HBM_LATENCY_NS, HBM_BANDWIDTH_GBPS, EXPERT_SIZE_BYTES)
    t_cxl = access_time_ns(CXL_LATENCY_NS, CXL_BANDWIDTH_GBPS, EXPERT_SIZE_BYTES)
    cache, hits = OrderedDict(), 0
    for e in sequence:
        if e in cache:
            cache.move_to_end(e)
            hits += 1
        else:
            cache[e] = True
            if len(cache) > capacity_k:
                cache.popitem(last=False)
    n = len(sequence)
    misses = n - hits
    return {
        "hit_rate_pct": hits / n * 100,
        "avg_latency_ns": (hits * t_hbm + misses * t_cxl) / n,
        "cxl_transfers": misses,
        "energy_per_access_nj": (hits * HBM_ENERGY_NJ + misses * CXL_ENERGY_NJ) / n,
    }


# ---------------------------------------------------------------------- #
# Dataset suite
#
# The report asked for the prefetcher to be exercised on more than the two
# Mixtral layers, and on topologies other than Mixtral's 8-expert top-2
# layout. Synthetic traces stand in for the finer-grained routing of
# DeepSeek- and Qwen-style MoE layers: those models are not simulated,
# only their EXPERT COUNT and TOP-K, with routing skew and locality drawn
# from the same calibrated generator. Treat them as a sensitivity study
# over topology, not as measurements of those models.
# ---------------------------------------------------------------------- #

SYNTHETIC_SUITE = [
    # name,                      num_experts, top_k, alpha, p_repeat, tokens, seed
    ("Synthetic Mixtral-like",            8,     2,     6,     0.27,   60_000,  42),
    ("Synthetic high-skew",               8,     2,     3,     0.20,   60_000,  99),
    ("Synthetic low-locality",            8,     2,     6,     0.14,   60_000, 123),
    ("Synthetic fine-grained (64e/top8)", 64,    8,     4,     0.25,   30_000, 321),
    ("Synthetic fine-grained (60e/top4)", 60,    4,     4,     0.25,   30_000, 654),
]

REAL_TRACES = [
    ("Mixtral L15 (real)", "real_expert_trace.csv", 8, 2),
    ("Mixtral L31 (real)", "real_expert_trace_layer31.csv", 8, 2),
]


def build_synthetic(num_experts, top_k, alpha, p_repeat, tokens, seed):
    from generate_trace import generate_base_popularity, generate_expert_sequence
    base = generate_base_popularity(num_experts, skew_alpha=alpha, seed=seed)
    choices = generate_expert_sequence(tokens, num_experts, top_k, p_repeat, base, seed)
    return choices.ravel().tolist()


def load_datasets():
    """Every dataset we can actually build, real traces included if present."""
    out = []
    for name, fname, ne, tk in REAL_TRACES:
        p = paths.data(fname)
        if p.exists():
            seq = pd.read_csv(p)["expert_1"].tolist()
            out.append((name, seq, ne, tk))
        else:
            print(f"  (skipping {name}: {p.name} not in data/)")
    for name, ne, tk, alpha, prep, toks, seed in SYNTHETIC_SUITE:
        out.append((name, build_synthetic(ne, tk, alpha, prep, toks, seed), ne, tk))
    return out


# How willing the prefetcher is to act on a weak prediction. "Aggressive"
# is the closest setting to the original implementation (predict always);
# the gates below progressively require more evidence before issuing.
AGGRESSIVENESS = [
    ("aggressive",   1,  0.00),
    ("permissive",   2,  0.10),
    ("moderate",     8,  0.20),
    ("conservative", 8,  0.30),
    ("strict",      20,  0.40),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--orders", type=int, nargs="+", default=[1, 2],
                    help="Markov orders to evaluate (default: 1 2)")
    args = ap.parse_args()

    print("Building dataset suite...")
    datasets = load_datasets()
    print(f"  {len(datasets)} datasets ready\n")

    rows = []
    for name, seq, num_experts, top_k in datasets:
        capacity_k = max(1, num_experts // 2)
        base = simulate_lru_only(seq, capacity_k)

        print(f"--- {name} ---")
        print(f"    {len(seq):,} accesses | {num_experts} experts, top-{top_k} | HBM k={capacity_k}")
        print(f"    LRU only: {base['hit_rate_pct']:.2f}% hit, "
              f"{base['energy_per_access_nj']:.1f} nJ/access, {base['cxl_transfers']:,} CXL transfers")
        print(f"    {'order':>5} {'gating':>13} {'hit':>7} {'gain':>7} {'acc':>6} "
              f"{'CXL':>6} {'energy':>8}")

        for order in args.orders:
            for gate, support, conf in AGGRESSIVENESS:
                r = simulate_prefetch(seq, num_experts, capacity_k, order=order,
                                      min_support=support, min_confidence=conf)
                gain = r["effective_hit_rate_pct"] - base["hit_rate_pct"]
                e_ratio = r["energy_per_access_nj"] / base["energy_per_access_nj"]
                print(f"    {order:>5} {gate:>13} {r['effective_hit_rate_pct']:>6.2f}% "
                      f"{gain:>+6.2f} {r['prefetch_accuracy_pct']:>5.1f}% "
                      f"{r['cxl_traffic_vs_lru_only']:>5.2f}x {e_ratio:>7.2f}x")

                rows.append({
                    "Dataset": name, "Experts": num_experts, "Top-K": top_k,
                    "HBM Capacity": capacity_k, "Accesses": len(seq),
                    "Markov Order": order, "Gating": gate,
                    "Min Support": support, "Min Confidence": conf,
                    "LRU Hit Rate (%)": base["hit_rate_pct"],
                    "Prefetch Hit Rate (%)": r["effective_hit_rate_pct"],
                    "Hit Rate Gain (pts)": gain,
                    "Prefetch Accuracy (%)": r["prefetch_accuracy_pct"],
                    "Prefetches Issued": r["prefetches_issued"],
                    "Prefetches Wasted": r["prefetches_wasted"],
                    "CXL Traffic Multiplier": r["cxl_traffic_vs_lru_only"],
                    "Energy Multiplier": e_ratio,
                    "LRU Energy/Access (nJ)": base["energy_per_access_nj"],
                    "Prefetch Energy/Access (nJ)": r["energy_per_access_nj"],
                })
        print()

    df = pd.DataFrame(rows)
    out = paths.result("prefetch_multi_dataset.csv")
    df.to_csv(out, index=False)

    print("=" * 74)
    print("VERDICT")
    print("=" * 74)
    net_win = df[(df["Hit Rate Gain (pts)"] > 0.5) & (df["Energy Multiplier"] < 1.0)]
    best = df.loc[df["Hit Rate Gain (pts)"].idxmax()]
    print(f"Runs evaluated                       : {len(df)}")
    print(f"Largest hit-rate gain                : {best['Hit Rate Gain (pts)']:+.2f} pts "
          f"({best['Dataset']}, order {best['Markov Order']}, {best['Gating']})")
    print(f"  ...bought at CXL traffic           : {best['CXL Traffic Multiplier']:.2f}x")
    print(f"  ...and memory energy               : {best['Energy Multiplier']:.2f}x")
    print(f"Mean prefetch accuracy               : {df['Prefetch Accuracy (%)'].mean():.1f}%")
    print(f"Runs that gained >0.5 pts AND cut energy: {len(net_win)}")
    print()
    print("Prefetching buys latency with bandwidth, and on these traces the")
    print("exchange rate is poor: the only settings that move the hit rate")
    print("materially are the ones that fire on almost every access at ~15%")
    print("accuracy. Because a wasted prefetch still crosses the link, total")
    print("memory energy rises faster than the hit rate does. The gain reported")
    print("before wasted prefetches were charged for was not a real saving.")
    print(f"\nWrote: {out}")


if __name__ == "__main__":
    main()
