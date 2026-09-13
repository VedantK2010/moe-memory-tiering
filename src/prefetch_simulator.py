"""
Markov Prefetching for Expert Tiering -- a negative result
===========================================================
A miss only hurts if you have to wait for it. If the next expert can be
predicted while the current one is still computing, its weights can be
pulled across CXL early and the link latency disappears behind useful work.

WHY THE TIMING IS PLAUSIBLE
---------------------------
In batch-1 decode a single token takes roughly 25-30 ms (the whole model's
active weights stream from memory). Pulling one 352 MB expert across a
64 GB/s CXL link takes ~5.5 ms. So there IS a real prefetch window, which is
not true of most cache-prefetch schemes. Whether prediction is good enough to
use it is what this script measures.

THE PREDICTOR
-------------
An order-N Markov model trained online -- no offline training pass, no
look-ahead -- running alongside an LRU-managed HBM tier. It works at TOKEN
granularity: the router picks all of a token's top-k experts at once, so the
only real prefetch window is between tokens. After each token the model
guesses one expert the NEXT token will need, conditioned on the first-choice
experts of the last `order` tokens. (Treating the flattened access stream as
a sequence would let it "predict" a token's second expert from its first --
information it never has in time -- and inflates the gain.)

THREE FIXES OVER THE FIRST VERSION
----------------------------------
1. Cold-start bias. The original took the argmax over an all-zero
   transition row, which always returned expert 0, so early in the run it
   prefetched expert 0 continuously and scored free hits whenever expert 0
   happened to be next. An untrained context now predicts nothing, and a
   context is only trusted once it has `min_support` observations.
2. Wrong guesses were free. A prefetch moves an expert across CXL whether
   or not the guess was right. Every issued prefetch is now charged in the
   CXL-traffic and energy columns. So prefetching can never SAVE traffic or
   energy -- a correct prefetch moves the same bytes, only earlier -- it can
   only hide latency, and the columns show what that costs. Latency is
   reported optimistically (a correct prefetch fully hides the fetch), so the
   hit-rate gain is an UPPER BOUND.
3. One cost model. Time and energy per expert fetch come from config.py
   (real 352 MB expert, DRAMSim3-measured pJ/bit), like every other script.
   Real traces are fed as the full top-2 access stream, not expert_1 only.

THE DATASET SUITE
-----------------
Both real Mixtral layers, plus synthetic traces that vary skew, locality,
expert count and top-k. The fine-grained ones stand in for DeepSeek/Qwen-style
layers by EXPERT COUNT AND TOP-K ONLY -- they are a sensitivity study over
topology, not measurements of those models. HBM holds half the experts in
every dataset (4 of 8 for Mixtral, matching the headline budget).

Outputs:
  results/prefetch_multi_dataset.csv
"""

import argparse
from collections import OrderedDict

import pandas as pd

from config import (
    RESULTS_DIR, ENCODING, REAL_TRACES,
    HBM_TIME_NS, CXL_TIME_NS, HBM_EXPERT_FETCH_PJ, CXL_EXPERT_FETCH_PJ,
    CXL_ENERGY_BASIS,
)
from generate_trace import generate_stationary

PJ_PER_MJ = 1e9


class MarkovPrefetcher:
    """
    Order-N Markov predictor over tokens.

    The context is the first-choice expert of each of the last `order` tokens;
    the successor counts record which experts the following token used.
    order=2 captures alternating patterns (A B A B) that order=1 averages
    away. Higher orders are sparser, so `min_support` guards against acting on
    a context seen only a few times.
    """

    def __init__(self, num_experts, order=1, min_support=8, min_confidence=0.30):
        self.num_experts = num_experts
        self.order = order
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.counts = {}   # context tuple -> per-expert count of "next token used it"
        self.totals = {}   # context tuple -> tokens observed after this context
        self.history = ()

    def observe(self, token_experts):
        """Record which experts followed the current context, then advance it."""
        if len(self.history) == self.order:
            row = self.counts.get(self.history)
            if row is None:
                row = self.counts[self.history] = [0] * self.num_experts
            for e in token_experts:
                row[e] += 1
            self.totals[self.history] = self.totals.get(self.history, 0) + 1
        self.history = (self.history + (token_experts[0],))[-self.order:]

    def predict(self):
        """Best guess at an expert the next token will use, or None when the
        context has not been seen often enough or no successor is clearly
        favoured. Returning None is what stops the predictor from spending CXL
        bandwidth on noise. Confidence = share of past next-tokens that used
        the guessed expert."""
        total = self.totals.get(self.history, 0)
        if total == 0 or total < self.min_support:
            return None
        row = self.counts[self.history]
        best = max(range(self.num_experts), key=row.__getitem__)
        if row[best] / total < self.min_confidence:
            return None
        return best


def _energy_mj(hbm_fetches, cxl_fetches):
    return (hbm_fetches * HBM_EXPERT_FETCH_PJ + cxl_fetches * CXL_EXPERT_FETCH_PJ) / PJ_PER_MJ


def simulate_prefetch(tokens, num_experts, capacity_k, order=1,
                      min_support=8, min_confidence=0.30):
    """
    LRU-managed HBM with a Markov prefetcher running alongside it.
    `tokens` is a list of per-token expert lists (routing order).

    Per access within a token:
      - resident in HBM                                  -> hit
      - not resident, but the prefetch issued after the
        previous token guessed it                        -> hit, latency hidden
      - otherwise                                        -> demand miss, CXL fetch
    One prefetch at most is issued between consecutive tokens.
    """
    cache = OrderedDict()
    model = MarkovPrefetcher(num_experts, order, min_support, min_confidence)

    in_flight = None
    lru_hits = prefetch_hits = demand_misses = issued = n = 0

    for experts in tokens:
        for expert_id in experts:
            n += 1
            if expert_id in cache:
                cache.move_to_end(expert_id)
                lru_hits += 1
            else:
                if expert_id == in_flight:
                    prefetch_hits += 1    # the guess paid off; latency hidden
                    in_flight = None
                else:
                    demand_misses += 1    # stall for the full CXL fetch
                cache[expert_id] = True
                if len(cache) > capacity_k:
                    cache.popitem(last=False)

        model.observe(experts)

        # Only worth issuing if the prediction is not already resident.
        guess = model.predict()
        in_flight = guess if (guess is not None and guess not in cache) else None
        if in_flight is not None:
            issued += 1

    # Critical-path time: a correct prefetch is used at HBM speed.
    total_time_ns = (lru_hits + prefetch_hits) * HBM_TIME_NS + demand_misses * CXL_TIME_NS
    # Every prefetch crosses the link whether or not it was used, as does
    # every demand miss. This is the traffic and energy actually spent.
    cxl_transfers = demand_misses + issued

    return {
        "accesses": n,
        "effective_hit_rate_pct": (lru_hits + prefetch_hits) / n * 100,
        "prefetches_issued": issued,
        "prefetches_wasted": issued - prefetch_hits,
        "prefetch_accuracy_pct": (prefetch_hits / issued * 100) if issued else 0.0,
        "avg_time_ns": total_time_ns / n,
        "cxl_transfers": cxl_transfers,
        "energy_mj_per_access": _energy_mj(lru_hits + prefetch_hits, cxl_transfers) / n,
    }


def simulate_lru_only(tokens, capacity_k):
    """Baseline: the same cache with the prefetcher switched off."""
    cache, hits, n = OrderedDict(), 0, 0
    for experts in tokens:
        for e in experts:
            n += 1
            if e in cache:
                cache.move_to_end(e)
                hits += 1
            else:
                cache[e] = True
                if len(cache) > capacity_k:
                    cache.popitem(last=False)
    misses = n - hits
    return {
        "hit_rate_pct": hits / n * 100,
        "avg_time_ns": (hits * HBM_TIME_NS + misses * CXL_TIME_NS) / n,
        "cxl_transfers": misses,
        "energy_mj_per_access": _energy_mj(hits, misses) / n,
    }


def _tokens(trace_df):
    """Per-token expert lists, in routing order (expert_1, expert_2, ...)."""
    cols = [c for c in trace_df.columns if c.startswith("expert_")]
    return trace_df[cols].to_numpy().tolist()


SYNTHETIC_SUITE = [
    # name,                               experts, top_k, alpha, p_repeat, tokens, seed
    ("Synthetic Mixtral-like",                  8,     2,     6,     0.27, 60_000,   42),
    ("Synthetic high-skew",                     8,     2,     3,     0.20, 60_000,   99),
    ("Synthetic low-locality",                  8,     2,     6,     0.14, 60_000,  123),
    ("Synthetic fine-grained (64e/top8)",      64,     8,     4,     0.25, 30_000,  321),
    ("Synthetic fine-grained (60e/top4)",      60,     4,     4,     0.25, 30_000,  654),
]

# How willing the prefetcher is to act on a weak prediction. "aggressive" is
# closest to the original implementation (predict always); the gates below
# progressively require more evidence before issuing.
AGGRESSIVENESS = [
    ("aggressive",   1, 0.00),
    ("permissive",   2, 0.10),
    ("moderate",     8, 0.20),
    ("conservative", 8, 0.30),
    ("strict",      20, 0.40),
]


def load_datasets():
    """Every dataset we can build, real traces included if present."""
    out = []
    for name, path in REAL_TRACES.items():
        if path.exists():
            out.append((f"Mixtral {name.replace('Layer ', 'L')} (real)",
                        _tokens(pd.read_csv(path)), 8, 2))
        else:
            print(f"  (skipping {name}: {path.name} not in data/)")
    for name, ne, tk, alpha, prep, toks, seed in SYNTHETIC_SUITE:
        df, _ = generate_stationary(num_tokens=toks, num_experts=ne, top_k=tk,
                                    skew_alpha=alpha, p_repeat=prep, seed=seed)
        out.append((name, _tokens(df), ne, tk))
    return out


def main():
    ap = argparse.ArgumentParser(description="Markov prefetching across a dataset suite")
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
        print(f"    {len(seq):,} tokens ({len(seq) * top_k:,} accesses) | "
              f"{num_experts} experts, top-{top_k} | HBM k={capacity_k}")
        print(f"    LRU only: {base['hit_rate_pct']:.2f}% hit, "
              f"{base['energy_mj_per_access']:.2f} mJ/access, {base['cxl_transfers']:,} CXL transfers")
        print(f"    {'order':>5} {'gating':>13} {'hit':>7} {'gain':>7} {'acc':>6} "
              f"{'CXL':>6} {'energy':>8}")

        for order in args.orders:
            for gate, support, conf in AGGRESSIVENESS:
                r = simulate_prefetch(seq, num_experts, capacity_k, order=order,
                                      min_support=support, min_confidence=conf)
                gain = r["effective_hit_rate_pct"] - base["hit_rate_pct"]
                traffic = r["cxl_transfers"] / max(1, base["cxl_transfers"])
                e_ratio = r["energy_mj_per_access"] / base["energy_mj_per_access"]
                print(f"    {order:>5} {gate:>13} {r['effective_hit_rate_pct']:>6.2f}% "
                      f"{gain:>+6.2f} {r['prefetch_accuracy_pct']:>5.1f}% "
                      f"{traffic:>5.2f}x {e_ratio:>7.2f}x")
                rows.append({
                    "dataset": name, "experts": num_experts, "top_k": top_k,
                    "hbm_capacity": capacity_k, "tokens": len(seq),
                    "accesses": r["accesses"],
                    "markov_order": order, "gating": gate,
                    "min_support": support, "min_confidence": conf,
                    "lru_hit_rate_pct": base["hit_rate_pct"],
                    "prefetch_hit_rate_pct": r["effective_hit_rate_pct"],
                    "hit_rate_gain_pts": gain,
                    "prefetch_accuracy_pct": r["prefetch_accuracy_pct"],
                    "prefetches_issued": r["prefetches_issued"],
                    "prefetches_wasted": r["prefetches_wasted"],
                    "cxl_traffic_multiplier": traffic,
                    "energy_multiplier": e_ratio,
                    "lru_energy_mj_per_access": base["energy_mj_per_access"],
                    "prefetch_energy_mj_per_access": r["energy_mj_per_access"],
                    "cxl_energy_basis": CXL_ENERGY_BASIS,
                })
        print()

    df = pd.DataFrame(rows)
    out = RESULTS_DIR / "prefetch_multi_dataset.csv"
    df.to_csv(out, index=False, encoding=ENCODING)

    best = df.loc[df["hit_rate_gain_pts"].idxmax()]
    print("=" * 74)
    print("VERDICT")
    print("=" * 74)
    print(f"Runs evaluated          : {len(df)}")
    print(f"Mean prefetch accuracy  : {df['prefetch_accuracy_pct'].mean():.1f}%")
    print(f"Largest hit-rate gain   : {best['hit_rate_gain_pts']:+.2f} pts "
          f"({best['dataset']}, order {best['markov_order']}, {best['gating']}) "
          f"at {best['cxl_traffic_multiplier']:.2f}x CXL traffic, "
          f"{best['energy_multiplier']:.2f}x energy")
    real = df[df["dataset"].str.contains("real")]
    for name, sub in real.groupby("dataset", sort=False):
        b = sub.loc[sub["hit_rate_gain_pts"].idxmax()]
        print(f"{name:<24}: best {b['hit_rate_gain_pts']:+.2f} pts at "
              f"{b['cxl_traffic_multiplier']:.2f}x CXL traffic, "
              f"{b['energy_multiplier']:.2f}x energy")
    print("\nPrefetching can only HIDE latency. It never reduces CXL traffic or")
    print("energy: a correct prefetch still crosses the link (just earlier) and a")
    print("wrong one crosses it for nothing. The question is the exchange rate --")
    print("latency hidden per extra byte moved -- and on these traces it is poor")
    print("for a bandwidth-constrained expansion link. Report as a negative result.")
    print(f"\nWrote: {out.name}")


if __name__ == "__main__":
    main()
