"""
Real Mixtral Trace Benchmark
==============================
Runs the three placement policies over the recorded Mixtral 8x7B routing
traces and reports HBM hit rate per layer.

FAIRNESS FIX (important -- read before comparing against older numbers)
-----------------------------------------------------------------------
The first version of this script profiled the top-k experts over the
ENTIRE trace and then scored that same trace. That is look-ahead: the
static policy got to see the future, while LRU and periodic re-profiling
did not. It inflated static's hit rate by an unknown amount and made the
comparison invalid.

This version profiles static placement on a held-out PREFIX of the trace
(default: the first 10%) and scores every policy on the remainder only,
so all three face the identical, unseen evaluation window. The old
oracle figure is still computed and printed alongside, labelled as such,
so the size of the bias is visible rather than hidden -- but the honest
number is the one to quote.

Numbers published before this fix (Static 57.51% at layer 15, 65.76% at
layer 31) came from the oracle variant and should be regenerated.
"""

import argparse
from collections import OrderedDict

import numpy as np
import pandas as pd

import paths

NUM_EXPERTS = 8       # Mixtral 8x7B
CAPACITY_K = 4        # HBM sized for half the experts
PROFILE_FRACTION = 0.10
REPROFILE_INTERVALS = [100, 500, 1000, 2000, 5000, 10000, 20000]


def _top_k(experts, num_experts, k):
    """The k most frequently accessed experts in `experts`, ties by id."""
    counts = np.bincount(np.asarray(experts, dtype=np.int64), minlength=num_experts)
    return set(int(e) for e in np.argsort(-counts, kind="stable")[:k])


def eval_static(train, test, num_experts, k):
    """Profile once on the held-out prefix, then freeze for the eval window."""
    resident = _top_k(train, num_experts, k)
    hits = sum(1 for e in test if e in resident)
    return hits / len(test), resident


def eval_static_oracle(test, num_experts, k):
    """The old, unfair variant: profile on the evaluation window itself."""
    resident = _top_k(test, num_experts, k)
    return sum(1 for e in test if e in resident) / len(test)


def eval_lru(test, k):
    """Recency-driven cache. O(1) per access via an OrderedDict."""
    cache = OrderedDict()
    hits = 0
    for e in test:
        if e in cache:
            cache.move_to_end(e)
            hits += 1
        else:
            cache[e] = True
            if len(cache) > k:
                cache.popitem(last=False)
    return hits / len(test)


def eval_periodic(test, num_experts, k, interval, warm_start=None):
    """
    Re-rank every `interval` accesses using only the window that just
    finished, and hold that ranking for the next window. Never looks
    ahead.

    The very first window has nothing to rank from. Previously it used
    experts 0..k-1, which is an arbitrary guess that happens to be a good
    one when low-numbered experts are hot. It now starts from the same
    held-out profile the static policy gets (`warm_start`), so the two
    differ only in whether they keep adapting -- which is the comparison
    we actually want to make.
    """
    resident = set(warm_start) if warm_start else set(range(k))
    hits = 0
    n = len(test)
    arr = np.asarray(test, dtype=np.int64)

    for start in range(0, n, interval):
        window = arr[start:start + interval]
        hits += int(np.isin(window, list(resident)).sum())
        resident = _top_k(window, num_experts, k)

    return hits / n


def run_layer(trace_path, layer_name, num_experts=NUM_EXPERTS, k=CAPACITY_K):
    df = pd.read_csv(trace_path)
    experts = df["expert_1"].to_numpy()

    split = int(len(experts) * PROFILE_FRACTION)
    train, test = experts[:split], experts[split:]

    static_rate, profile = eval_static(train, test, num_experts, k)
    oracle_rate = eval_static_oracle(test, num_experts, k)
    lru_rate = eval_lru(test, k)

    periodic = {i: eval_periodic(test, num_experts, k, i, warm_start=profile)
                for i in REPROFILE_INTERVALS}
    best_interval = max(periodic, key=periodic.get)

    print(f"\n=== {layer_name} ===")
    print(f"  trace           : {trace_path}")
    print(f"  tokens          : {len(experts):,}  "
          f"(profile on first {len(train):,}, score on remaining {len(test):,})")
    print(f"  experts / HBM k : {num_experts} / {k}")
    print(f"  {'Pure Static (held-out profile)':<34} {static_rate * 100:6.2f}%")
    print(f"  {'Pure LRU':<34} {lru_rate * 100:6.2f}%")
    print(f"  {f'Periodic (every {best_interval})':<34} {periodic[best_interval] * 100:6.2f}%")
    print(f"  {'-' * 34} {'-' * 7}")
    print(f"  {'Static, ORACLE profile (unfair)':<34} {oracle_rate * 100:6.2f}%"
          f"   <- bias = {(oracle_rate - static_rate) * 100:+.2f} pts")
    print(f"\n  Periodic interval sweep:")
    for i, r in periodic.items():
        mark = "  <- best" if i == best_interval else ""
        print(f"    every {i:>6,} accesses : {r * 100:6.2f}%{mark}")

    rows = [
        {"Layer": layer_name, "Strategy": "Pure Static", "HBM Hit Rate (%)": static_rate * 100},
        {"Layer": layer_name, "Strategy": "Pure LRU", "HBM Hit Rate (%)": lru_rate * 100},
        {"Layer": layer_name, "Strategy": "Periodic Re-profile",
         "HBM Hit Rate (%)": periodic[best_interval] * 100},
    ]
    for r in rows:
        r["Best Reprofile Interval"] = best_interval if r["Strategy"] == "Periodic Re-profile" else ""
        r["Tokens Scored"] = len(test)
    rows.append({"Layer": layer_name, "Strategy": "Pure Static (ORACLE, unfair)",
                 "HBM Hit Rate (%)": oracle_rate * 100,
                 "Best Reprofile Interval": "", "Tokens Scored": len(test)})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--layer", action="append", nargs=2, metavar=("NAME", "FILE"),
                    help="layer name and trace filename under data/ "
                         "(repeatable; defaults to layers 15 and 31)")
    ap.add_argument("--capacity", type=int, default=CAPACITY_K)
    ap.add_argument("--num-experts", type=int, default=NUM_EXPERTS)
    args = ap.parse_args()

    layers = args.layer or [
        ("Layer 15", "real_expert_trace.csv"),
        ("Layer 31", "real_expert_trace_layer31.csv"),
    ]

    all_rows = []
    for name, fname in layers:
        all_rows += run_layer(paths.require_data(fname), name,
                              args.num_experts, args.capacity)

    out = paths.result("real_trace_hit_rates.csv")
    pd.DataFrame(all_rows).to_csv(out, index=False)
    print(f"\nWrote: {out}")


if __name__ == "__main__":
    main()
