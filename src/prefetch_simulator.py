"""
EXPLORATORY: Markov-Chain Expert Prefetching
=============================================
STATUS: exploratory. Reports an UPPER BOUND, not a deployable result.
Do not put this on a headline slide without the caveats below.

IDEA
----
Expert routing has temporal structure (our real traces show a 23% consecutive
repeat rate at both layers). A first-order Markov chain over expert
transitions can predict the next token's expert and start pulling it from
CXL early, hiding the transfer behind the current token's compute.

WHY THE TIMING ACTUALLY WORKS
-----------------------------
This is the part worth defending out loud. In batch-1 decode a single token
takes roughly 25-30 ms (the whole model's active weights stream from memory).
Pulling one 352 MB expert across a 64 GB/s CXL link takes ~5.5 ms. So there
IS a real prefetch window -- roughly 5x more time than the transfer needs.
Prefetching an expert one token ahead is physically plausible, which is not
true of most cache-prefetch schemes.

THREE REASONS THIS IS AN UPPER BOUND
------------------------------------
1. NO BANDWIDTH CONTENTION. A prefetch consumes CXL bandwidth that the
   current token's own demand fetches are using. At batch 1 there is
   headroom; under load there is not.
2. NO CACHE POLLUTION ON A WRONG GUESS. A mispredicted prefetch is simply
   discarded here. A real prefetcher installs it, evicting something useful.
3. NO MIGRATION COST. Installing the prefetched expert into HBM is free in
   this model, as it is in the LRU baseline.

Correcting any of these lowers the number. Quote it as "up to", and say
which assumptions produce it.

Outputs:
  results/prefetch_exploratory.csv
"""

import numpy as np
import pandas as pd

from config import (
    RESULTS_DIR, ENCODING, NUM_EXPERTS, CAPACITY_K, REAL_TRACES,
    HBM_TIME_NS, CXL_TIME_NS,
)


def run_prefetch(trace_df, capacity_k=CAPACITY_K, num_experts=NUM_EXPERTS):
    """Top-1 sequence only: the Markov chain models token-to-token
    transitions of the first-choice expert."""
    experts = trace_df["expert_1"].to_numpy()

    cache, clock = {}, 0
    transitions = np.zeros((num_experts, num_experts), dtype=np.int64)
    seen = np.zeros(num_experts, dtype=bool)

    lru_hits = prefetch_hits = misses = 0
    predicted = None
    prev = None
    correct_predictions = 0
    predictions_made = 0

    for e in experts:
        clock += 1
        if e in cache:
            cache[e] = clock
            lru_hits += 1
        elif predicted is not None and e == predicted:
            # Correct prediction: the transfer was already in flight, so the
            # CXL latency is hidden and this behaves like an HBM access.
            prefetch_hits += 1
            if len(cache) >= capacity_k:
                del cache[min(cache, key=cache.get)]
            cache[e] = clock
        else:
            misses += 1
            if len(cache) >= capacity_k:
                del cache[min(cache, key=cache.get)]
            cache[e] = clock

        if predicted is not None:
            predictions_made += 1
            if predicted == e:
                correct_predictions += 1

        if prev is not None:
            transitions[prev, e] += 1

        # Predict the next expert. Only bother if it is NOT already resident
        # (no point prefetching something we already have). Require at least
        # one observed transition, so we do not "predict" expert 0 by virtue
        # of an all-zero row.
        row = transitions[e]
        if seen[e] and row.max() > 0:
            guess = int(np.argmax(row))
            predicted = guess if guess not in cache else None
        else:
            predicted = None
        seen[e] = True
        prev = e

    total = len(experts)
    effective = (lru_hits + prefetch_hits) / total
    time_ns = (lru_hits + prefetch_hits) * HBM_TIME_NS + misses * CXL_TIME_NS
    return {
        "accesses": total,
        "lru_only_hit_rate_pct": lru_hits / total * 100,
        "prefetch_hits": prefetch_hits,
        "effective_hit_rate_pct": effective * 100,
        "prediction_accuracy_pct": (correct_predictions / predictions_made * 100)
                                   if predictions_made else 0.0,
        "avg_time_ns": time_ns / total,
    }


def main():
    rows = []
    for name, path in REAL_TRACES.items():
        if not path.exists():
            continue
        r = run_prefetch(pd.read_csv(path))
        r["layer"] = name
        r["status"] = "UPPER BOUND - no bandwidth contention, no pollution, no migration cost"
        rows.append(r)

    if not rows:
        print("No real traces found -- skipping.")
        return

    df = pd.DataFrame(rows)[[
        "layer", "accesses", "lru_only_hit_rate_pct", "prefetch_hits",
        "effective_hit_rate_pct", "prediction_accuracy_pct", "avg_time_ns", "status",
    ]]
    df.to_csv(RESULTS_DIR / "prefetch_exploratory.csv", index=False, encoding=ENCODING)

    print("=== Markov prefetching (EXPLORATORY, top-1 routing) ===")
    for _, r in df.iterrows():
        print(f"\n{r['layer']}:")
        print(f"  LRU alone                : {r['lru_only_hit_rate_pct']:.2f}%")
        print(f"  + correct prefetches     : {r['prefetch_hits']:,}")
        print(f"  Effective hit rate       : {r['effective_hit_rate_pct']:.2f}%")
        print(f"  Next-expert prediction   : {r['prediction_accuracy_pct']:.2f}% accurate")
    print("\nThis is an UPPER BOUND. It charges nothing for prefetch bandwidth,")
    print("nothing for a wrong guess, and nothing for installing the expert in")
    print("HBM. Present it as headroom for future work, not as a result.")


if __name__ == "__main__":
    main()
