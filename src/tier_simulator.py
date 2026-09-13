"""
HBM vs HBM+CXL Memory Tiering Simulator (Lightweight, Pure Python)
====================================================================
This replaces DRAMSim3 for this project's purposes. Instead of a full
cycle-accurate DRAM simulator, this models memory access TIME using
published latency and bandwidth figures for HBM and CXL, applied to
our synthetic MoE expert-access trace.

This is a coarser-grained model than DRAMSim3 (it doesn't simulate bank
conflicts, row buffer hits/misses, refresh cycles, etc.) but it is
sufficient to answer this project's actual question: "what happens to
access time if some experts move from HBM to CXL, and how does that
change as we vary how many experts get to stay in HBM?"

CITED MEMORY PARAMETERS (August 2026):
  HBM latency   : ~150 ns average random-access latency
                  (typical HBM2e/HBM3 range reported across multiple
                  engineering sources, e.g. Wevolver HBM3 guide, 2025)
  HBM bandwidth : ~800 GB/s per stack
                  (HBM3 commonly cited in the 500-820+ GB/s range)
  CXL added latency : ~70 ns on top of DRAM-class latency
                  (Introl CXL memory expansion blog, 2026; also
                  consistent with "A Case for CXL-Centric Server
                  Processors", arXiv:2305.05033, which cites a commonly
                  reported ~70ns overhead figure)
  CXL bandwidth : ~64 GB/s per link
                  (CXL memory pooling performance reports, 2026)

These are defensible ballpark figures with sources, NOT vendor-exact
numbers for a specific chip -- call this out explicitly in your report.
"""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import paths

# ----------------------------------------------------------------------
# Memory tier parameters (cited above)
# ----------------------------------------------------------------------
HBM_LATENCY_NS = 150
HBM_BANDWIDTH_GBPS = 800

CXL_ADDED_LATENCY_NS = 70
CXL_LATENCY_NS = HBM_LATENCY_NS + CXL_ADDED_LATENCY_NS  # 220 ns
CXL_BANDWIDTH_GBPS = 64

# Placeholder size for a single expert's weights, in bytes. This is a
# stand-in figure (not tied to any specific model's exact FFN dimensions)
# -- swap in a precise number later if you calculate it from a specific
# reference model (e.g. Mixtral's per-expert FFN parameter count).
EXPERT_SIZE_BYTES = 16 * 1024 * 1024  # 16 MiB
EXPERT_SIZE_GB = EXPERT_SIZE_BYTES / (1024 ** 3)


def access_time_ns(latency_ns, bandwidth_gbps, size_bytes):
    """
    Time to complete one memory access = fixed latency (time to first
    byte) + transfer time (size / bandwidth).

    This is the standard simplified memory access time model used in
    systems papers: total_time = latency + (size / bandwidth).
    """
    size_gb = size_bytes / (1024 ** 3)
    transfer_time_s = size_gb / bandwidth_gbps
    transfer_time_ns = transfer_time_s * 1e9
    return latency_ns + transfer_time_ns


def assign_tiers(expert_ids_by_popularity, num_experts_in_hbm):
    """
    Given experts ranked from hottest to coldest, put the top
    `num_experts_in_hbm` in the HBM tier and the rest in the CXL tier.

    Returns a dict: {expert_id: 'HBM' or 'CXL'}
    """
    tier_map = {}
    for rank, expert_id in enumerate(expert_ids_by_popularity):
        tier_map[expert_id] = "HBM" if rank < num_experts_in_hbm else "CXL"
    return tier_map


def expert_columns(trace_df):
    """The expert-choice columns of a trace, in routing-preference order."""
    return [c for c in trace_df.columns if c.startswith("expert_")]


def access_counts(trace_df, num_experts=None):
    """
    Number of accesses to each expert across the whole trace.

    This is the only thing a STATIC tier assignment's cost depends on.
    A static placement charges every access to the same tier for the
    whole run, so the total time is a function of per-expert access
    counts alone -- the order the accesses arrived in cannot change it.
    Collapsing the trace to eight integers once, instead of re-walking
    millions of rows for every candidate placement, is what makes the
    sweeps below cheap. (Order-dependent policies such as LRU cannot use
    this and still walk the trace; see simulate_lru_cache.)
    """
    cols = expert_columns(trace_df)
    flat = trace_df[cols].to_numpy().ravel()
    if num_experts is None:
        num_experts = int(flat.max()) + 1
    return np.bincount(flat, minlength=num_experts)


def _simulate_from_counts(counts, tier_map):
    """Shared core of simulate(): cost a static placement from counts."""
    t_hbm = access_time_ns(HBM_LATENCY_NS, HBM_BANDWIDTH_GBPS, EXPERT_SIZE_BYTES)
    t_cxl = access_time_ns(CXL_LATENCY_NS, CXL_BANDWIDTH_GBPS, EXPERT_SIZE_BYTES)

    hbm_accesses = int(sum(c for e, c in enumerate(counts) if tier_map.get(e) == "HBM"))
    cxl_accesses = int(counts.sum()) - hbm_accesses

    hbm_time_ns = hbm_accesses * t_hbm
    cxl_time_ns = cxl_accesses * t_cxl
    total_time_ns = hbm_time_ns + cxl_time_ns
    total_accesses = hbm_accesses + cxl_accesses

    return {
        "total_time_ns": total_time_ns,
        "total_accesses": total_accesses,
        "avg_time_per_access_ns": total_time_ns / total_accesses,
        "hbm_accesses": hbm_accesses,
        "cxl_accesses": cxl_accesses,
        "hbm_time_ns": hbm_time_ns,
        "cxl_time_ns": cxl_time_ns,
    }


def simulate(trace_df, tier_map):
    """
    Total access time for a fixed (static) HBM/CXL assignment over the
    whole trace, with a per-tier breakdown.

    Numerically identical to walking the trace access by access -- see
    access_counts() for why the walk is unnecessary here.
    """
    return _simulate_from_counts(access_counts(trace_df, len(tier_map)), tier_map)


def rank_experts(trace_df, num_experts, by="first"):
    """
    Rank experts hottest-first for static placement.

    by="first" : rank on first-choice (top-1) selections only.
    by="all"   : rank on ALL accesses, including second-choice ones.

    "first" is the default because it is what produced every number in
    results/ and in the report, and changing it would silently invalidate
    them. It is, however, the less principled of the two: the simulator
    charges for second-choice accesses but this ranking ignores them, so
    an expert that is rarely first but often second can be sent to CXL
    despite carrying real traffic. Re-run the pipeline with by="all" to
    see the difference; expect the gap to widen as top_k grows.
    """
    if by == "all":
        counts = access_counts(trace_df, num_experts)
        ranked = list(np.argsort(-counts, kind="stable"))
    elif by == "first":
        first_col = expert_columns(trace_df)[0]
        ranked = trace_df[first_col].value_counts().index.tolist()
    else:
        raise ValueError(f"rank_experts: unknown by={by!r} (use 'first' or 'all')")

    # Experts that never appear still need a place in the ordering
    for e in range(num_experts):
        if e not in ranked:
            ranked.append(e)
    return [int(e) for e in ranked]


def sweep_hbm_capacity(trace_df, num_experts, rank_by="first"):
    """
    Run the simulation once for every possible number of experts kept
    in HBM (from 0 = everything in CXL, to num_experts = everything in
    HBM, matching the original HBM-only baseline). Returns a DataFrame
    of results for plotting.
    """
    ranked_experts = rank_experts(trace_df, num_experts, by=rank_by)
    counts = access_counts(trace_df, num_experts)

    results = []
    for k in range(num_experts + 1):  # k = number of experts kept in HBM
        tier_map = assign_tiers(ranked_experts, k)
        stats = _simulate_from_counts(counts, tier_map)
        stats["num_experts_in_hbm"] = k
        results.append(stats)

    return pd.DataFrame(results)


def plot_sweep(results_df, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(results_df["num_experts_in_hbm"], results_df["avg_time_per_access_ns"],
                 marker="o", color="#4C72B0")
    axes[0].set_xlabel("Number of experts kept in HBM (rest in CXL)")
    axes[0].set_ylabel("Avg access time (ns)")
    axes[0].set_title("Access latency vs. HBM capacity")
    axes[0].grid(alpha=0.3)

    axes[1].bar(results_df["num_experts_in_hbm"] - 0.2, results_df["hbm_accesses"],
                width=0.4, label="HBM accesses", color="#4C72B0")
    axes[1].bar(results_df["num_experts_in_hbm"] + 0.2, results_df["cxl_accesses"],
                width=0.4, label="CXL accesses", color="#DD8452")
    axes[1].set_xlabel("Number of experts kept in HBM")
    axes[1].set_ylabel("Access count")
    axes[1].set_title("Access split: HBM vs CXL")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)


def compare_smart_vs_random(trace_df, num_experts, num_random_trials=20, seed=42, rank_by="first"):
    """
    THE ACTUAL TEST OF WHETHER TIERING STRATEGY MATTERS.

    For each possible HBM capacity budget k (0..num_experts), compare:
      - "Smart" placement: hottest k experts (by access frequency) go in HBM
      - "Random" placement: a RANDOM set of k experts goes in HBM instead

    If smart placement gives lower average access time than random at the
    SAME k (same memory budget), that proves the hot/cold heuristic is
    doing real, useful work -- not just that "less CXL is faster," which
    is true regardless of which experts you pick.

    Random placement is run multiple times (num_random_trials) and
    averaged, since a single random draw could get lucky or unlucky.
    """
    rng = np.random.default_rng(seed)

    # Smart ranking: hottest first, based on actual observed frequency
    ranked_experts = rank_experts(trace_df, num_experts, by=rank_by)

    # Collapse the trace ONCE. Every placement below is static, so all
    # 189 simulations (9 budgets x (1 smart + 20 random)) read these same
    # eight integers instead of re-walking the trace each time.
    counts = access_counts(trace_df, num_experts)

    all_experts = list(range(num_experts))
    results = []

    for k in range(num_experts + 1):
        # Smart: top-k hottest experts in HBM
        smart_tier_map = assign_tiers(ranked_experts, k)
        smart_stats = _simulate_from_counts(counts, smart_tier_map)

        # Random: average over several random k-sized HBM subsets
        random_avg_times = []
        for trial in range(num_random_trials):
            random_hbm_set = set(rng.choice(all_experts, size=k, replace=False)) if k > 0 else set()
            random_tier_map = {e: ("HBM" if e in random_hbm_set else "CXL") for e in all_experts}
            random_stats = _simulate_from_counts(counts, random_tier_map)
            random_avg_times.append(random_stats["avg_time_per_access_ns"])

        random_mean = float(np.mean(random_avg_times))
        random_std = float(np.std(random_avg_times))

        results.append({
            "num_experts_in_hbm": k,
            "smart_avg_time_ns": smart_stats["avg_time_per_access_ns"],
            "random_avg_time_ns_mean": random_mean,
            "random_avg_time_ns_std": random_std,
            "smart_advantage_pct": (random_mean - smart_stats["avg_time_per_access_ns"]) / random_mean * 100
            if random_mean > 0 else 0.0,
        })

    return pd.DataFrame(results)


def plot_smart_vs_random(comparison_df, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(comparison_df["num_experts_in_hbm"], comparison_df["smart_avg_time_ns"],
                 marker="o", label="Smart (hot/cold) placement", color="#4C72B0")
    axes[0].errorbar(comparison_df["num_experts_in_hbm"], comparison_df["random_avg_time_ns_mean"],
                      yerr=comparison_df["random_avg_time_ns_std"],
                      marker="s", label="Random placement (avg \u00b1 std)", color="#DD8452", capsize=3)
    axes[0].set_xlabel("Number of experts kept in HBM (capacity budget)")
    axes[0].set_ylabel("Avg access time (ns)")
    axes[0].set_title("Smart vs. Random tiering, same capacity budget")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].bar(comparison_df["num_experts_in_hbm"], comparison_df["smart_advantage_pct"],
                color="#55A868")
    axes[1].set_xlabel("Number of experts kept in HBM (capacity budget)")
    axes[1].set_ylabel("Smart advantage over random (%)")
    axes[1].set_title("How much does hot/cold placement help?")
    axes[1].grid(alpha=0.3)
    axes[1].axhline(0, color="black", linewidth=0.8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)


def simulate_lru_cache(trace_df, capacity_k):
    """
    A DYNAMIC, locality-aware placement strategy: treat HBM as a
    fixed-size LRU (Least Recently Used) cache of experts, rather than a
    fixed, pre-decided set.

    Unlike the static "smart" strategy (which locks in a hot/cold split
    ONCE, based on overall frequency, and never changes it), this walks
    through the trace in true chronological order and, for every single
    expert access:
      - if that expert is already "in HBM" (in the cache) -> HBM-speed hit
      - if not -> CXL-speed access, and then the expert is moved into
        HBM, evicting whichever expert has gone longest without being
        used (if the cache is full)

    This directly exploits the temporal locality effect documented in
    Mixtral Table 5 (consecutive tokens reusing the same expert more
    than chance would predict) -- something the static frequency-based
    strategy cannot react to at all.

    SIMPLIFICATION: we do not model the cost of migrating an expert's
    weights INTO HBM on a miss (i.e. the write cost of installing it in
    the cache) -- only the read cost of fetching it from CXL. Call this
    out as a simplification in the report; a more complete model would
    add a migration penalty on every cache miss.
    """
    from collections import OrderedDict

    cache = OrderedDict()  # acts as our LRU structure: front = most recently used
    hits = 0
    misses = 0

    t_hbm = access_time_ns(HBM_LATENCY_NS, HBM_BANDWIDTH_GBPS, EXPERT_SIZE_BYTES)
    t_cxl = access_time_ns(CXL_LATENCY_NS, CXL_BANDWIDTH_GBPS, EXPERT_SIZE_BYTES)

    # Walk the trace in TRUE chronological order: for each token, visit
    # its experts in the order they were chosen (expert_1, then expert_2).
    # Flattening to a plain Python list first is a few times faster than
    # itertuples and keeps the loop body doing only cache work.
    for expert_id in access_sequence(trace_df):
        if expert_id in cache:
            # HBM hit -- already resident, and it becomes "most recent"
            cache.move_to_end(expert_id)
            hits += 1
        else:
            # CXL miss -- fetch from CXL, then install into the cache
            misses += 1
            cache[expert_id] = True
            if len(cache) > capacity_k:
                cache.popitem(last=False)  # evict least-recently-used (front of the dict)

    total_accesses = hits + misses
    total_time_ns = hits * t_hbm + misses * t_cxl
    return {
        "avg_time_per_access_ns": total_time_ns / total_accesses,
        "hit_rate_pct": hits / total_accesses * 100,
        "hits": hits,
        "misses": misses,
    }


def access_sequence(trace_df):
    """
    The trace flattened into one chronological list of expert accesses:
    for each token, expert_1 then expert_2, and so on.
    """
    return trace_df[expert_columns(trace_df)].to_numpy().ravel().tolist()


def lru_hit_rates_all_k(trace_df, num_experts):
    """
    LRU hit rate at EVERY capacity from 0..num_experts, in a single pass
    over the trace.

    LRU is a stack algorithm (Mattson et al., 1970): the set resident at
    capacity k is always a subset of the set resident at capacity k+1.
    That means an access is a hit at capacity k if and only if its
    *stack distance* -- the number of distinct experts touched since the
    last access to this one -- is strictly less than k. So one pass that
    records the stack-distance histogram gives the hit rate for every
    capacity at once, instead of re-running the whole trace once per
    capacity.

    Returns a numpy array of hit rates in percent, indexed by capacity.
    """
    stack = []                                  # most-recently-used first
    dist_hist = np.zeros(num_experts + 1, dtype=np.int64)  # last bin = cold miss
    total = 0

    for expert_id in access_sequence(trace_df):
        total += 1
        try:
            d = stack.index(expert_id)          # stack distance
            stack.pop(d)
        except ValueError:
            d = num_experts                     # never seen before: compulsory miss
        dist_hist[d] += 1
        stack.insert(0, expert_id)

    # hits at capacity k = accesses whose stack distance was < k
    hits_at_k = np.concatenate(([0], np.cumsum(dist_hist[:num_experts])))
    return hits_at_k / total * 100.0


def compare_three_strategies(trace_df, num_experts, num_random_trials=20, seed=42, rank_by="first"):
    """
    Compare all three placement strategies at every possible capacity
    budget k:
      1. Random      -- no intelligence, baseline
      2. Static Smart -- fixed hot/cold split by overall frequency
      3. LRU (dynamic) -- adapts continuously based on recency
    """
    static_random_df = compare_smart_vs_random(trace_df, num_experts, num_random_trials, seed, rank_by)

    # One stack-distance pass covers every capacity (see lru_hit_rates_all_k)
    hit_rates = lru_hit_rates_all_k(trace_df, num_experts)
    t_hbm = access_time_ns(HBM_LATENCY_NS, HBM_BANDWIDTH_GBPS, EXPERT_SIZE_BYTES)
    t_cxl = access_time_ns(CXL_LATENCY_NS, CXL_BANDWIDTH_GBPS, EXPERT_SIZE_BYTES)

    lru_df = pd.DataFrame({
        "num_experts_in_hbm": range(num_experts + 1),
        "lru_avg_time_ns": (hit_rates / 100.0) * t_hbm + (1 - hit_rates / 100.0) * t_cxl,
        "lru_hit_rate_pct": hit_rates,
    })

    combined = static_random_df.merge(lru_df, on="num_experts_in_hbm")
    return combined


def plot_three_strategies(combined_df, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(combined_df["num_experts_in_hbm"], combined_df["random_avg_time_ns_mean"],
                 marker="s", label="Random", color="#DD8452")
    axes[0].plot(combined_df["num_experts_in_hbm"], combined_df["smart_avg_time_ns"],
                 marker="o", label="Static Smart (frequency)", color="#4C72B0")
    axes[0].plot(combined_df["num_experts_in_hbm"], combined_df["lru_avg_time_ns"],
                 marker="^", label="LRU (dynamic, recency)", color="#55A868")
    axes[0].set_xlabel("HBM capacity budget (num experts)")
    axes[0].set_ylabel("Avg access time (ns)")
    axes[0].set_title("Three placement strategies compared")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(combined_df["num_experts_in_hbm"], combined_df["lru_hit_rate_pct"],
                 marker="^", color="#55A868")
    axes[1].axhline(100 / num_experts, color="gray", linestyle="--", label="Uniform baseline (1/N)")
    axes[1].set_xlabel("HBM capacity budget (num experts)")
    axes[1].set_ylabel("LRU cache hit rate (%)")
    axes[1].set_title("How often does the LRU cache already have what's needed?")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)


if __name__ == "__main__":
    trace_df = pd.read_csv(paths.require_data("expert_trace.csv"))
    num_experts = 8  # matches our generator's NUM_EXPERTS

    # --- Baseline comparison: HBM-only vs. an example tiered split ---
    all_experts = list(range(num_experts))
    hbm_only_tier_map = {e: "HBM" for e in all_experts}
    hbm_only_stats = simulate(trace_df, hbm_only_tier_map)

    example_k = 4  # keep 4 hottest experts in HBM, 4 coldest in CXL
    ranked = rank_experts(trace_df, num_experts)
    tiered_map = assign_tiers(ranked, example_k)
    tiered_stats = simulate(trace_df, tiered_map)

    print("=== HBM-only baseline ===")
    print(f"  Total time: {hbm_only_stats['total_time_ns']/1e6:.3f} ms")
    print(f"  Avg access time: {hbm_only_stats['avg_time_per_access_ns']:.2f} ns")

    print(f"\n=== Tiered ({example_k} experts in HBM, {num_experts-example_k} in CXL) ===")
    print(f"  Total time: {tiered_stats['total_time_ns']/1e6:.3f} ms")
    print(f"  Avg access time: {tiered_stats['avg_time_per_access_ns']:.2f} ns")
    print(f"  HBM accesses: {tiered_stats['hbm_accesses']}  |  CXL accesses: {tiered_stats['cxl_accesses']}")

    slowdown_pct = (tiered_stats["avg_time_per_access_ns"] / hbm_only_stats["avg_time_per_access_ns"] - 1) * 100
    print(f"\n  --> Tiering slows down average access by {slowdown_pct:.2f}% vs. HBM-only,")
    print(f"      while freeing {(num_experts - example_k)/num_experts*100:.0f}% of expert capacity from HBM.")

    # --- Full sweep across every possible HBM/CXL split ---
    results_df = sweep_hbm_capacity(trace_df, num_experts)
    results_df.to_csv(paths.result("tiering_sweep_results.csv"), index=False)
    plot_sweep(results_df, paths.result("tiering_sweep.png"))

    # --- THE ACTUAL TEST: does smart (hot/cold) placement beat random? ---
    comparison_df = compare_smart_vs_random(trace_df, num_experts, num_random_trials=20)
    comparison_df.to_csv(paths.result("smart_vs_random_results.csv"), index=False)
    plot_smart_vs_random(comparison_df, paths.result("smart_vs_random.png"))

    print("\n=== Smart (hot/cold) vs Random placement, same HBM budget ===")
    print(comparison_df.to_string(index=False))

    # --- STRETCH: does a dynamic, recency-aware LRU cache beat both? ---
    combined_df = compare_three_strategies(trace_df, num_experts, num_random_trials=20)
    combined_df.to_csv(paths.result("three_strategy_comparison.csv"), index=False)
    plot_three_strategies(combined_df, paths.result("three_strategy_comparison.png"))

    print("\n=== Random vs Static-Smart vs LRU (dynamic), same HBM budget ===")
    print(combined_df[["num_experts_in_hbm", "random_avg_time_ns_mean",
                        "smart_avg_time_ns", "lru_avg_time_ns", "lru_hit_rate_pct"]].to_string(index=False))

    print(f"\nWrote into {paths.RESULTS_DIR}:"
          f"\n       tiering_sweep_results.csv / .png"
          f"\n       smart_vs_random_results.csv / .png"
          f"\n       three_strategy_comparison.csv / .png")
