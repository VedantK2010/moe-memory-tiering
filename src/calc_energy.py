"""
Memory Energy from HBM Hit Rates
==================================
Converts each policy's HBM hit rate into total memory energy, using
per-fetch energy figures grounded in the DRAMSim3 run.

WHAT THE CONSTANT ACTUALLY IS (read before quoting the millijoules)
--------------------------------------------------------------------
21.08 nJ/fetch comes from dividing DRAMSim3's reported TOTAL energy
(674,878,320 pJ) by the request count (32,000). That total includes
background and refresh power accumulated over the whole simulated
window, not just the energy attributable to moving 64 bytes. A 64-byte
burst on real HBM2 costs on the order of a quarter of a nanojoule at
~4 pJ/bit, so this figure is roughly two orders of magnitude above the
marginal cost of a burst -- it is "energy per request in that window",
which is a different quantity.

That does NOT invalidate the comparison. Every policy is charged the
same constant, so it cancels in every ratio: the percentage savings
below are sound. The absolute millijoule column is not transferable to
real silicon and should be presented as a relative result only.

The CXL multiplier (3x) is a published ballpark for driving a PCIe Gen5
PHY rather than a local HBM stack. It is an assumption, not a
measurement, and the sensitivity sweep below shows how much the
conclusion depends on it.
"""

import argparse

import numpy as np
import pandas as pd

import paths

# Physical constants derived from the DRAMSim3 run (see caveat above)
HBM_ENERGY_NJ_PER_FETCH = 21.08
CXL_MULTIPLIER = 3.0

# Hit rates measured on the real Mixtral traces. These are the fallback
# values used when results/real_trace_hit_rates.csv has not been
# generated yet; run src/run_real_benchmark.py to refresh them.
#
# NOTE: these were produced before the look-ahead fix in
# run_real_benchmark.py, so the "Pure Static" figures are optimistic.
# Regenerate them when the real traces are available.
FALLBACK_HIT_RATES = {
    "Layer 15": {"Pure Static": 57.51, "Periodic Re-profile": 61.02, "Pure LRU": 64.94},
    "Layer 31": {"Pure Static": 65.76, "Pure LRU": 65.91, "Periodic Re-profile": 67.26},
}

TOTAL_TOKENS = 829_441


def load_hit_rates():
    """Prefer freshly measured hit rates; fall back to the published ones."""
    src = paths.RESULTS_DIR / "real_trace_hit_rates.csv"
    if src.exists():
        df = pd.read_csv(src)
        df = df[~df["Strategy"].str.contains("ORACLE")]
        out = {}
        for layer, grp in df.groupby("Layer"):
            out[layer] = dict(zip(grp["Strategy"], grp["HBM Hit Rate (%)"]))
        print(f"Using measured hit rates from {src.name}")
        return out
    print("Using published fallback hit rates "
          "(run src/run_real_benchmark.py to regenerate)")
    return FALLBACK_HIT_RATES


def energy_rows(hit_rates, total_tokens, cxl_multiplier):
    cxl_nj = HBM_ENERGY_NJ_PER_FETCH * cxl_multiplier
    rows = []
    for layer, strategies in hit_rates.items():
        for strategy, hit_pct in strategies.items():
            hit = hit_pct / 100.0
            hbm_fetches = total_tokens * hit
            cxl_fetches = total_tokens * (1 - hit)
            total_nj = hbm_fetches * HBM_ENERGY_NJ_PER_FETCH + cxl_fetches * cxl_nj
            rows.append({
                "Layer": layer,
                "Strategy": strategy,
                "HBM Hit Rate (%)": hit_pct,
                "Total Energy (mJ)": total_nj / 1e6,
                "Avg Energy per Token (nJ)": total_nj / total_tokens,
            })
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tokens", type=int, default=TOTAL_TOKENS)
    ap.add_argument("--cxl-multiplier", type=float, default=CXL_MULTIPLIER)
    args = ap.parse_args()

    hit_rates = load_hit_rates()
    df = pd.DataFrame(energy_rows(hit_rates, args.tokens, args.cxl_multiplier))

    # Savings relative to the worst policy in the same layer -- the ratio
    # that survives the caveat on the absolute constant.
    df["Saving vs worst (%)"] = df.groupby("Layer")["Total Energy (mJ)"].transform(
        lambda s: (1 - s / s.max()) * 100
    )

    out = paths.result("energy_metrics.csv")
    df.to_csv(out, index=False)
    print("\n" + df.to_string(index=False))

    # --- How much does the conclusion lean on the 3x CXL assumption? ---
    print("\n=== Sensitivity: best-vs-worst saving at different CXL multipliers ===")
    print(f"{'CXL x HBM':>10} {'Layer 15':>12} {'Layer 31':>12}")
    for mult in (1.5, 2.0, 3.0, 4.0, 6.0):
        s = pd.DataFrame(energy_rows(hit_rates, args.tokens, mult))
        cells = []
        for layer in sorted(hit_rates):
            e = s[s["Layer"] == layer]["Total Energy (mJ)"]
            cells.append(f"{(1 - e.min() / e.max()) * 100:11.2f}%")
        print(f"{mult:>9.1f}x " + " ".join(cells))
    print("\nThe saving scales with the multiplier but never reverses sign:")
    print("a better policy is a lower-energy policy at every assumption tested.")

    print(f"\nWrote: {out}")


if __name__ == "__main__":
    main()
