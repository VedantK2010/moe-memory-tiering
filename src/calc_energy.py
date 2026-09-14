"""
Experiment 5: Energy Cost of Expert Placement
==============================================
Converts measured hit rates into energy, on a pJ/bit basis.

WHAT CHANGED AND WHY
--------------------
The previous version multiplied a per-token count by 21.08 nJ, a constant
obtained by dividing DRAMSim3's `total_energy` by its request count. That
constant was 95.7% standby and refresh power -- energy the DRAM burns as a
function of TIME, not of work -- and 84.8% of it was consumed in epochs
after the trace had already finished. It would have doubled had the
simulator simply been left running longer. It was not an energy-per-access.

This version:
  1. Uses only the ACCESS-PROPORTIONAL components DRAMSim3 measured
     (read_energy + act_energy per DRAM read command), read by config.py
     from results/dramsim3_*_summary.csv -- never retyped.
  2. Scales by actual bytes moved, so the real 352 MB expert size is
     respected rather than a 64 B burst standing in for a whole expert.
  3. Accounts for TOP_K experts per token, not one.
  4. Reads hit rates from results/real_trace_results.csv instead of having
     them retyped as literals.

HONEST FRAMING OF THE RESULT
----------------------------
Because both tiers have a fixed pJ/bit and every access moves the same
number of bytes, energy per token is an exact linear function of hit rate --
just as access time is. The energy result is therefore NOT independent
evidence; it is the hit-rate result expressed in joules. That is still worth
reporting (it converts a cache statistic into a datacentre operating cost)
but it must not be presented as a second, corroborating finding.

CAVEAT: the CXL pJ/bit figure is only PARTLY measured. Its DRAM component
comes from a DRAMSim3 DDR4-3200 run (DRAMSim3 has no DDR5 config); its link
component (SerDes + PHY) is a cited value, not simulated.
config.CXL_ENERGY_BASIS states this, and this script labels every row with it.
Because published link energies span a wide range, every saving is also
re-computed over config.CXL_LINK_PJ_PER_BIT_SWEEP.

Outputs:
  results/energy_metrics.csv            at the cited link energy
  results/energy_link_sensitivity.csv   the same savings across the sweep
  results/energy_metrics.png
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (
    RESULTS_DIR, ENCODING, TOP_K, EXPERT_SIZE_BYTES,
    HBM_PJ_PER_BIT, CXL_PJ_PER_BIT, CXL_DRAM_PJ_PER_BIT, CXL_LINK_PJ_PER_BIT,
    CXL_LINK_PJ_PER_BIT_SWEEP, CXL_LINK_SOURCE,
    CXL_ENERGY_IS_MEASURED, CXL_ENERGY_BASIS,
    HBM_EXPERT_FETCH_PJ, CXL_EXPERT_FETCH_PJ,
    energy_pj, require_dramsim_results,
)

PJ_PER_MJ = 1e9


def energy_per_token_pj(hit_rate, cxl_fetch_pj=CXL_EXPERT_FETCH_PJ):
    """Energy to serve one token's TOP_K expert fetches at a given hit rate."""
    per_access = hit_rate * HBM_EXPERT_FETCH_PJ + (1 - hit_rate) * cxl_fetch_pj
    return per_access * TOP_K


def energy_table(sub, cxl_fetch_pj):
    """Energy per token and saving against the worst strategy in each layer."""
    df = pd.DataFrame({
        "layer": sub["layer"].values,
        "strategy": sub["strategy"].values,
        "hit_rate_pct": sub["hit_rate_pct"].values,
    })
    per_token = energy_per_token_pj(df["hit_rate_pct"] / 100.0, cxl_fetch_pj)
    df["energy_per_token_mj"] = per_token / PJ_PER_MJ
    df["energy_per_1k_tokens_j"] = per_token * 1000 / 1e12
    baseline = df.groupby("layer")["energy_per_token_mj"].transform("max")
    df["saving_vs_worst_pct"] = (baseline - df["energy_per_token_mj"]) / baseline * 100
    return df


def link_sensitivity(sub):
    """Every saving re-computed with the link energy swept. The cited value is
    one of the rows (is_cited), so it reproduces energy_metrics.csv exactly."""
    frames = []
    for link in sorted(set(CXL_LINK_PJ_PER_BIT_SWEEP) | {CXL_LINK_PJ_PER_BIT}):
        cxl_pj_per_bit = CXL_DRAM_PJ_PER_BIT + link
        t = energy_table(sub, energy_pj(EXPERT_SIZE_BYTES, cxl_pj_per_bit))
        t.insert(0, "is_cited", link == CXL_LINK_PJ_PER_BIT)
        t.insert(0, "cxl_pj_per_bit", cxl_pj_per_bit)
        t.insert(0, "link_pj_per_bit", link)
        t["energy_rank"] = t.groupby("layer")["energy_per_token_mj"].rank(method="min").astype(int)
        frames.append(t.drop(columns=["energy_per_1k_tokens_j"]))
    return pd.concat(frames, ignore_index=True)


def main():
    require_dramsim_results()
    src = RESULTS_DIR / "real_trace_results.csv"
    if not src.exists():
        print("real_trace_results.csv not found -- run real_benchmark.py first.")
        return

    results = pd.read_csv(src)
    # Energy is a per-token quantity, so use the top-2 (real workload) rows.
    sub = results[results["routing"] == "top-2"].copy()

    # Savings are quoted against the worst strategy within each layer.
    df = energy_table(sub, CXL_EXPERT_FETCH_PJ)
    df.insert(5, "cxl_energy_basis", CXL_ENERGY_BASIS)
    df.to_csv(RESULTS_DIR / "energy_metrics.csv", index=False, encoding=ENCODING)

    sens = link_sensitivity(sub)
    sens.to_csv(RESULTS_DIR / "energy_link_sensitivity.csv", index=False, encoding=ENCODING)

    print("=== Energy per token (top-2 routing, real Mixtral traces) ===")
    print(f"  HBM : {HBM_PJ_PER_BIT:.3f} pJ/bit  [MEASURED from DRAMSim3 HBM2]")
    print(f"  CXL : {CXL_PJ_PER_BIT:.3f} pJ/bit = DRAM {CXL_DRAM_PJ_PER_BIT:.3f} "
          f"[MEASURED, DDR4 proxy] + link {CXL_LINK_PJ_PER_BIT:.3f} [CITED: {CXL_LINK_SOURCE}]")
    print(f"  One expert fetch: HBM {HBM_EXPERT_FETCH_PJ/PJ_PER_MJ:.2f} mJ, "
          f"CXL {CXL_EXPERT_FETCH_PJ/PJ_PER_MJ:.2f} mJ "
          f"({EXPERT_SIZE_BYTES/1e6:.1f} MB x {TOP_K} per token)\n")
    print(df.to_string(index=False, float_format=lambda v: f"{v:10.3f}"))

    ranks = sens.pivot_table(index=["layer", "strategy"], columns="link_pj_per_bit",
                             values="energy_rank")
    stable = bool((ranks.nunique(axis=1) == 1).all())
    links = sorted(sens["link_pj_per_bit"].unique())
    print(f"\n=== Link-energy sensitivity ({links[0]:g}-{links[-1]:g} pJ/bit) ===")
    print(sens.pivot_table(index=["layer", "strategy"], columns="link_pj_per_bit",
                           values="saving_vs_worst_pct", sort=False)
          .to_string(float_format=lambda v: f"{v:6.2f}"))
    print(f"  Energy ranking of strategies identical at every link value: {stable}")
    print("\nNOTE: energy is an exact linear function of hit rate here, so this")
    print("      restates the hit-rate result in joules. It is not independent")
    print("      corroboration -- present it as a cost translation.")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    layers = list(dict.fromkeys(df["layer"]))
    pivot = df.pivot_table(index="strategy", columns="layer",
                           values="energy_per_token_mj", sort=False)
    pivot.plot(kind="barh", ax=ax, color=["#4C72B0", "#DD8452"][:len(layers)])
    ax.set_xlabel("Energy per token (mJ) -- lower is better")
    ax.set_ylabel("")
    title = "Energy per token by placement strategy"
    if not CXL_ENERGY_IS_MEASURED:
        title += "\n[CXL: DRAM measured (DDR4 proxy), link energy cited (Bichan et al., CICC 2020)]"
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "energy_metrics.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
