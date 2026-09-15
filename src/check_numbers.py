"""
Documentation Number Check
==========================
Every number README.md quotes must match results/. The dashboard cannot go
stale (build_dashboard.py injects its numbers), but the README is
hand-written, so this script checks it.

Each check is a sentence template from a document, with `#` where a number
is quoted and `…` for a short stretch of text to skip, plus the value(s)
that number must have, computed here from results/. A quoted number passes
if it equals the computed value rounded to the precision the document uses
("8.4%" is checked against the result rounded to one decimal). A check
fails if:
  - the quoted number differs from results/ (the document is stale), or
  - the claim no longer holds (e.g. a policy the text names as the winner
    is not the winner any more), or
  - the sentence is not found (the wording changed: update the template).

Numbers that no check covers are counted, and listed with --list-unchecked;
most are citations, section numbers or fixed inputs.

    python src/check_numbers.py                  # exit 1 if anything is stale
    python src/check_numbers.py --list-unchecked

Run as the last stage of run_all.py, so a pipeline run that changes a result
fails until the README is updated.
"""

import argparse
import re
import sys

import pandas as pd

from config import PROJECT_ROOT, RESULTS_DIR, ENCODING, CAPACITY_K, TOP_K

DOCS = {
    "README": PROJECT_ROOT / "README.md",
}
NUM = r"(-?\d[\d,]*(?:\.\d+)?)"
ANY_NUM = re.compile(r"(?<![\w.])\d(?:[\d,]*\d)?(?:\.\d+)?")


def normalise(text):
    """Markdown emphasis and code marks removed, all whitespace one space."""
    text = text.replace("**", "").replace("`", "").replace("*", "")
    return re.sub(r"\s+", " ", text)


def compile_template(template):
    parts = []
    for tok in re.split(r"(#|…)", normalise(template)):
        if tok == "#":
            parts.append(NUM)
        elif tok == "…":
            parts.append(r".{0,200}?")
        else:
            parts.append(re.escape(tok))
    return re.compile("".join(parts))


def quoted_matches(quoted, value):
    """True if `value` rounds to the number as the document wrote it."""
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    s = quoted.replace(",", "")
    decimals = len(s.split(".")[1]) if "." in s else 0
    return f"{float(value):.{decimals}f}" == s


# ----------------------------------------------------------------------
# Facts, computed from results/ only
# ----------------------------------------------------------------------
def facts():
    rd = lambda name: pd.read_csv(RESULTS_DIR / name)
    F = {}
    params = rd("model_parameters.csv")
    P = dict(zip(params["name"], params["value"]))
    F["P"] = P

    real = rd("real_trace_results.csv")

    def hit(layer, routing, prefix):
        r = real[(real.layer == layer) & (real.routing == routing)
                 & real.strategy.str.startswith(prefix)]
        return float(r.hit_rate_pct.iloc[0])

    def winner(layer, routing):
        r = real[(real.layer == layer) & (real.routing == routing)
                 & ~real.strategy.str.contains("oracle")]
        return r.loc[r.hit_rate_pct.idxmax(), "strategy"]

    deployable2 = real[(real.routing == "top-2") & ~real.strategy.str.contains("oracle")]
    F.update(hit=hit, winner=winner,
             deploy_min=deployable2.hit_rate_pct.min(), deploy_max=deployable2.hit_rate_pct.max())

    res = rd("real_token_residency.csv").set_index("layer")
    F["res"] = res
    F["tokens"] = int(res.tokens.iloc[0])

    per = rd("real_periodic_sweep.csv")
    per = per[per.routing == "top-2"]

    def best_interval(layer):
        q = per[per.layer == layer]
        return int(q.loc[q.hit_rate_pct.idxmax(), "interval"])

    def first_below_static(layer):
        """Smallest interval from which every longer interval scores below static."""
        q = per[per.layer == layer].sort_values("interval")
        below = q.hit_rate_pct < hit(layer, "top-2", "Static (profiled")
        for i in range(len(q)):
            if below.iloc[i:].all():
                return int(q.interval.iloc[i])
        return None

    F.update(best_interval=best_interval, first_below_static=first_below_static,
             shortest_interval=int(per.interval.min()))

    mig = rd("migration_sensitivity.csv")
    F["mig_adv"] = lambda layer, f: float(
        mig[(mig.layer == layer) & (mig.migration_factor == f)].lru_advantage_pct.iloc[0])
    F["breakeven_pct"] = lambda layer: float(
        mig[mig.layer == layer].breakeven_factor.iloc[0]) * 100

    en = rd("energy_metrics.csv")
    F["esave"] = lambda layer, prefix: float(
        en[(en.layer == layer) & en.strategy.str.startswith(prefix)].saving_vs_worst_pct.iloc[0])
    F["periodic_every"] = int(re.search(r"every ([\d,]+)",
                                        en[en.strategy.str.startswith("Periodic")].strategy.iloc[0])
                              .group(1).replace(",", ""))
    el = rd("energy_link_sensitivity.csv")
    F["link_min"], F["link_max"] = el.link_pj_per_bit.min(), el.link_pj_per_bit.max()

    def link_span(layer, prefix):
        v = el[(el.layer == layer) & el.strategy.str.startswith(prefix)].saving_vs_worst_pct
        return float(v.min()), float(v.max())

    F["link_span"] = link_span
    F["link_rank_stable"] = bool(
        (el.groupby(["layer", "strategy"]).energy_rank.nunique() == 1).all())

    batch = rd("batch_sensitivity.csv").set_index("batch_size")
    F["batch"] = batch
    over = batch[batch.pct_steps_union_fits_budget < 50]
    F["batch_over_budget"] = int(over.index.min()) if len(over) else None

    scal = rd("scalability_sweep.csv")
    half = scal[scal.budget_fraction == 0.5]
    fixed = half[half.regime.str.startswith("fixed")]
    scaled = half[half.regime.str.startswith("scaled")].set_index("num_experts")
    F.update(fixed_min=fixed.tokens_fully_resident_pct.min(),
             fixed_max=fixed.tokens_fully_resident_pct.max(),
             fixed_n_min=int(fixed.num_experts.min()), fixed_n_max=int(fixed.num_experts.max()),
             scaled=scaled)

    pool = rd("pooling_study.csv")
    F["pool32"] = pool[(pool.replicas == 32) & (pool.pool_links == 1)].iloc[0]

    pf = rd("prefetch_multi_dataset.csv")
    best = [pf.loc[pf[pf.dataset == d].hit_rate_gain_pts.idxmax()]
            for d in pf.dataset.unique() if "(real)" in d]
    F.update(pf_runs=len(pf), pf_datasets=pf.dataset.nunique(),
             pf_acc=pf.prefetch_accuracy_pct.mean(),
             pf_gain=[b.hit_rate_gain_pts for b in best],
             pf_traffic=[(b.cxl_traffic_multiplier - 1) * 100 for b in best],
             pf_energy=[(b.energy_multiplier - 1) * 100 for b in best])

    rob = rd("robustness_check.csv")
    periodic = rob[rob.strategy.str.lower().str.startswith("periodic")].iloc[0]
    F.update(rob_same_rank=bool((rob.dataset1_rank == rob.dataset2_rank).all()),
             rob_periodic_first=bool(periodic.dataset1_rank == 1 and periodic.dataset2_rank == 1),
             rob_i1=int(periodic.dataset1_periodic_interval),
             rob_i2=int(periodic.dataset2_periodic_interval))

    F["ds"] = {label: rd(f"dramsim3_{label}_summary.csv").iloc[0]
               for label in ("hbm2_loaded", "ddr4_cxl", "hbm2_idle", "ddr4_idle")}
    return F


# ----------------------------------------------------------------------
# The checks: (document, template, value per `#`, or a bool for a claim)
# ----------------------------------------------------------------------
def checks(F):
    P, hit, res, ds, batch = F["P"], F["hit"], F["res"], F["ds"], F["batch"]
    L15, L31 = "Layer 15", "Layer 31"
    gib = 1024**3 / 1e9                     # the old units bug, for the "was" figures
    lat_share = P["cxl_latency_ns"] / P["cxl_time_ns"] * 100
    s15, s31 = F["link_span"](L15, "LRU"), F["link_span"](L31, "Periodic")
    l15_first_top1 = F["winner"](L15, "top-1") == "LRU"
    l31_first_top1 = F["winner"](L31, "top-1").startswith("Periodic")
    best_is_shortest = (F["best_interval"](L15) == F["best_interval"](L31) == F["shortest_interval"])
    sc128, sc256 = F["scaled"].loc[128], F["scaled"].loc[256]
    p32 = F["pool32"]
    choices_m = F["tokens"] * 2 * TOP_K / 1e6          # two layers x top-k per token

    R = "README"
    C = [
        # --- README: introduction and concepts
        (R, "holds about # GB of expert weights and uses # of # experts",
         (P["expert_gb_all_layers"], P["top_k"], P["num_experts"])),
        (R, "# tokens at each of two layers, top-2, so # million expert choices",
         (F["tokens"], choices_m)),
        (R, "Mixtral: # experts per layer, top-#, # MB per expert",
         (P["num_experts"], P["top_k"], P["expert_bytes"] / 1e6)),

        # --- README: key findings
        (R, "A CXL expert fetch takes #× an HBM fetch", (P["cxl_time_ns"] / P["hbm_time_ns"],)),
        (R, "At # MB per expert, latency is under", (P["expert_bytes"] / 1e6,)),
        (R, "latency is under #% of a CXL fetch", (lat_share < 0.002,)),
        (R, "(# µs per HBM fetch, # ms per CXL fetch), so CXL's +# ns adder",
         (P["hbm_time_ns"] / 1e3, P["cxl_time_ns"] / 1e6, P["cxl_added_latency_ns"])),
        (R, "At layer 15 LRU hits #% of accesses but serves only #% of tokens",
         (hit(L15, "top-2", "LRU"), res.loc[L15, "lru_all_resident_pct"])),
        (R, "than LRU (#% vs #%)",
         (res.loc[L31, "static_all_resident_pct"], res.loc[L31, "lru_all_resident_pct"])),
        (R, "LRU wins at layer 15 (#%) and periodic re-profiling at layer 31 (#%)",
         (hit(L15, "top-1", "LRU") if l15_first_top1 else None,
          hit(L31, "top-1", "Periodic") if l31_first_top1 else None)),
        (R, "every deployable policy at both layers lands within #–#%",
         (F["deploy_min"], F["deploy_max"])),
        (R, "LRU is #% faster than static at layer 31", (F["mig_adv"](L31, 0.0),)),
        (R, "once an install costs #% of a full HBM write", (F["breakeven_pct"](L31),)),
        (R, "At layer 15 it is still #% ahead at a full write", (F["mig_adv"](L15, 1.0),)),
        (R, "the best interval is the shortest tested, # tokens",
         (F["shortest_interval"] if best_is_shortest else None,)),
        (R, "At #+ tokens (layer 15) or #+ tokens (layer 31) it scores below static",
         (F["first_below_static"](L15), F["first_below_static"](L31))),
        (R, "(HBM # pJ/bit; CXL DRAM # pJ/bit using DDR4-3200 as a DDR5 proxy, plus an # pJ/bit",
         (P["hbm_pj_per_bit"], P["cxl_dram_pj_per_bit"], P["cxl_link_pj_per_bit"])),
        (R, "LRU saves #% of memory energy at layer 15 and #% at layer 31",
         (F["esave"](L15, "LRU"), F["esave"](L31, "LRU"))),
        (R, "periodic re-profiling (every # tokens) saves #% and #%",
         (F["periodic_every"], F["esave"](L15, "Periodic"), F["esave"](L31, "Periodic"))),
        (R, "Sweeping the link energy from # to # pJ/bit moves LRU's layer-15 saving between #% and #%",
         (F["link_min"], F["link_max"], *s15)),
        (R, "and never changes the ranking", (F["link_rank_stable"],)),
        (R, "At batch # a decode step needs # experts; from batch # most steps need more than a #-expert budget",
         (1, batch.loc[1, "mean_union_size"], F["batch_over_budget"], CAPACITY_K)),
        (R, "by batch # a step touches # of the # experts on average",
         (32, batch.loc[32, "mean_union_size"], P["num_experts"])),
        (R, "Steps that never touch CXL fall from #% to #",
         (batch.loc[1, "pct_steps_fully_resident"], batch.pct_steps_fully_resident.iloc[-1])),
        (R, "and top-2 fixed, #–#% of tokens are served fully from HBM from # to # experts",
         (F["fixed_min"], F["fixed_max"], F["fixed_n_min"], F["fixed_n_max"])),
        (R, "it falls to #% at # experts (top-#) and #% at # experts (top-#",
         (sc128.tokens_fully_resident_pct, 128, sc128.top_k,
          sc256.tokens_fully_resident_pct, 256, sc256.top_k)),
        (R, "across # replicas saves # GB (#× consolidation), but on one link each replica's fetches are #× slower",
         (p32.replicas, p32.capacity_saved_gb, p32.consolidation_ratio, p32.pooled_slowdown_x)),
        (R, "Across # configurations on # datasets, mean prediction accuracy is #%",
         (F["pf_runs"], F["pf_datasets"], F["pf_acc"])),
        (R, "hides #–# more points of accesses for #–#% more CXL traffic and #–#% more energy",
         (min(F["pf_gain"]), max(F["pf_gain"]), min(F["pf_traffic"]), max(F["pf_traffic"]),
          min(F["pf_energy"]), max(F["pf_energy"]))),
        (R, "all four policies keep their rank", (F["rob_same_rank"],)),
        (R, "periodic re-profiling is first on both (best intervals # and # tokens)",
         (F["rob_i1"] if F["rob_periodic_first"] else None, F["rob_i2"])),

        # --- README: DRAMSim3 section and limitations
        (R, "Both hold the same # # B reads", (ds["hbm2_loaded"].read_cmds, 64)),
        (R, "HBM2_8Gb_x128.ini, -c # | HBM energy per bit | #/# channels, #% utilised (DRAMSim3's trace reader caps HBM2 at #%), # pJ/bit",
         (ds["hbm2_loaded"].cycles, ds["hbm2_loaded"].active_channels, ds["hbm2_loaded"].channels,
          ds["hbm2_loaded"].utilisation_pct, ds["hbm2_loaded"].frontend_ceiling_pct,
          ds["hbm2_loaded"].dynamic_pj_per_bit)),
        (R, "DDR4_8Gb_x8_3200.ini, -c # | CXL-side DRAM energy per bit | #% utilised, # pJ/bit",
         (ds["ddr4_cxl"].cycles, ds["ddr4_cxl"].utilisation_pct, ds["ddr4_cxl"].dynamic_pj_per_bit)),
        (R, "HBM2_8Gb_x128.ini, -c # | HBM read latency | # ns",
         (ds["hbm2_idle"].cycles, ds["hbm2_idle"].avg_read_latency_ns)),
        (R, "DDR4_8Gb_x8_3200.ini, -c # | CXL-side DRAM read latency | # ns (+ # ns cited CXL adder)",
         (ds["ddr4_idle"].cycles, ds["ddr4_idle"].avg_read_latency_ns, P["cxl_added_latency_ns"])),
        (R, "the idle HBM2 run gives # pJ/bit", (ds["hbm2_idle"].dynamic_pj_per_bit,)),
        (R, "the loaded run is lower (#) because streaming reads hit an open DRAM row more often (#% vs #%)",
         (ds["hbm2_loaded"].dynamic_pj_per_bit, ds["hbm2_loaded"].row_hit_rate_pct,
          ds["hbm2_idle"].row_hit_rate_pct)),
        (R, "Latencies (HBM # ns, CXL-side DRAM # ns) are idle DRAM read latencies",
         (P["hbm_latency_ns"], P["cxl_device_latency_ns"])),
        (R, "the CXL link adds a cited # ns", (P["cxl_added_latency_ns"],)),
        (R, "# GB/s per HBM3-class stack, # GB/s for one CXL x16 link",
         (P["hbm_bandwidth_gbps"], P["cxl_bandwidth_gbps"])),
        (R, "the # pJ/bit link figure is cited", (P["cxl_link_pj_per_bit"],)),
        (R, "every energy saving is re-computed from # to # pJ/bit", (F["link_min"], F["link_max"])),
        (R, "the CXL link energy (# pJ/bit)", (P["cxl_link_pj_per_bit"],)),

    ]
    return C


def main():
    ap = argparse.ArgumentParser(description="Check documented numbers against results/.")
    ap.add_argument("--list-unchecked", action="store_true",
                    help="list every number in the documents that no check covers")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding=ENCODING, errors="replace")   # µ, ×, – on any console

    texts = {k: normalise(p.read_text(encoding=ENCODING)) for k, p in DOCS.items()}
    covered = {k: set() for k in DOCS}
    failures, passed = [], 0

    for doc, template, expected in checks(facts()):
        pattern = compile_template(template)
        hits = list(pattern.finditer(texts[doc]))
        if not hits:
            failures.append(f"{doc}: sentence not found -- {template!r}")
            continue
        for m in hits:
            # One value per quoted number; any values beyond those are
            # true/false conditions the sentence asserts ("never changes").
            n = len(m.groups())
            values, conditions = expected[:n], expected[n:]
            if len(values) < n or not all(isinstance(c, bool) for c in conditions):
                failures.append(f"{doc}: template/value count mismatch -- {template!r}")
                continue
            bad = [] if all(conditions) else ["the sentence's claim no longer holds"]
            for i, value in enumerate(values):
                quoted = m.group(i + 1)
                covered[doc].add(m.start(i + 1))
                if isinstance(value, bool):
                    ok, shown = value, "claim no longer holds"
                else:
                    ok = quoted_matches(quoted, value)
                    shown = "claim no longer holds" if value is None else f"results/ gives {float(value):.6g}"
                if not ok:
                    bad.append(f"'{quoted}' ({shown})")
            if bad:
                failures.append(f"{doc}: {template!r}\n      " + "; ".join(bad))
            else:
                passed += 1

    unchecked = []
    for doc, text in texts.items():
        for m in ANY_NUM.finditer(text):
            if m.start() not in covered[doc]:
                unchecked.append((doc, m.group(), text[max(0, m.start() - 45):m.end() + 25]))

    n_checked = sum(len(v) for v in covered.values())
    print(f"Checked {n_checked} quoted numbers in {passed} sentences "
          f"({', '.join(p.name for p in DOCS.values())}) against results/.")
    if args.list_unchecked:
        print(f"\n{len(unchecked)} numbers not covered by a check "
              f"(citations, section numbers, fixed inputs, history):")
        for doc, num, ctx in unchecked:
            print(f"  {doc:<8} {num:>12}   ...{ctx}...")
    else:
        print(f"{len(unchecked)} other numbers (citations, section numbers, inputs) are not "
              f"checked; --list-unchecked shows them.")

    if failures:
        print(f"\n{len(failures)} FAILED -- these documents quote numbers results/ no longer supports:")
        for f in failures:
            print("  - " + f)
        return 1
    print("All documented numbers match results/.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
