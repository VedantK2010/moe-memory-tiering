"""
Dashboard Builder
=================
Writes every number the dashboard shows into dashboard/index.html, straight
from results/. The page is hand-authored except for one generated block
between the @@DATA-BEGIN@@ / @@DATA-END@@ markers; all figures, tables and
in-text numbers on the page are computed in the browser from that block.
Nothing on the page is typed by hand, so re-running the pipeline can never
leave the dashboard showing stale numbers.

It also writes results/model_parameters.csv: the model's input constants
from config.py (bandwidths, latencies, expert size, pJ/bit and the per-fetch
energies derived from them), each labelled with its basis -- measured, cited,
assumed or derived -- so those figures trace to a row in results/ too.

Run last in run_all.py, after every result exists.
"""

import json
import math

import pandas as pd

from config import (
    PROJECT_ROOT, RESULTS_DIR, ENCODING,
    NUM_EXPERTS, TOP_K, NUM_LAYERS, CAPACITY_K, NUM_RANDOM_TRIALS,
    EXPERT_SIZE_BYTES, EXPERT_BYTES_ALL_LAYERS,
    HBM_LATENCY_NS, HBM_BANDWIDTH_GBPS, CXL_DEVICE_LATENCY_NS, CXL_ADDED_LATENCY_NS,
    CXL_LATENCY_NS, CXL_BANDWIDTH_GBPS, HBM_TIME_NS, CXL_TIME_NS,
    HBM_PJ_PER_BIT, CXL_DRAM_PJ_PER_BIT, CXL_LINK_PJ_PER_BIT, CXL_PJ_PER_BIT,
    HBM_EXPERT_FETCH_PJ, CXL_EXPERT_FETCH_PJ, CXL_ENERGY_BASIS, CXL_LINK_SOURCE,
    REAL_WARMUP_TOKENS,
    require_dramsim_results,
)

DASHBOARD = PROJECT_ROOT / "dashboard" / "index.html"
BEGIN = "/* @@DATA-BEGIN@@ written by src/build_dashboard.py from results/ -- do not edit */"
END = "/* @@DATA-END@@ */"


def model_parameters():
    """Every model input the dashboard quotes, with where it comes from."""
    rows = [
        ("num_experts", NUM_EXPERTS, "experts/layer", "Mixtral 8x7B architecture"),
        ("top_k", TOP_K, "experts/token", "Mixtral 8x7B architecture"),
        ("num_layers", NUM_LAYERS, "layers", "Mixtral 8x7B architecture"),
        ("capacity_k", CAPACITY_K, "experts in HBM", "headline HBM budget (assumed)"),
        ("num_random_trials", NUM_RANDOM_TRIALS, "placements", "random baseline draws per budget"),
        ("expert_bytes", EXPERT_SIZE_BYTES, "B", "derived: 3 x 4096 x 14336 x 2 B"),
        ("expert_gb_all_layers", EXPERT_BYTES_ALL_LAYERS / 1e9, "GB", "derived"),
        ("hbm_latency_ns", HBM_LATENCY_NS, "ns", "measured: DRAMSim3 HBM2, idle trace"),
        ("hbm_bandwidth_gbps", HBM_BANDWIDTH_GBPS, "GB/s", "assumed: HBM3-class stack"),
        ("cxl_device_latency_ns", CXL_DEVICE_LATENCY_NS, "ns",
         "measured: DRAMSim3 DDR4-3200 (DDR5 proxy), idle trace"),
        ("cxl_added_latency_ns", CXL_ADDED_LATENCY_NS, "ns", "cited: CXL controller + PHY adder"),
        ("cxl_latency_ns", CXL_LATENCY_NS, "ns", "derived: device + adder"),
        ("cxl_bandwidth_gbps", CXL_BANDWIDTH_GBPS, "GB/s",
         "assumed: one CXL x16 link at PCIe Gen5 rates, per direction"),
        ("hbm_time_ns", HBM_TIME_NS, "ns/expert fetch", "derived: latency + size/bandwidth"),
        ("cxl_time_ns", CXL_TIME_NS, "ns/expert fetch", "derived: latency + size/bandwidth"),
        ("hbm_pj_per_bit", HBM_PJ_PER_BIT, "pJ/bit", "measured: DRAMSim3 HBM2"),
        ("cxl_dram_pj_per_bit", CXL_DRAM_PJ_PER_BIT, "pJ/bit",
         "measured: DRAMSim3 DDR4-3200 as DDR5 proxy"),
        ("cxl_link_pj_per_bit", CXL_LINK_PJ_PER_BIT, "pJ/bit", f"cited: {CXL_LINK_SOURCE}"),
        ("cxl_pj_per_bit", CXL_PJ_PER_BIT, "pJ/bit", "derived: DRAM + link"),
        ("hbm_fetch_mj", HBM_EXPERT_FETCH_PJ / 1e9, "mJ/expert fetch", "derived"),
        ("cxl_fetch_mj", CXL_EXPERT_FETCH_PJ / 1e9, "mJ/expert fetch", "derived"),
        ("real_warmup_tokens", REAL_WARMUP_TOKENS, "tokens", "profiling prefix, never scored"),
    ]
    return pd.DataFrame(rows, columns=["name", "value", "unit", "basis"])


def _clean(v):
    """JSON-safe scalar: NaN -> None, floats trimmed to 6 significant digits."""
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return None
        return float(f"{v:.6g}")
    if hasattr(v, "item"):          # numpy scalar
        return _clean(v.item())
    return v


def records(name, cols=None):
    df = pd.read_csv(RESULTS_DIR / name)
    if cols:
        df = df[cols]
    return [{k: _clean(v) for k, v in row.items()} for row in df.to_dict("records")]


def build_data():
    require_dramsim_results()
    params = model_parameters()
    params.to_csv(RESULTS_DIR / "model_parameters.csv", index=False, encoding=ENCODING)

    profile = pd.read_csv(RESULTS_DIR / "real_trace_profile.csv")
    residency = pd.read_csv(RESULTS_DIR / "real_token_residency.csv")
    prof = {}
    for layer, sub in profile.groupby("layer", sort=False):
        rep = residency.loc[residency["layer"] == layer, "repeat_rate"]
        prof[layer] = {
            "first": [_clean(v) for v in sub["first_choice_share_pct"]],
            "second": [_clean(v) for v in sub["second_choice_share_pct"]],
            "repeat_rate": _clean(float(rep.iloc[0])) if len(rep) else None,
        }

    return {
        "params": {r["name"]: _clean(r["value"]) for r in params.to_dict("records")},
        "energy_basis": CXL_ENERGY_BASIS,
        "link_source": CXL_LINK_SOURCE,
        "dramsim": records("dramsim3_hbm2_loaded_summary.csv")
                   + records("dramsim3_ddr4_cxl_summary.csv")
                   + records("dramsim3_hbm2_idle_summary.csv")
                   + records("dramsim3_ddr4_idle_summary.csv"),
        "capacity": records("capacity_sweep.csv"),
        "real": records("real_trace_results.csv"),
        "residency": records("real_token_residency.csv"),
        "profile": prof,
        "periodic_sweep": records("real_periodic_sweep.csv"),
        "phases": records("nonstationary_by_phase.csv"),
        "robustness": records("robustness_check.csv"),
        "migration": records("migration_sensitivity.csv"),
        "energy": records("energy_metrics.csv"),
        "energy_link": records("energy_link_sensitivity.csv"),
        "batch": records("batch_sensitivity.csv"),
        "scalability": records("scalability_sweep.csv"),
        "pooling": records("pooling_study.csv"),
        "prefetch": records("prefetch_multi_dataset.csv", [
            "dataset", "experts", "top_k", "markov_order", "gating",
            "lru_hit_rate_pct", "prefetch_hit_rate_pct", "hit_rate_gain_pts",
            "prefetch_accuracy_pct", "cxl_traffic_multiplier", "energy_multiplier"]),
    }


def inject(data):
    html = DASHBOARD.read_text(encoding=ENCODING)
    i, j = html.find(BEGIN), html.find(END)
    if i < 0 or j < i:
        raise SystemExit(f"Data markers not found in {DASHBOARD}; refusing to guess.")
    block = f"{BEGIN}\nconst DATA = {json.dumps(data, separators=(',', ':'))};\n"
    DASHBOARD.write_text(html[:i] + block + html[j:], encoding=ENCODING, newline="\n")


def main():
    data = build_data()
    inject(data)
    n = sum(len(v) for v in data.values() if isinstance(v, list))
    print(f"Wrote {n} result rows + {len(data['params'])} model parameters into "
          f"{DASHBOARD.relative_to(PROJECT_ROOT)}")
    print("Wrote results/model_parameters.csv")


if __name__ == "__main__":
    main()
