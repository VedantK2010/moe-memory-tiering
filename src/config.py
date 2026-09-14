"""
Central configuration for the CXL-based MoE memory tiering study.
=================================================================
Every path, model parameter and memory-tier constant used anywhere in this
project is defined here exactly once. No script should contain a hardcoded
absolute path or a magic number.

If you change a number here, re-run `python run_all.py` from the project
root and every result and figure regenerates consistently.
"""

import csv
from pathlib import Path

# ----------------------------------------------------------------------
# Paths (all relative to the project root -- no absolute paths anywhere)
# ----------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
DRAMSIM_DIR = PROJECT_ROOT / "dramsim3"

for _d in (DATA_DIR, RESULTS_DIR, DRAMSIM_DIR):
    _d.mkdir(exist_ok=True)

# Always read and write text as UTF-8. Windows defaults to cp1252, which is
# what corrupted the em-dashes and emoji in the original dashboard.
ENCODING = "utf-8"


# ----------------------------------------------------------------------
# Reference model: Mixtral 8x7B
# ----------------------------------------------------------------------
# Mixtral's MoE block has 8 experts per layer, routes each token to the top 2,
# and each expert is a 3-matrix SwiGLU FFN (w1, w2, w3).
#
#   params_per_expert = 3 * hidden * intermediate
#                     = 3 * 4096 * 14336
#                     = 176,160,768 params
#   bytes_per_expert  = 176,160,768 * 2 bytes (bf16)
#                     = 352,321,536 B = 336 MiB = 352.3 MB
#
# The original code used a 16 MiB placeholder, which understated expert size
# by ~21x and made the capacity argument look trivial. Use the real geometry.
NUM_EXPERTS = 8
TOP_K = 2
NUM_LAYERS = 32
HIDDEN_SIZE = 4096
FFN_INTERMEDIATE = 14336
BYTES_PER_PARAM = 2  # bf16 / fp16

PARAMS_PER_EXPERT = 3 * HIDDEN_SIZE * FFN_INTERMEDIATE          # 176,160,768
EXPERT_SIZE_BYTES = PARAMS_PER_EXPERT * BYTES_PER_PARAM         # 352,321,536

# Total expert weight footprint across the whole model. This is the number
# that motivates the entire study: ~90 GB of a ~93 GB model is expert weights,
# and only TOP_K/NUM_EXPERTS = 25% of it is touched per token.
EXPERT_BYTES_PER_LAYER = EXPERT_SIZE_BYTES * NUM_EXPERTS        # ~2.82 GB
EXPERT_BYTES_ALL_LAYERS = EXPERT_BYTES_PER_LAYER * NUM_LAYERS   # ~90.2 GB


# ----------------------------------------------------------------------
# DRAMSim3 results -- read from results/, never retyped
# ----------------------------------------------------------------------
# src/parse_dramsim3.py writes one whole-system summary row per DRAMSim3 run
# (results/dramsim3_<label>_summary.csv); the raw stats and the exact command
# for each run are in dramsim3/. Four runs feed this model:
#   hbm2_loaded / ddr4_cxl : streaming trace, back-to-back -> energy per bit
#   hbm2_idle   / ddr4_idle: same addresses, 1 read per 100 cycles -> latency
DRAMSIM_SUMMARIES = {
    "hbm": "hbm2_loaded", "cxl_dram": "ddr4_cxl",
    "hbm_idle": "hbm2_idle", "cxl_dram_idle": "ddr4_idle",
}


def _dramsim_field(label, field):
    """One value from the whole-system row parse_dramsim3.py wrote for run
    `label`. A missing file gives NaN rather than failing at import, because
    parse_dramsim3.py itself imports this module to create those files;
    run_all.py, calc_energy.py and build_dashboard.py refuse to run on it
    (see require_dramsim_results)."""
    path = RESULTS_DIR / f"dramsim3_{label}_summary.csv"
    if not path.exists():
        return float("nan")
    with open(path, encoding=ENCODING, newline="") as f:
        return float(next(csv.DictReader(f))[field])


def require_dramsim_results():
    """Fail loudly if any DRAMSim3 summary the model reads is missing."""
    missing = [f"results/dramsim3_{label}_summary.csv"
               for label in DRAMSIM_SUMMARIES.values()
               if not (RESULTS_DIR / f"dramsim3_{label}_summary.csv").exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing {', '.join(missing)}. Run DRAMSim3 and "
            f"src/parse_dramsim3.py first (see README.md).")


# ----------------------------------------------------------------------
# Memory tier parameters
# ----------------------------------------------------------------------
# Device latencies are MEASURED with DRAMSim3 on the idle trace: the average
# read latency (controller + DRAM, no queueing to speak of) of HBM2 and of
# DDR4-3200, which stands in for the DDR5 behind a CXL expander. They are
# DRAM-side latencies, not end-to-end load-to-use; the CXL link and
# controller traversal is added separately as a cited adder.
#
# At expert granularity none of this matters much: latency is < 0.02% of any
# 352 MB fetch (see access_time_ns). It is measured so that no number in the
# model is typed in or borrowed from an untraceable run.
HBM_LATENCY_NS = _dramsim_field(DRAMSIM_SUMMARIES["hbm_idle"], "avg_read_latency_ns")
HBM_BANDWIDTH_GBPS = 800  # ASSUMED: HBM3-class, per stack

# CXL adds a controller + PCIe PHY traversal on top of the DRAM latency.
# ~70 ns is the commonly reported round-trip adder (arXiv:2305.05033 and
# CXL memory-expansion vendor reports).
CXL_ADDED_LATENCY_NS = 70  # CITED
CXL_DEVICE_LATENCY_NS = _dramsim_field(DRAMSIM_SUMMARIES["cxl_dram_idle"], "avg_read_latency_ns")
CXL_LATENCY_NS = CXL_DEVICE_LATENCY_NS + CXL_ADDED_LATENCY_NS
# ASSUMED: one CXL link at PCIe Gen5 rates, x16 (64 GB/s per direction; an x8
# link would be 32 GB/s). x16 is the width of current CXL memory controllers.
CXL_BANDWIDTH_GBPS = 64


def access_time_ns(latency_ns, bandwidth_gbps, size_bytes):
    """Time for one memory access = latency (time to first byte) + transfer.

    Bandwidths are decimal GB/s (1e9 B/s), so sizes are converted with 1e9
    too. (An earlier version divided by 1024**3, understating every transfer
    time by 7.4%; ratios between tiers were unaffected.)

    IMPORTANT CAVEAT, state this in the report: at expert granularity the
    transfer term dominates completely. For a 352 MB expert the fixed
    latency is < 0.02% of total access time, so this study is effectively a
    BANDWIDTH study -- CXL's +70 ns adder is invisible at this granularity.
    That is a finding, not an oversight.
    """
    return latency_ns + size_bytes / bandwidth_gbps


# Pre-computed per-access times for a full expert fetch.
HBM_TIME_NS = access_time_ns(HBM_LATENCY_NS, HBM_BANDWIDTH_GBPS, EXPERT_SIZE_BYTES)
CXL_TIME_NS = access_time_ns(CXL_LATENCY_NS, CXL_BANDWIDTH_GBPS, EXPERT_SIZE_BYTES)

# Cost of installing an expert into HBM after a miss (the write half of a
# migration). Charged as an HBM-bandwidth write of the full expert.
# MIGRATION_COST_FACTOR scales it: 0.0 reproduces the original "migration is
# free" assumption, 1.0 charges the full write. Phase 2 sweeps this.
MIGRATION_WRITE_NS = access_time_ns(HBM_LATENCY_NS, HBM_BANDWIDTH_GBPS, EXPERT_SIZE_BYTES)
MIGRATION_COST_FACTOR = 0.0


# ----------------------------------------------------------------------
# Energy, read from our own DRAMSim3 runs -- never retyped
# ----------------------------------------------------------------------
# Both DRAM figures are READ from the one-row summaries that
# src/parse_dramsim3.py writes into results/, so every energy number in the
# project traces to a run in dramsim3/ (the exact command is in
# dramsim3/<label>_run.txt). Both runs use the same trace
# (data/dramsim3_loaded.txt: 256,000 back-to-back 64 B reads, each expert
# streamed through its own region) and end when the trace ends.
#
# Only ACCESS-PROPORTIONAL energy is used: (read_energy + act_energy) per
# DRAM read COMMAND. Two traps this avoids, both hit earlier in the project:
#   - total_energy includes standby/refresh power for the whole simulated
#     window; an early run left the simulator idling after the trace ended,
#     so 84.8% of its total_energy had nothing to do with the workload.
#   - DRAMSim3 merges a read into an identical queued read and counts both
#     as done; dividing energy by reads-done halved every pJ/bit when the
#     trace repeated addresses.
DRAMSIM_ACCESS_BYTES = 64

HBM_DYNAMIC_PJ_PER_ACCESS = _dramsim_field(DRAMSIM_SUMMARIES["hbm"], "dynamic_pj_per_access")
HBM_PJ_PER_BIT = _dramsim_field(DRAMSIM_SUMMARIES["hbm"], "dynamic_pj_per_bit")

# CXL-attached memory behind a PCIe Gen5 PHY. Two components:
#   - the DRAM itself: MEASURED. DRAMSim3 ships no DDR5 config, so DDR4-3200
#     (DDR4_8Gb_x8_3200.ini) stands in for the DDR5 a CXL expander would use.
#   - the CXL link (SerDes + PHY + controller): CITED, not measured. DRAMSim3
#     does not model a link. Confirm its source before quoting it.
# The CXL figure is therefore PARTLY measured; CXL_ENERGY_BASIS says so on
# every output that uses it.
CXL_DRAM_PJ_PER_BIT = _dramsim_field(DRAMSIM_SUMMARIES["cxl_dram"], "dynamic_pj_per_bit")
CXL_LINK_PJ_PER_BIT = 5.0       # CITED: PCIe Gen5 SerDes+PHY, not simulated
CXL_PJ_PER_BIT = CXL_DRAM_PJ_PER_BIT + CXL_LINK_PJ_PER_BIT
CXL_DRAM_ENERGY_IS_MEASURED = True
CXL_LINK_ENERGY_IS_MEASURED = False
CXL_ENERGY_IS_MEASURED = CXL_DRAM_ENERGY_IS_MEASURED and CXL_LINK_ENERGY_IS_MEASURED
CXL_ENERGY_BASIS = "DRAM measured (DRAMSim3, DDR4-3200 as DDR5 proxy); link cited"


def energy_pj(size_bytes, pj_per_bit):
    """Energy to move `size_bytes` at a given pJ/bit rate."""
    return size_bytes * 8 * pj_per_bit


HBM_EXPERT_FETCH_PJ = energy_pj(EXPERT_SIZE_BYTES, HBM_PJ_PER_BIT)
CXL_EXPERT_FETCH_PJ = energy_pj(EXPERT_SIZE_BYTES, CXL_PJ_PER_BIT)


# ----------------------------------------------------------------------
# Experiment defaults
# ----------------------------------------------------------------------
CAPACITY_K = 4          # HBM budget, in experts, for the headline experiments
SEED = 42
NUM_RANDOM_TRIALS = 20

# Synthetic trace generation
SYNTH_NUM_TOKENS = 50_000
SYNTH_SKEW_ALPHA = 6
SYNTH_P_REPEAT = 0.27
NONSTATIONARY_TOKENS_PER_PHASE = 20_000
NONSTATIONARY_NUM_PHASES = 3
NONSTATIONARY_SEED = 7

# Second, independent dataset for the robustness check
DATASET2_SEED = 99
DATASET2_SKEW_ALPHA = 3
DATASET2_P_REPEAT = 0.20

REPROFILE_INTERVALS = [100, 200, 500, 1000, 2000, 5000, 10000, 20000]

# Real Mixtral routing traces (allenai/analysis_mixtral)
REAL_TRACES = {
    "Layer 15": DATA_DIR / "real_expert_trace.csv",
    "Layer 31": DATA_DIR / "real_expert_trace_layer31.csv",
}
# Real-trace comparisons: the first REAL_WARMUP_TOKENS tokens are a profiling
# prefix (deployable static placement ranks experts on them) and are never
# scored. Every strategy is scored on the same remainder of the trace.
REAL_WARMUP_TOKENS = 20_000


def summary():
    """Print the derived constants. Useful as a sanity check and as a source
    of the numbers quoted in the report and dashboard."""
    lines = [
        "=" * 68,
        "DERIVED CONFIGURATION",
        "=" * 68,
        f"Expert size          : {EXPERT_SIZE_BYTES:,} B "
        f"({EXPERT_SIZE_BYTES / 1024**2:.1f} MiB / {EXPERT_SIZE_BYTES / 1e6:.1f} MB)",
        f"Experts per layer    : {NUM_EXPERTS}  ->  {EXPERT_BYTES_PER_LAYER / 1e9:.2f} GB/layer",
        f"All layers           : {NUM_LAYERS}  ->  {EXPERT_BYTES_ALL_LAYERS / 1e9:.1f} GB of expert weights",
        "",
        f"HBM access (1 expert): {HBM_TIME_NS:,.1f} ns "
        f"(latency {HBM_LATENCY_NS:.2f} ns [MEASURED, DRAMSim3 idle] = "
        f"{HBM_LATENCY_NS / HBM_TIME_NS * 100:.4f}% of total)",
        f"CXL access (1 expert): {CXL_TIME_NS:,.1f} ns "
        f"(latency {CXL_DEVICE_LATENCY_NS:.2f} ns DRAM [MEASURED, DDR4 idle] + "
        f"{CXL_ADDED_LATENCY_NS} ns link [CITED] = {CXL_LATENCY_NS / CXL_TIME_NS * 100:.4f}% of total)",
        f"CXL/HBM time ratio   : {CXL_TIME_NS / HBM_TIME_NS:.2f}x "
        f"(bandwidth ratio is {HBM_BANDWIDTH_GBPS / CXL_BANDWIDTH_GBPS:.2f}x -- "
        f"the model is bandwidth-dominated)",
        f"Migration write      : {MIGRATION_WRITE_NS:,.1f} ns (factor {MIGRATION_COST_FACTOR})",
        "",
        f"HBM dynamic energy   : {HBM_DYNAMIC_PJ_PER_ACCESS:.1f} pJ per {DRAMSIM_ACCESS_BYTES} B "
        f"= {HBM_PJ_PER_BIT:.3f} pJ/bit  [MEASURED, DRAMSim3 HBM2]",
        f"CXL energy           : {CXL_PJ_PER_BIT:.3f} pJ/bit = "
        f"DRAM {CXL_DRAM_PJ_PER_BIT:.3f} [MEASURED, DDR4 proxy] + "
        f"link {CXL_LINK_PJ_PER_BIT:.3f} [CITED]",
        f"HBM expert fetch     : {HBM_EXPERT_FETCH_PJ / 1e9:.2f} mJ",
        f"CXL expert fetch     : {CXL_EXPERT_FETCH_PJ / 1e9:.2f} mJ",
        "=" * 68,
    ]
    print("\n".join(lines))


if __name__ == "__main__":
    summary()
