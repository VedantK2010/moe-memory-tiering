#!/usr/bin/env python3
"""
Regenerate every trace, result and figure in this project, from scratch.

    python run_all.py                # everything
    python run_all.py --skip-real    # skip the real-trace stage (it is slow)
    python run_all.py --list         # show the stages without running them

Every stage is deterministic (fixed seeds) and writes only into data/ and
results/. There are no absolute paths and no manual steps: a clean clone
plus the two real-trace CSVs in data/ reproduces every number in the report.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"

STAGES = [
    ("generate_trace",           "Generate synthetic routing traces + DRAMSim3 address traces"),
    ("capacity_sweep",           "Capacity vs latency; smart vs random vs LRU"),
    ("nonstationary_experiment", "Four strategies under a shifting workload"),
    ("robustness_check",         "Does the strategy ranking hold on a second dataset?"),
    ("real_benchmark",           "Real Mixtral traces: top-1 vs top-2, both layers"),
    ("calc_energy",              "Energy per token from measured pJ/bit"),
    ("batch_sensitivity",        "Where expert tiering stops paying (batch size)"),
    ("migration_sensitivity",    "Does LRU still win once migration costs something?"),
    ("scalability_sweep",        "Tiering headroom vs expert count and top_k"),
    ("pooling_study",            "CXL memory expansion and pooling"),
    ("prefetch_simulator",       "Markov prefetching across a dataset suite (negative result)"),
    ("build_dashboard",          "Write results/ into dashboard/index.html"),
    ("check_numbers",            "Check every number the README quotes against results/"),
]

REAL_STAGES = {"real_benchmark", "calc_energy", "batch_sensitivity",
               "migration_sensitivity", "prefetch_simulator"}


def run_stage(name, description):
    print("\n" + "=" * 72)
    print(f"  {name}  --  {description}")
    print("=" * 72)
    start = time.time()
    proc = subprocess.run([sys.executable, f"{name}.py"], cwd=SRC)
    elapsed = time.time() - start
    if proc.returncode != 0:
        print(f"\n!! {name} failed with exit code {proc.returncode}")
        return False, elapsed
    print(f"\n[{name} completed in {elapsed:.1f}s]")
    return True, elapsed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-real", action="store_true",
                    help="skip stages that need the large real Mixtral traces")
    ap.add_argument("--list", action="store_true", help="list stages and exit")
    args = ap.parse_args()

    if args.list:
        for name, desc in STAGES:
            print(f"  {name:<26} {desc}")
        return 0

    # The model reads latency and energy from the DRAMSim3 summaries in
    # results/; without them every stage would compute NaN silently.
    sys.path.insert(0, str(SRC))
    import config
    config.require_dramsim_results()

    stages = [s for s in STAGES if not (args.skip_real and s[0] in REAL_STAGES)]

    total = 0.0
    failed = []
    for name, desc in stages:
        ok, elapsed = run_stage(name, desc)
        total += elapsed
        if not ok:
            failed.append(name)

    print("\n" + "=" * 72)
    if failed:
        print(f"  {len(failed)} stage(s) FAILED: {', '.join(failed)}")
    else:
        print(f"  All {len(stages)} stages completed in {total:.1f}s")
    print("=" * 72)
    print("\nDRAMSim3 runs happen outside this pipeline (WSL, see README.md); their")
    print("stats are parsed into results/ with src/parse_dramsim3.py. Open")
    print("dashboard/index.html to see every result.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
