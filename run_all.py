"""
Reproduce every result in results/ from scratch.

    python run_all.py                  # everything that needs no downloads
    python run_all.py --stage traces   # just regenerate the input traces
    python run_all.py --list           # show the stages and what they make

Stages run in dependency order; each one writes into results/ (figures
and summary CSVs) or data/ (traces, which are gitignored because of
their size). Stages that need the real Mixtral traces are skipped with a
notice when data/real_expert_trace*.csv are absent -- see
src/convert_real_data.py for how to produce them.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
SRC = REPO / "src"

# (stage, script, what it produces, needs the real Mixtral traces?)
STAGES = [
    ("traces", "generate_trace.py",
     "expert_trace.csv, dramsim3_trace.txt, trace_summary.png", False),
    ("baseline", "tier_simulator.py",
     "tiering_sweep, smart_vs_random, three_strategy_comparison", False),
    ("burst", "generate_burst_trace.py",
     "dramsim3_burst_trace.txt (input for the DRAMSim3 run)", False),
    ("shift", "nonstationary_experiment.py",
     "nonstationary_trace.csv, nonstationary_comparison", False),
    ("hybrid", "hybrid_strategy.py",
     "hybrid_split_sweep, hybrid_nonstationary_comparison", False),
    ("reprofile", "periodic_reprofile.py",
     "reprofile_interval_sweep, four_strategy_comparison", False),
    ("robustness", "robustness_check.py",
     "robustness_check_summary.csv, dataset1/2 traces", False),
    ("tune-hybrid", "tune_hybrid_dataset2.py",
     "dataset2_hybrid_split_sweep, dataset2_hybrid_tuning.png", False),
    ("prefetch", "prefetch_simulator.py",
     "prefetch_multi_dataset.csv (7-dataset Markov study)", False),
    ("real-trace", "run_real_benchmark.py",
     "real_trace_hit_rates.csv", True),
    ("energy", "calc_energy.py",
     "energy_metrics.csv", False),
]

REAL_TRACES = ["real_expert_trace.csv", "real_expert_trace_layer31.csv"]


def have_real_traces():
    return all((REPO / "data" / f).exists() for f in REAL_TRACES)


def run(script):
    """Run a stage script with src/ on the path and the repo as cwd."""
    print(f"\n{'=' * 70}\n  {script}\n{'=' * 70}")
    t0 = time.perf_counter()
    proc = subprocess.run([sys.executable, str(SRC / script)], cwd=SRC)
    dt = time.perf_counter() - t0
    if proc.returncode != 0:
        print(f"\n  !! {script} exited {proc.returncode} after {dt:.1f}s")
        return False
    print(f"\n  -- {script} finished in {dt:.1f}s")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", action="append",
                    help="run only these stages (repeatable)")
    ap.add_argument("--list", action="store_true", help="list stages and exit")
    ap.add_argument("--keep-going", action="store_true",
                    help="continue after a stage fails")
    args = ap.parse_args()

    if args.list:
        print(f"{'stage':<14} {'script':<28} produces")
        for name, script, makes, needs_real in STAGES:
            tag = " [needs real traces]" if needs_real else ""
            print(f"{name:<14} {script:<28} {makes}{tag}")
        return 0

    wanted = set(args.stage) if args.stage else {s[0] for s in STAGES}
    unknown = wanted - {s[0] for s in STAGES}
    if unknown:
        print(f"Unknown stage(s): {', '.join(sorted(unknown))}")
        print(f"Known: {', '.join(s[0] for s in STAGES)}")
        return 2

    real_ok = have_real_traces()
    if not real_ok:
        print("NOTE: data/real_expert_trace*.csv not found -- stages that need the")
        print("      recorded Mixtral traces will be skipped. Everything else runs.")

    failed, skipped, ran = [], [], []
    for name, script, _makes, needs_real in STAGES:
        if name not in wanted:
            continue
        if needs_real and not real_ok:
            skipped.append(name)
            continue
        if run(script):
            ran.append(name)
        else:
            failed.append(name)
            if not args.keep_going:
                break

    print(f"\n{'=' * 70}")
    print(f"  ran     : {', '.join(ran) if ran else 'none'}")
    if skipped:
        print(f"  skipped : {', '.join(skipped)}  (need the real Mixtral traces)")
    if failed:
        print(f"  FAILED  : {', '.join(failed)}")
    print(f"{'=' * 70}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
