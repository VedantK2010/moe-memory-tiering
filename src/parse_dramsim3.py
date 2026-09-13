"""
DRAMSim3 Output Parser
=======================
Turns a DRAMSim3 run into the handful of numbers this project actually uses,
and writes them to results/ so that nothing downstream is hand-typed.

WHY THIS EXISTS
---------------
The dashboard previously displayed DRAMSim3 statistics as hardcoded literals.
Those numbers could not be traced to a run, and one of them was wrong in a
way nobody could catch by reading the dashboard. Everything DRAMSim3 tells us
now flows through this parser.

THE ENERGY TRAP (read this before quoting any energy number)
------------------------------------------------------------
DRAMSim3's `total_energy` includes standby and refresh power for the ENTIRE
simulated window, whether or not the workload was still running. In our first
run the trace finished inside epoch 0 and the simulator idled for nine more
epochs, so 84.8% of `total_energy` was burned after the workload ended.
Dividing that by the request count gives a figure that scales with how long
you left the simulator running -- it is not an energy-per-access.

This parser therefore separates:
  - ACCESS-PROPORTIONAL energy (read_energy + act_energy). Scales with work.
    This is what we use, expressed as pJ/bit so it can be scaled to any
    transfer size.
  - TIME-PROPORTIONAL energy (refresh + precharge/active standby). Scales
    with wall-clock. Reported separately, never divided by request count.

It also reports the active-epoch utilisation, so you can see immediately
whether a run measured the loaded or the idle regime.

THE MERGED-READ TRAP
--------------------
DRAMSim3's controller merges a read into an identical read already waiting
in its queue: only one DRAM command is issued, but both count towards
`num_reads_done`. Energy is spent per COMMAND, so dividing it by reads-done
understates energy per access. A trace that re-read the same 64 addresses
merged ~50% of its reads and halved every pJ/bit it produced. Per-access
energy and bandwidth here are therefore computed from `num_read_cmds`, and
the merged share is reported so it cannot hide again.

THE TRACE-FRONTEND CEILING
--------------------------
DRAMSim3's trace reader hands the memory system at most one request per
cycle. On 8-channel HBM2 (1 ns, 64 B) that is 64 GB/s of a 256 GB/s peak, so
a trace-driven run can never exceed 25% average utilisation. Utilisation is
judged against that ceiling, not against the device peak.

USAGE
-----
    python src/parse_dramsim3.py <stats.json|stats.txt> [--epoch <epoch.json>]
                                 [--label hbm2_loaded] [--device HBM2]
                                 [--expected-reads 256000]
"""

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

from config import RESULTS_DIR, ENCODING, DRAMSIM_ACCESS_BYTES

# Peak bandwidth per channel, GB/s, for utilisation reporting.
# DDR4 is DDR4-3200 (DDR4_8Gb_x8_3200.ini): 3200 MT/s x 64-bit bus = 25.6 GB/s.
PEAK_BW_GBPS = {"HBM2": 32.0, "HBM3": 51.2, "DDR4": 25.6, "DDR5": 32.0}


def parse_txt(path):
    """Parse DRAMSim3's plain-text per-channel stats dump."""
    channels, current = [], None
    header = re.compile(r"##\s*Statistics of Channel\s+(\d+)")
    kv = re.compile(r"^(\w[\w\[\]\-.]*)\s*=\s*([-\d.e+]+)")
    for line in Path(path).read_text(encoding=ENCODING).splitlines():
        h = header.search(line)
        if h:
            current = {"channel": int(h.group(1))}
            channels.append(current)
            continue
        if current is None:
            continue
        m = kv.match(line.strip())
        if m:
            try:
                current[m.group(1)] = float(m.group(2))
            except ValueError:
                pass
    return channels


def parse_json(path):
    data = json.loads(Path(path).read_text(encoding=ENCODING))
    if isinstance(data, dict):
        return [v for _, v in sorted(data.items(), key=lambda kv: int(kv[0]))]
    return data


NAN = float("nan")


def _flatten(value):
    """DRAMSim3 reports some per-rank stats as {rank: value}. Energies add up
    across ranks, so this sums them."""
    if isinstance(value, dict):
        return sum(float(v) for v in value.values())
    return float(value or 0.0)


def _mean_over_ranks(value):
    """For per-rank cycle counts, where summing ranks would exceed 100%."""
    if isinstance(value, dict):
        vals = [float(v) for v in value.values()]
        return sum(vals) / len(vals) if vals else 0.0
    return float(value or 0.0)


def summarize(channels, access_bytes=DRAMSIM_ACCESS_BYTES, device="HBM2", tck_ns=None):
    peak = PEAK_BW_GBPS.get(device, NAN)
    rows = []
    for ch in channels:
        reads = float(ch.get("num_reads_done", 0))
        read_cmds = float(ch.get("num_read_cmds", 0))
        cycles = float(ch.get("num_cycles", 0)) or 1.0
        bw = float(ch.get("average_bandwidth", 0))

        # Infer tCK from the run itself. DRAMSim3's average_bandwidth is
        # reads_done * request bytes / (cycles * tCK), merged reads included.
        inferred_tck = None
        if bw > 0 and reads > 0:
            inferred_tck = (reads * access_bytes) / (bw * 1e9) / cycles * 1e9
        tck = tck_ns or inferred_tck or 1.0

        read_e = _flatten(ch.get("read_energy"))
        act_e = _flatten(ch.get("act_energy"))
        ref_e = _flatten(ch.get("ref_energy")) + _flatten(ch.get("refb_energy"))
        stb_e = _flatten(ch.get("pre_stb_energy")) + _flatten(ch.get("act_stb_energy"))
        total_e = _flatten(ch.get("total_energy"))

        dynamic_e = read_e + act_e
        # Bandwidth actually moved by DRAM: one request-sized burst per read
        # command. bytes / ns == GB/s.
        dram_bw = read_cmds * access_bytes / (cycles * tck)
        latency_cycles = float(ch.get("average_read_latency", 0)) if reads else NAN
        rows.append({
            "channel": int(ch.get("channel", len(rows))),
            "cycles": cycles,
            "reads": reads,
            "read_cmds": read_cmds,
            "merged_reads_pct": (reads - read_cmds) / reads * 100 if reads else NAN,
            "avg_read_latency_cycles": latency_cycles,
            "tck_ns": tck,
            "avg_read_latency_ns": latency_cycles * tck,
            "row_hit_rate_pct": (float(ch.get("num_read_row_hits", 0)) / read_cmds * 100
                                 if read_cmds else NAN),
            "avg_bandwidth_gbps": bw,  # as DRAMSim3 reports it: counts merged reads
            "dram_bandwidth_gbps": dram_bw,
            "peak_bandwidth_gbps": peak,
            "utilisation_pct": dram_bw / peak * 100,
            "idle_cycles_pct": _mean_over_ranks(ch.get("all_bank_idle_cycles")) / cycles * 100,
            "dynamic_energy_pj": dynamic_e,
            # Per read COMMAND -- see THE MERGED-READ TRAP above.
            "dynamic_pj_per_access": dynamic_e / read_cmds if read_cmds else NAN,
            "dynamic_pj_per_bit": dynamic_e / read_cmds / (access_bytes * 8) if read_cmds else NAN,
            "background_energy_pj": ref_e + stb_e,
            "background_share_pct": (ref_e + stb_e) / total_e * 100 if total_e else NAN,
            "total_energy_pj": total_e,
        })
    return pd.DataFrame(rows)


def system_summary(df, access_bytes=DRAMSIM_ACCESS_BYTES, label="", device=""):
    """One row of whole-memory-system numbers, so figures quoted from a
    multi-channel run trace to a row rather than to a hand-averaged value."""
    n = len(df)
    active = df[df["read_cmds"] > 0]
    reads, cmds = df["reads"].sum(), df["read_cmds"].sum()
    tck = df["tck_ns"].iloc[0]
    peak_total = df["peak_bandwidth_gbps"].iloc[0] * n
    # Trace frontend: at most one request per cycle (see module docstring).
    frontend_gbps = access_bytes / tck
    return pd.DataFrame([{
        "label": label,
        "device": device,
        "channels": n,
        "active_channels": len(active),
        "cycles": df["cycles"].iloc[0],
        "reads": reads,
        "read_cmds": cmds,
        "merged_reads_pct": (reads - cmds) / reads * 100 if reads else NAN,
        "tck_ns": tck,
        "avg_read_latency_ns": active["avg_read_latency_ns"].mean(),
        "row_hit_rate_pct": active["row_hit_rate_pct"].mean(),
        "dram_bandwidth_gbps": df["dram_bandwidth_gbps"].sum(),
        "peak_bandwidth_gbps": peak_total,
        "utilisation_pct": df["dram_bandwidth_gbps"].sum() / peak_total * 100,
        "frontend_ceiling_pct": min(100.0, frontend_gbps / peak_total * 100),
        "idle_cycles_pct": df["idle_cycles_pct"].mean(),
        "dynamic_energy_pj": df["dynamic_energy_pj"].sum(),
        "dynamic_pj_per_access": df["dynamic_energy_pj"].sum() / cmds if cmds else NAN,
        "dynamic_pj_per_bit": df["dynamic_energy_pj"].sum() / cmds / (access_bytes * 8) if cmds else NAN,
        "background_share_pct": df["background_energy_pj"].sum() / df["total_energy_pj"].sum() * 100,
    }])


def epoch_report(epoch_path, channel=0):
    """Per-epoch view for one channel. Shows whether the trace finished early
    and left the simulator idling -- the thing that corrupts total_energy."""
    data = json.loads(Path(epoch_path).read_text(encoding=ENCODING))
    entries = data if isinstance(data, list) else list(data.values())
    rows = [
        {
            "epoch": i,
            "reads": float(e.get("num_reads_done", 0)),
            "bandwidth_gbps": float(e.get("average_bandwidth", 0)),
            "total_energy_pj": _flatten(e.get("total_energy")),
        }
        for i, e in enumerate(e2 for e2 in entries if e2.get("channel") == channel)
    ]
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stats", help="DRAMSim3 stats file (.json or .txt)")
    ap.add_argument("--epoch", help="per-epoch stats json", default=None)
    ap.add_argument("--label", default="hbm2", help="name for the output files")
    ap.add_argument("--device", default="HBM2", choices=sorted(PEAK_BW_GBPS))
    ap.add_argument("--access-bytes", type=int, default=DRAMSIM_ACCESS_BYTES)
    ap.add_argument("--expected-reads", type=int, default=None,
                    help="trace length; warns if the run ended before the trace did")
    args = ap.parse_args()

    path = Path(args.stats)
    channels = parse_json(path) if path.suffix == ".json" else parse_txt(path)
    if not channels:
        sys.exit(f"No channel statistics found in {path}")

    df = summarize(channels, args.access_bytes, args.device)
    out = RESULTS_DIR / f"dramsim3_{args.label}.csv"
    df.to_csv(out, index=False, encoding=ENCODING)
    s = system_summary(df, args.access_bytes, args.label, args.device)
    out_sys = RESULTS_DIR / f"dramsim3_{args.label}_summary.csv"
    s.to_csv(out_sys, index=False, encoding=ENCODING)
    m = s.iloc[0]

    print(f"=== DRAMSim3 {args.device} :: {args.label} "
          f"({m['active_channels']} of {m['channels']} channels active) ===")
    print(f"  run length           : {m['cycles']:,.0f} cycles x {m['tck_ns']:.3f} ns (tCK inferred)")
    print(f"  reads done           : {m['reads']:,.0f} x {args.access_bytes} B")
    print(f"  DRAM read commands   : {m['read_cmds']:,.0f}  "
          f"({m['merged_reads_pct']:.2f}% of reads merged in the queue)")
    print(f"  avg read latency     : {m['avg_read_latency_ns']:.2f} ns (active channels; "
          f"includes queueing)")
    print(f"  row-buffer hit rate  : {m['row_hit_rate_pct']:.2f}% (active channels)")
    print(f"  DRAM bandwidth       : {m['dram_bandwidth_gbps']:.3f} GB/s of "
          f"{m['peak_bandwidth_gbps']:.1f} peak = {m['utilisation_pct']:.2f}% utilised")
    print(f"  frontend ceiling     : {m['frontend_ceiling_pct']:.2f}% "
          f"(1 request/cycle from the trace reader)")
    print(f"  all-bank idle cycles : {m['idle_cycles_pct']:.2f}% (mean over ranks)")
    print()
    print(f"  DYNAMIC energy (use this), per DRAM read command:")
    print(f"    {m['dynamic_pj_per_access']:.1f} pJ per {args.access_bytes} B access "
          f"= {m['dynamic_pj_per_bit']:.3f} pJ/bit")
    print(f"  BACKGROUND energy (do NOT divide by request count):")
    print(f"    {m['background_share_pct']:.1f}% of total_energy")

    if args.expected_reads is not None and m["reads"] < args.expected_reads:
        print(f"\n  !! WARNING: only {m['reads']:,.0f} of {args.expected_reads:,} trace reads")
        print( "     completed. The run ended before the trace did -- raise -c.")
    if m["merged_reads_pct"] > 1:
        print(f"\n  !! WARNING: {m['merged_reads_pct']:.1f}% of reads were merged with an")
        print( "     identical queued read and never reached DRAM. The trace repeats")
        print( "     addresses; bandwidth figures from DRAMSim3 itself are inflated.")
    if m["active_channels"] < m["channels"]:
        print(f"\n  !! WARNING: only {m['active_channels']} of {m['channels']} channels saw "
              f"traffic. Check the trace's address spread.")
    if m["utilisation_pct"] < 0.8 * m["frontend_ceiling_pct"]:
        print(f"\n  !! WARNING: {m['utilisation_pct']:.2f}% utilisation, well under the")
        print(f"     {m['frontend_ceiling_pct']:.2f}% the trace frontend allows. This run does")
        print( "     not measure the loaded regime our bandwidth-bound model assumes.")

    if args.epoch:
        ep = epoch_report(args.epoch)
        ep.to_csv(RESULTS_DIR / f"dramsim3_{args.label}_epochs.csv",
                  index=False, encoding=ENCODING)
        active = ep[ep["reads"] > 0]
        idle = ep[ep["reads"] == 0]
        print(f"\n  Per-epoch: {len(active)} active, {len(idle)} idle")
        if len(idle):
            wasted = idle["total_energy_pj"].sum() / ep["total_energy_pj"].sum() * 100
            print(f"  !! {wasted:.1f}% of total_energy was burned in epochs with ZERO reads.")
            print( "     Shorten the run to the trace length so total_energy means something.")

    print(f"\nWrote {out.name}, {out_sys.name}")


if __name__ == "__main__":
    main()
