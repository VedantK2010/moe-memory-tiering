# CXL-Based Memory Tiering for MoE Models

Mixture-of-Experts models keep every expert's weights in memory but route each token to only a
few of them. Mixtral 8x7B holds about 90 GB of expert weights and uses 2 of 8 experts per layer
per token. This project asks what happens if the cold experts are demoted from HBM to
CXL-attached memory, and which placement policy keeps the hot ones local.

We replay recorded Mixtral routing — 829,440 tokens at each of two layers, top-2, so 3.3 million
expert choices — through an analytic HBM/CXL tier model, and measure both tiers' DRAM energy per
bit with DRAMSim3.

---

## Quick start

```bash
pip install -r requirements.txt
python src/convert_real_data.py   # once: downloads the real Mixtral traces into data/
python run_all.py                 # regenerates every CSV, figure and the dashboard (~2.5 min)
```

Then open **[`dashboard/index.html`](dashboard/index.html)** in any browser. It is a single page
with a live policy replay, a hardware configurator, and every result below.

`python run_all.py --list` shows the stages; `--skip-real` skips the ones that need the real
traces.

---

## Concepts

- **MoE (Mixture-of-Experts).** Each feed-forward layer is split into experts; a router picks
  the top-k per token. Mixtral: 8 experts per layer, top-2, 352 MB per expert.
- **Memory tiering.** HBM is fast but small and expensive. CXL attaches a larger pool of cheaper,
  slower memory over a PCIe-class link.
- **The question.** Keep the active experts in HBM and demote the rest to CXL, and you free real
  capacity — but the active set moves. How well a policy tracks it decides the cost.

---

## Key findings

Numbers are from `results/` as of 14 Sept 2026; the dashboard always shows the current values.

1. **This is a bandwidth problem, not a latency one.** A CXL expert fetch takes 12.5× an HBM
   fetch — exactly the bandwidth ratio. At 352 MB per expert, latency is under 0.002% of a CXL
   fetch (440 µs per HBM fetch, 5.51 ms per CXL fetch), so CXL's +70 ns adder is invisible.
   (`model_parameters.csv`)
2. **Per-access hit rate overstates what a token sees.** Under top-2, a token avoids CXL only if
   *both* experts are resident. At layer 15 LRU hits 59.5% of accesses but serves only 37.1% of
   tokens entirely from HBM. At layer 31, static placement keeps *more* tokens fully resident
   than LRU (43.6% vs 39.8%) despite a lower per-access hit rate. (`real_token_residency.csv`)
3. **The "depth changes the winner" story was a top-1 artifact.** Reading only the first-choice
   expert, LRU wins at layer 15 (64.9%) and periodic re-profiling at layer 31 (67.4%). On the
   real top-2 workload, every deployable policy at both layers lands within 55.0–59.5%.
   (`real_trace_results.csv`)
4. **Migration cost erases LRU's edge at layer 31.** LRU is 0.6% faster than static at layer 31
   when installing an expert is free, and loses that lead once an install costs 8.8% of a full
   HBM write. At layer 15 it is still 1.9% ahead at a full write. (`migration_sensitivity.csv`)
5. **Periodic re-profiling's best cadence undercuts its purpose.** On the real traces the best
   interval is the shortest tested, 100 tokens. At 2,000+ tokens (layer 15) or 1,000+ tokens
   (layer 31) it scores below static placement. (`real_periodic_sweep.csv`)
6. **Energy follows the hit rate.** With both tiers' DRAM measured (HBM 1.632 pJ/bit; CXL DRAM
   9.080 pJ/bit using DDR4-3200 as a DDR5 proxy, plus a cited 5.0 pJ/bit link), LRU saves 7.8% of
   memory energy at layer 15 and 0.6% at layer 31; periodic re-profiling (every 100 tokens) saves
   6.0% and 4.5%. Energy is a linear function of hit rate, so this is a cost translation, not
   independent evidence. (`energy_metrics.csv`)
7. **Tiering is a small-batch technique.** At batch 1 a decode step needs 2 experts; from batch 4
   most steps need more than a 4-expert budget holds, and by batch 32 every expert is touched every
   step. Steps that never touch CXL fall from 32.5% to 0. (`batch_sensitivity.csv`)
8. **More, smaller experts make tiering harder, not easier.** With HBM holding half the experts
   and top-2 fixed, 35–44% of tokens are served fully from HBM from 8 to 256 experts. Let top-k
   grow with N and it falls to 18.6% at 128 experts (top-4) and 3.3% at 256 experts (top-8,
   DeepSeek-V3's topology). top-k dominates, not N. (`scalability_sweep.csv`)
9. **Pooling trades capacity for bandwidth.** Sharing one CXL copy of the cold experts across 32
   replicas saves 1,398 GB (1.94× consolidation), but on one link each replica's fetches are 28.7×
   slower. Links must scale with replicas. (`pooling_study.csv`)
10. **Prefetching is a negative result.** Across 70 configurations on 7 datasets, mean prediction
    accuracy is 29.1%. On the real traces the best setting hides 3.1–4.9 more points of accesses
    for 14–15% more CXL traffic and energy. Prefetching can only hide latency; it never saves
    bandwidth or energy. (`prefetch_multi_dataset.csv`)
11. **The policy ranking is robust.** On an independent synthetic dataset with sharper skew, all
    four policies keep their rank; periodic re-profiling is first on both (best intervals 2,000
    and 500 tokens). (`robustness_check.csv`)

---

## Repository tour

```
run_all.py                      regenerates everything, dashboard last
dashboard/index.html            the interactive dashboard (numbers injected from results/)
src/config.py                   every path and constant; reads measured pJ/bit from results/
src/tier_simulator.py           the engine: static, LRU, periodic, hybrid, migration cost
src/generate_trace.py           synthetic routing traces + DRAMSim3 address traces
src/convert_real_data.py        downloads the real Mixtral traces into data/
src/capacity_sweep.py           capacity vs time; ranked vs random vs LRU
src/nonstationary_experiment.py four policies under workload shift; hybrid split sweep
src/robustness_check.py         the same comparison on a second, independent dataset
src/real_benchmark.py           real traces: top-1 vs top-2, token-level residency, interval sweep
src/calc_energy.py              energy per token from measured pJ/bit
src/batch_sensitivity.py        where tiering stops paying
src/migration_sensitivity.py    does LRU survive migration cost?
src/scalability_sweep.py        expert count and top-k
src/pooling_study.py            CXL expansion and pooling
src/prefetch_simulator.py       Markov prefetching across a dataset suite
src/parse_dramsim3.py           DRAMSim3 stats -> results/
src/build_dashboard.py          results/ -> dashboard/index.html, model_parameters.csv
results/                        every CSV and figure — the single source for all numbers
dramsim3/                       DRAMSim3 stats and the exact command for each run
data/                           traces (gitignored; regenerable)
```

**No number is typed by hand anywhere downstream of `results/`.** `config.py` reads the measured
energy from the DRAMSim3 summaries, and `build_dashboard.py` injects every dashboard figure —
including the numbers in its prose — from the result files.

---

## DRAMSim3

DRAMSim3 runs outside the pipeline (WSL2), on two traces `generate_trace.py` writes. Both hold
the same 256,000 64 B reads, each expert fetch continuing through that expert's address range;
`dramsim3_loaded.txt` issues them back-to-back, `dramsim3_unloaded.txt` one every 100 cycles.
Four runs, each ending when its trace does:

| Run | Config | Gives the model | Result |
|---|---|---|---|
| `hbm2_loaded` | `HBM2_8Gb_x128.ini`, `-c 271000` | HBM energy per bit | 8/8 channels, 23.6% utilised (DRAMSim3's trace reader caps HBM2 at 25%), **1.632 pJ/bit** |
| `ddr4_cxl` | `DDR4_8Gb_x8_3200.ini`, `-c 1250000` | CXL-side DRAM energy per bit | 81.3% utilised, **9.080 pJ/bit** |
| `hbm2_idle` | `HBM2_8Gb_x128.ini`, `-c 25600000` | HBM read latency | **32.66 ns** |
| `ddr4_idle` | `DDR4_8Gb_x8_3200.ini`, `-c 25600000` | CXL-side DRAM read latency | **28.33 ns** (+ 70 ns cited CXL adder) |

Energy is read + activate energy per DRAM command. DDR4-3200 stands in for DDR5, which DRAMSim3
does not ship. Stats and exact commands are in `dramsim3/` (kept byte-for-byte — re-running
DRAMSim3 reproduces them exactly); `parse_dramsim3.py` turns them into
`results/dramsim3_*_summary.csv`, which `config.py` reads. As a cross-check, the idle HBM2 run
gives 1.764 pJ/bit, matching the 1.756 of the project's original idle run; the loaded run is
lower (1.632) because streaming reads hit an open DRAM row more often (96% vs 88%).

---

## Data

| File | How to get it |
|---|---|
| `data/expert_trace.csv`, `nonstationary_trace.csv`, `dataset2_trace.csv`, `dramsim3_*.txt` | generated by `python run_all.py` |
| `data/real_expert_trace.csv` (layer 15), `data/real_expert_trace_layer31.csv` | `python src/convert_real_data.py` (downloads ~24 MB once) |

---

## Limitations

- **Analytic, not cycle-accurate.** Latency comes from a two-tier model (`t = L + S/B`);
  DRAMSim3 supplies DRAM energy per bit only.
- **Two-valued model.** Every access costs one HBM or one CXL fetch, so time and energy are
  linear functions of hit rate and every latency chart is a hit-rate chart rescaled.
- **Per-access re-fetch.** Each access is charged a full expert transfer, so absolute times are
  expert-fetch times, not decode latencies. Comparisons between policies survive.
- **Latencies** (HBM 32.66 ns, CXL-side DRAM 28.33 ns) are idle DRAM read latencies measured
  with DRAMSim3, not end-to-end load-to-use; the CXL link adds a cited 70 ns.
- **Bandwidths** are assumed: 800 GB/s per HBM3-class stack, 64 GB/s for one CXL x16 link at
  PCIe Gen5 rates (per direction).
- **Energy.** CXL DRAM is measured with DDR4-3200 standing in for DDR5 (DRAMSim3 has no DDR5
  config); the 5.0 pJ/bit link figure is cited, not simulated. The DRAMSim3 trace is a scaled
  sample (8 × 64 B reads per expert fetch).
- **Batch-1 decode** everywhere except the batch-size study.
- **Real traces** cover two layers of one model family; expert popularity is domain-dependent.
- **Synthetic traces** are used where a controlled change is needed (shift, robustness,
  scalability, capacity sweep).
- **Prefetching** reports an upper bound on hidden latency.

---

## References

- Jiang et al., *Mixtral of Experts*, arXiv:2401.04088, 2024.
- Fedus, Zoph & Shazeer, *Switch Transformers*, JMLR 23(120), 2022.
- Li, Yang, Reddy, Srivastava & Jacob, *DRAMsim3: A Cycle-Accurate, Thermal-Capable DRAM
  Simulator*, IEEE Computer Architecture Letters 19(2), 2020 — used here for DRAM energy per bit.
- CXL Consortium, *Compute Express Link Specification, Revision 2.0*, 2020.
- Sun et al., *Demystifying CXL Memory with Genuine CXL-Ready Systems and Devices*, MICRO-56,
  2023 — measured CXL latency overheads.
- Mattson, Gecsei, Slutz & Traiger, *Evaluation Techniques for Storage Hierarchies*, IBM Systems
  Journal 9(2), 1970.
- Allen Institute for AI, `allenai/analysis_mixtral` (Hugging Face) — the recorded routing traces.

---

Vedant Kabra · Neil Verma
