# CXL-Based Memory Tiering for MoE Models

Mixture-of-Experts models keep eight FFN experts per layer resident in HBM but touch only two
per token. This project asks what happens if the cold ones are demoted to CXL-attached memory,
and which placement policy keeps the hot ones local. We evaluate four policies against
1.66 million recorded Mixtral 8x7B routing decisions and a cycle-accurate DRAM model.

**The headline finding: there is no single best policy — the right one depends on how deep in
the network you are.**

---

## Quick start

```bash
pip install -r requirements.txt
python run_all.py          # regenerates everything in results/ (~90 s)
```

Then open the dashboard, either way:

| | |
|---|---|
| **No install** | Open [`dashboard/index.html`](dashboard/index.html) in any browser. Self-contained, includes a live policy simulator. |
| **Streamlit** | `streamlit run dashboard/app.py` — same results, plus sliders that recompute the hardware model live. |

`run_all.py --list` shows the individual stages. Stages needing the recorded Mixtral traces are
skipped automatically when those files are absent (see [Data](#data)).

---

## Concepts

- **MoE (Mixture-of-Experts).** Models like Mixtral split each feed-forward layer into eight
  "experts". A router picks two per token; the other six sit idle but still occupy memory.
- **Memory tiering (HBM + CXL).** HBM is fast but small and expensive. Compute Express Link
  attaches a much larger pool of cheaper, slower memory over PCIe.
- **The question.** Put the active experts in HBM and idle ones in CXL and you free real
  capacity — but the active set moves. How well a policy tracks it decides whether tiering is
  cheap or ruinous.

---

## Key findings

**1. Middle layers reward reactivity.** At layer 15, context shifts fast and a frozen top-4
goes stale. Pure LRU reaches a **64.9% HBM hit rate**, over 7 points ahead of static placement.

**2. Deep layers reward stability.** At layer 31, routing concentrates on a stable handful of
experts. **Periodic re-profiling wins at 67.3%**, and LRU beats static by only 0.15 points —
inside the noise. The policy ordering *inverts* with depth.

**3. Re-profiling cadence has a real optimum.** Re-ranking every ~2,000 tokens beats all three
fixed alternatives. Stretched to 20,000 it is worse than doing nothing: static placement with
extra migration traffic.

**4. Energy tracks hit rate.** Charging CXL at ~3× local HBM per fetch, the best policy at
layer 15 cuts total memory energy by **8.0%**. The saving scales with the multiplier but never
reverses sign (see the sensitivity sweep in `calc_energy.py`).

**5. Two negative results we kept.** The hybrid static+LRU policy goes from 2nd best to worst
on an independent dataset — its 50/50 split was tuned, not principled. And Markov prefetching
never improved hit rate while reducing energy in **0 of 50** dataset/gating configurations: a
wrong prefetch still crosses the link. Both are reported rather than dropped.

---

## Repository tour

### Start here
- [`dashboard/index.html`](dashboard/index.html) — the interactive console. Live replay of all
  five policies on a shared trace, a hardware configurator, and every measured result.
- [`dashboard/app.py`](dashboard/app.py) — the Streamlit equivalent.

### Core simulation
- [`src/tier_simulator.py`](src/tier_simulator.py) — the timing model
  (`t = latency + size/bandwidth`), static placement, and the LRU cache. Also
  `lru_hit_rates_all_k`, which derives every capacity's hit rate from one stack-distance pass.
- [`src/periodic_reprofile.py`](src/periodic_reprofile.py) — the main experiment: static vs LRU
  vs periodic re-profiling, with no look-ahead for any policy.
- [`src/hybrid_strategy.py`](src/hybrid_strategy.py) — reserved HBM slots plus an LRU remainder.
- [`src/prefetch_simulator.py`](src/prefetch_simulator.py) — order-N Markov prefetching across
  seven datasets, charging for wasted prefetches.
- [`src/calc_energy.py`](src/calc_energy.py) — hit rates to memory energy, with a sensitivity
  sweep over the CXL multiplier.

### Traces and validation
- [`src/generate_trace.py`](src/generate_trace.py) — synthetic routing traces calibrated to
  Mixtral's reported expert skew and repeat rate.
- [`src/nonstationary_experiment.py`](src/nonstationary_experiment.py) — multi-phase traces
  whose hot experts rotate, for testing adaptation.
- [`src/generate_burst_trace.py`](src/generate_burst_trace.py) — burst-level DRAMSim3 input.
- [`src/convert_real_data.py`](src/convert_real_data.py) — pulls real routing decisions from
  `allenai/analysis_mixtral`.
- [`src/robustness_check.py`](src/robustness_check.py) — re-scores all four policies on a second
  independent dataset.

### Hardware validation
256,000 burst commands through DRAMSim3 on an HBM2 config: **60.71 ns** mean burst latency and
an **88.08%** row-buffer hit rate. The high row-buffer rate is what justifies treating a bulk
expert fetch as a near-streaming transfer in the analytic model.

---

## Data

The trace CSVs are gitignored — they are large and regenerable.

| File | How to get it |
|---|---|
| `data/expert_trace.csv`, `data/nonstationary_trace.csv`, `data/dataset{1,2}_trace.csv` | `python run_all.py` generates them |
| `data/real_expert_trace.csv`, `data/real_expert_trace_layer31.csv` | `python src/convert_real_data.py` (needs internet; runs well in Colab) |

Scripts that need a missing trace fail with the exact command to create it, rather than a
`FileNotFoundError`.

---

## Limitations

These are load-bearing. Each is a place to discount the result.

- **No migration cost.** A miss is charged the CXL read but not the cost of installing the
  expert into HBM or writing back the eviction. This flatters every adaptive policy — LRU most,
  since it migrates most — and is the single largest correction the model needs. Adding it could
  plausibly reverse the layer-15 result in favour of periodic re-profiling, whose appeal is
  bounded migration traffic. **That experiment is not yet run.**
- **Per-token re-fetch.** Every access is charged a full expert transfer, so absolute latencies
  (hundreds of µs) sit far above a real decode step, where a resident expert is read at tile
  granularity and never re-fetched. Ratios between policies are unaffected; absolute times are
  not comparable to measured inference.
- **Expert size.** 16 MiB is a placeholder for address-space layout. Mixtral's real per-expert
  FFN is 3 × 4096 × 14336 ≈ 176 M parameters — about 336 MiB at fp16, roughly 21× larger.
- **Top-1 only.** The real-trace analysis tracks the first-choice expert; Mixtral routes top-2,
  so true HBM pressure is higher and these hit rates are optimistic.
- **Energy constant.** 21.08 nJ/fetch is DRAMSim3 total energy over request count and includes
  amortised background and refresh power — it is not the marginal cost of a 64 B burst.
  Percentages are ratios built from the same constant, so they hold; absolute millijoules do not.
- **Memory parameters.** HBM 150 ns / 800 GB/s and CXL +70 ns / 64 GB/s are defensible published
  ballparks, not a specific part's datasheet numbers.
- **One model family.** Everything real is Mixtral 8x7B. Finer-grained MoE layers (64–128
  experts) were only tested as synthetic topologies in the prefetch study, not as real models.
- **Static profiling was unfair until recently.** `run_real_benchmark.py` originally profiled
  static placement on the same tokens it scored. It now profiles on a held-out prefix and prints
  the old oracle figure alongside so the bias is visible. Static hit rates published before that
  fix (57.51% / 65.76%) should be regenerated.

---

## References

- Jiang et al., *Mixtral of Experts*, 2024 — expert routing behaviour, locality by layer (Table 5).
- Fedus, Zoph & Shazeer, *Switch Transformers*, 2022 — expert load-balancing and selection skew.
- Li et al., *DRAMSim3: A Cycle-Accurate, Thermal-Capable DRAM Simulator*, IEEE CAL, 2020.
- *Compute Express Link Specification, Revision 2.0*, CXL Consortium.
- Sun et al., *Demystifying CXL Memory with Genuine CXL-Ready Systems*, MICRO 2023 — measured CXL latency overheads.
- Mattson et al., *Evaluation Techniques for Storage Hierarchies*, IBM Systems Journal, 1970 — the stack-distance result used in `lru_hit_rates_all_k`.
- `allenai/analysis_mixtral` (HuggingFace) — the recorded routing traces.
