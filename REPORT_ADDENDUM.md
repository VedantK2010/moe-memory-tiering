# Report addendum

For the report's authors. Two parts:

- **Part A** — paste-ready Limitations and References sections, rewritten for the corrected
  model (14 Sept 2026).
- **Part B** — what changed since the draft: claims the code no longer supports, and the numbers
  that must be updated.

**Rule for every number in the report:** take it from `results/` (or the dashboard, which reads
`results/`). The source file is named next to each figure below. Do not copy numbers from older
drafts, the old README, or this file if `results/` disagrees — `results/` wins.

---

# Part A — sections to paste into the report

## Section 11. Limitations and Threats to Validity

Every result in this report rests on a model, and every model leaves something out. The
omissions below are load-bearing: each is a reason to discount a specific claim.

### 11.1 The timing model is analytic and two-valued

Access time comes from `t = L + S/B` for a full expert transfer from HBM or from CXL, not from a
cycle-accurate simulation. DRAMSim3 supplies DRAM energy per bit only. Because every access costs
exactly one HBM fetch or one CXL fetch, average time — and energy — are exact linear functions of
hit rate. Every latency chart is the hit-rate chart rescaled, and the energy results restate the
hit-rate results in joules rather than corroborating them.

### 11.2 At expert granularity this is a bandwidth study

At 352 MB per expert the transfer term dominates: latency is under 0.002% of a CXL fetch, so the
CXL/HBM time ratio (12.5×) equals the bandwidth ratio. CXL's latency adder is invisible at this
granularity. That is a finding, but it also means our conclusions say nothing about latency-bound
access patterns. (`results/model_parameters.csv`)

### 11.3 Absolute times are fetch times, not inference latencies

The model charges a full expert transfer on every access. A real decoder reads a resident
expert's weights at tile granularity and does not re-fetch it per token, so absolute times should
not be quoted as predicted inference latencies. Comparisons between policies, charged under
identical rules, survive.

### 11.4 Energy is partly measured

DRAM energy per bit is measured with DRAMSim3 (read + activate energy per DRAM command) on HBM2
and on DDR4-3200, which stands in for the DDR5 behind a CXL expander because DRAMSim3 has no DDR5
configuration. The CXL link energy (SerDes + PHY) is a cited figure, not simulated: 11.4 pJ/bit,
from the first published PCIe Gen5 SerDes including PLL and clocking [9]. Short-reach transceivers
are cheaper and a CXL controller adds cost that figure does not count, so every energy saving is
re-computed with the link energy swept from 0 to 14 pJ/bit: LRU's layer-15 saving moves between
6.8% and 8.6%, and the ranking of policies never changes. (`results/energy_link_sensitivity.csv`)
The DRAMSim3 input is a scaled sample of the access pattern — eight 64 B reads per expert fetch,
each fetch continuing through the expert — not full 352 MB transfers. HBM2 utilisation is capped at 25% by
DRAMSim3's trace reader (one request per cycle); our run reached 23.6%. The two device
latencies (HBM2 32.66 ns, DDR4 28.33 ns) are DRAMSim3 idle read latencies — DRAM-side, not
end-to-end load-to-use — and the CXL link's +70 ns is cited. Bandwidths (800 GB/s per HBM stack,
64 GB/s for one CXL x16 Gen5 link) are assumptions. (`results/dramsim3_*_summary.csv`,
`results/model_parameters.csv`, `dramsim3/`)

### 11.5 Migration is charged only in the sensitivity study

The headline comparisons treat installing an expert into HBM as free, which flatters LRU (it
migrates on every miss). The migration study charges a fraction of a full HBM write per install:
LRU's advantage at layer 31 disappears at 8.8% of a full write, while at layer 15 LRU remains ahead
even at a full write. Real systems overlap part of an install with compute, so the true factor
lies between these bounds. (`results/migration_sensitivity.csv`)

### 11.6 Batch-1 decode only

Except for the batch-size study, every result assumes batch-1 decode. The batch study shows why
this matters: by batch 32 a step touches 7.99 of the 8 experts on average, and no placement
avoids CXL.
(`results/batch_sensitivity.csv`)

### 11.7 Real traces cover one model family and two layers

Real measurements come from two layers (15 and 31) of `allenai/analysis_mixtral`, with full top-2
routing. Expert popularity is domain-dependent, and this study's thesis is about workload shift.
Fine-grained topologies (64–256 experts) were tested only with synthetic traces that vary expert
count and top-k; they are not measurements of those models.

### 11.8 Evaluation protocol for the real traces

The first 20,000 tokens of each real trace are a profiling prefix and are never scored. Deployable
static placement ranks experts on that prefix; every policy is scored on the same remainder. A
static "oracle" that ranks on the scored window itself is reported alongside as a ceiling, not as a
policy.

### 11.9 Prefetching reports an upper bound

The Markov prefetcher predicts at token granularity (the router picks both of a token's experts
at once, so the only prefetch window is between tokens) and charges every issued prefetch in CXL
traffic and energy. A correct prefetch is assumed to hide the whole fetch, so the hit-rate gain is
an upper bound on latency hidden.

### 11.10 What would change our conclusions

| Correction | Expected effect |
|---|---|
| Model tile-granularity reads instead of full re-fetch | Lowers absolute times; ranking of policies likely preserved |
| Measure the CXL link energy, or use a real DDR5 config | Changes the CXL/HBM energy ratio and so every energy saving (a 0–14 pJ/bit link moves LRU's layer-15 saving between 6.8% and 8.6%); rankings and hit-rate results unaffected |
| Charge migration in every comparison | Penalises LRU; at layer 31 favours static or periodic placement |
| Larger batches | Removes most of the benefit of tiering (see 11.6) |
| Real 64–256-expert models | Unknown; synthetic results suggest top-k, not expert count, dominates |

---

## Section 12. References

1. Jiang, A. Q., Sablayrolles, A., Roux, A., et al. *Mixtral of Experts.* arXiv:2401.04088, 2024.
   — Expert routing behaviour, temporal locality by layer (Table 5), and the architecture our
   traces are drawn from.
2. Fedus, W., Zoph, B., and Shazeer, N. *Switch Transformers: Scaling to Trillion Parameter
   Models with Simple and Efficient Sparsity.* JMLR 23(120), 2022. — Expert load-balancing and
   the selection skew our synthetic traces are calibrated against.
3. Li, S., Yang, Z., Reddy, D., Srivastava, A., and Jacob, B. *DRAMsim3: A Cycle-Accurate,
   Thermal-Capable DRAM Simulator.* IEEE Computer Architecture Letters 19(2), 2020. — Used to
   measure DRAM energy per bit for the HBM2 and DDR4 tiers.
4. CXL Consortium. *Compute Express Link Specification, Revision 2.0.* 2020. — Memory pooling
   and type-3 device semantics underlying the expansion and pooling study.
5. Sun, Y., Yuan, Y., Yu, Z., et al. *Demystifying CXL Memory with Genuine CXL-Ready Systems and
   Devices.* MICRO-56, 2023. — Measured CXL latency overheads; the source of the ~70 ns adder.
6. Mattson, R. L., Gecsei, J., Slutz, D. R., and Traiger, I. L. *Evaluation Techniques for
   Storage Hierarchies.* IBM Systems Journal 9(2), 1970. — The stack property of LRU.
7. Belady, L. A. *A Study of Replacement Algorithms for a Virtual-Storage Computer.* IBM Systems
   Journal 5(2), 1966. — The optimal offline policy, the natural upper bound for future work.
8. Allen Institute for AI. *allenai/analysis_mixtral.* Hugging Face Datasets. — The recorded
   per-token, per-layer expert routing used for all real-trace results.
9. Bichan, M., Ting, C., Zand, B., et al. *A 32Gb/s NRZ 37dB SerDes in 10nm CMOS to Support PCI
   Express Gen 5 Protocol.* IEEE Custom Integrated Circuits Conference (CICC), 2020,
   doi:10.1109/CICC48029.2020.9075947. — The first PCIe Gen5 SerDes: 11.4 pJ/bit including PLL
   and clocking over a 37 dB channel; the CXL link energy in our model.

---

# Part B — what changed since the draft

### B1. Claims the code no longer supports

| Draft claim | Status | Replace with |
|---|---|---|
| "cycle-accurate DRAMSim3 simulations" | **False.** Latency is analytic. | "DRAMSim3-measured DRAM energy per bit; analytic timing model" (§11.1) |
| "Middle layers reward reactivity, deep layers reward stability" / "depth changes the winner" | **Top-1 artifact.** Under real top-2 routing every deployable policy lands within 55.0–59.5% at both layers. | The top-1 vs top-2 comparison and token-level residency (`real_trace_results.csv`, `real_token_residency.csv`) |
| "~8% energy savings" from a 21.08 nJ/fetch constant | **Wrong basis.** That constant was mostly standby power. | LRU saves 8.4% (layer 15) and 0.6% (layer 31) with measured DRAM energy and a cited link (`energy_metrics.csv`) |
| CXL link energy 5.0 pJ/bit (earlier drafts and code) | **No source was ever recorded.** | 11.4 pJ/bit cited from Bichan et al., CICC 2020 [9], with every saving swept from 0 to 14 pJ/bit (`energy_link_sensitivity.csv`) |
| "Re-profiling every ~2,000 tokens is optimal" | Holds on the synthetic phased trace only. On real traces the best interval is 100 tokens, and 2,000 tokens scores below static at layer 15. | Quote each periodic number with its interval and trace (`real_periodic_sweep.csv`, `nonstationary_by_phase.csv`) |
| "Hybrid falls from 2nd to 4th on dataset 2" | **No longer true.** All four policies keep their rank on both datasets. | `robustness_check.csv` |
| Expert size 16 MiB; HBM 150 ns | **Superseded.** Real Mixtral expert: 352 MB. HBM latency 32.66 ns and CXL-side DRAM 28.33 ns, both measured with DRAMSim3. | `model_parameters.csv` |
| Absolute fetch times from earlier runs (e.g. 410 µs per HBM fetch) | **7.4% too low** — sizes were converted with 1024³ against decimal GB/s. Ratios, hit rates and energy were unaffected. | 440 µs HBM, 5.51 ms CXL (`model_parameters.csv`) |
| Real traces are top-1 only | **Fixed.** Both routed experts are used. | — |
| "No migration cost; not yet run" | **Now run.** | §11.5, `migration_sensitivity.csv` |

### B2. New results the report should include

- **Token-level residency:** at layer 15, LRU serves 59.5% of accesses from HBM but only 37.1% of
  tokens entirely from HBM; at layer 31, static keeps more tokens fully resident than LRU
  (43.6% vs 39.8%).
- **Migration cost** (§11.5), **batch size** (§11.6), **CXL pooling** (`pooling_study.csv`).
- **Expert count vs top-k** (`scalability_sweep.csv`): with HBM holding half the experts, top-2
  keeps 35–44% of tokens fully in HBM from 8 to 256 experts; a DeepSeek-V3-like 256-expert top-8
  layer keeps 3.3%.
- **Prefetching negative result:** 70 configurations on 7 datasets, mean accuracy 29.1%; on the
  real traces it hides 3.1–4.9 more points of accesses for 15–16% more CXL traffic and 14–15%
  more energy.
- **Measured energy:** HBM 1.632 pJ/bit; CXL 20.48 pJ/bit = 9.080 measured DRAM (DDR4 proxy) +
  11.4 cited link. One expert fetch: 4.60 mJ from HBM, 57.72 mJ from CXL (`model_parameters.csv`).
- **Link-energy sensitivity** (`energy_link_sensitivity.csv`): with the link swept from 0 to
  14 pJ/bit, LRU's layer-15 saving stays between 6.8% and 8.6% and periodic re-profiling's
  layer-31 saving between 3.9% and 4.9%; the ranking of policies never changes.

### B3. Report structure

- Lead each section with its finding in one sentence, then support it.
- Title-page suggestion: *CXL-Based Memory Optimisation for Mixture-of-Experts Models —
  bandwidth, residency and placement across HBM and CXL-attached memory.* (The earlier subtitle,
  "Depth-dependent expert placement", describes the top-1 artifact.)
- Code: `https://github.com/VedantK2010/moe-memory-tiering`. Live dashboard:
  `https://vedantk2010.github.io/moe-memory-tiering/dashboard/`.
- Before submitting, run `python src/check_numbers.py`: it fails if any number quoted in the
  README or in this file no longer matches `results/`.
