# Report addendum

Two parts:

- **Part A** is paste-ready prose for the report — the Limitations and References sections the
  current draft does not have.
- **Part B** is a change-log for the authors: claims in the draft that the code no longer
  supports, and what to do about each.

---

# Part A — sections to paste into the report

## Section 11. Limitations and Threats to Validity

Every result in this report rests on a simulation, and every simulation leaves something out.
The omissions below are load-bearing: each is a specific reason a reader should discount a
specific claim, and each is a place where a more complete model could change the answer.

### 11.1 The model charges no migration cost

When an expert is not resident in HBM, we charge the CXL read that fetches it. We do not charge
the cost of installing it into HBM, and we do not charge the write-back of whatever it evicts.

This is the most consequential simplification in the report, and it is not neutral between the
policies we compare. It flatters exactly the policies that migrate most. Pure LRU can evict and
re-install on consecutive accesses; periodic re-profiling migrates at most once per window;
static placement never migrates at all. Our model gives LRU its adaptivity for free.

The direction of the bias is therefore known even though its magnitude is not: a migration-aware
model would penalise LRU hardest, static least, and periodic re-profiling somewhere in between.
Our Layer 15 finding — that LRU wins by 7.4 points — has the smallest margin of safety against
this correction, and could plausibly reverse. **This experiment has not been run.** It is the
single highest-value piece of future work in this project.

### 11.2 Absolute latencies are not comparable to real inference

The model charges a full expert weight transfer on every access. Real MoE inference does not
work this way: once an expert's weights are resident, a decoder step reads them at tile
granularity through the normal cache hierarchy and does not re-fetch the whole expert per token.

The consequence is that our absolute access times — hundreds of microseconds — are far above
what any real decode step costs. They should not be quoted as predicted inference latencies.

What survives is the comparison. Every policy is charged under identical rules, so the *ratios*
between policies, and the *slope* of latency against HBM capacity, remain meaningful. Read this
report's timing results as relative, never absolute.

### 11.3 Expert size is a placeholder

We model an expert as 16 MiB. This was chosen to lay out a non-overlapping address space for the
DRAMSim3 traces, not to match a real model.

Mixtral's actual per-expert FFN is three matrices of 4096 × 14336, about 176 million parameters,
or roughly 336 MiB at fp16 — around 21× our figure. Because access time is
`latency + size/bandwidth` and the transfer term dominates at these sizes, using the real figure
scales every absolute latency by approximately that factor. Hit rates, and therefore the policy
ranking, are unaffected. The interactive dashboard exposes expert size as a parameter so the
sensitivity can be inspected directly.

### 11.4 The real-trace analysis models top-1 routing only

Mixtral routes each token to its top-2 experts. Our extraction from
`allenai/analysis_mixtral` records the first-choice expert per token, and the real-trace
policy comparison (Section 6) operates on that single stream.

Real HBM pressure is therefore higher than we model, since a genuine top-2 workload touches
roughly twice as many experts per token. All hit rates reported from the real traces should be
read as optimistic. The synthetic experiments (Sections 4, 5, 7) do model top-2 routing.

### 11.5 The energy constant includes background power

Our 21.08 nJ/fetch figure is DRAMSim3's reported total energy divided by its request count. That
total includes background and refresh power accumulated across the whole simulated window, not
only the energy attributable to moving 64 bytes. A 64-byte HBM2 burst costs on the order of a
quarter of a nanojoule at a typical ~4 pJ/bit; our figure is roughly two orders of magnitude
above that, because it is measuring a different quantity — energy per request *in that window*.

This does not invalidate Section 9's conclusion. Every policy is charged the same constant, so
it cancels in every ratio, and the reported percentage savings are sound. **The absolute
millijoule figures are not transferable to silicon and should be presented as relative results
only.** The CXL 3× multiplier is likewise a published ballpark for driving a PCIe Gen5 PHY, not
a measurement of our system; a sensitivity sweep from 1.5× to 6× is included in
`src/calc_energy.py`, and the saving scales with the multiplier without ever reversing sign.

### 11.6 Memory parameters are published ballparks

HBM at 150 ns / 800 GB/s and CXL at +70 ns / 64 GB/s are defensible figures drawn from published
engineering sources, not the datasheet numbers of any specific part we tested. They are intended
to place the two tiers in a realistic relationship to each other, which is what the comparison
needs; they are not a claim about a particular product.

### 11.7 One model family, one topology

Every real measurement in this report comes from Mixtral 8x7B: eight experts per layer, top-2
routing. Contemporary MoE models increasingly use finer-grained routing — DeepSeek-V3 and
Qwen-MoE use dozens to hundreds of smaller experts — which produces a flatter selection
distribution and a larger working set.

A flatter distribution is materially harder for every policy studied here, since all four rely on
some experts being reliably hotter than others. We tested synthetic stand-ins for 64-expert
top-8 and 60-expert top-4 topologies in the prefetching study (Section 8), but these vary only
the expert count and top-k; they are not measurements of those models and should not be
described as such. **Whether the layer-depth finding generalises beyond Mixtral is untested.**

### 11.8 A correction to the static-placement baseline

Our original real-trace harness profiled the static policy's top-4 experts over the entire trace
and then scored that same trace. This is look-ahead: the static policy was given knowledge of the
future that LRU and periodic re-profiling were denied, which inflated its hit rate by an unknown
amount and made the three-way comparison invalid.

The harness now profiles static placement on a held-out prefix (the first 10% of the trace) and
scores all three policies on the remainder only, so every policy faces the same unseen window.
The old oracle figure is still computed and printed alongside, labelled as such, so the size of
the bias is visible rather than hidden.

The static hit rates quoted in earlier drafts (57.51% at layer 15, 65.76% at layer 31) come from
the oracle variant and must be regenerated before publication. Note that the correction makes
static *worse*, which strengthens rather than weakens the report's central finding at layer 15,
and narrows the already-small margin at layer 31.

### 11.9 Prefetching: what we now think

Section 8 originally reported that Markov prefetching improved effective hit rate. Two defects
in that experiment have since been found and fixed.

First, the predictor took an argmax over an all-zero transition table before it had observed
anything, which always returned expert 0. Early in each run it prefetched expert 0 continuously
and was credited with a hit whenever expert 0 happened to be next. An untrained context now
predicts nothing.

Second, and more importantly, a wrong prediction cost nothing in the model. A prefetch moves an
expert's weights across the CXL link whether or not the guess was correct, consuming bandwidth
and energy either way.

Charging for wasted prefetches changes the conclusion. Across 50 configurations — seven datasets,
two Markov orders, five confidence-gating settings — **no configuration improved hit rate while
reducing memory energy.** Mean prediction accuracy was 9.1%. The only settings that move the hit
rate materially are those that issue a prefetch on nearly every access; they gain up to 8 points
of effective hit rate at 1.7× CXL traffic and 1.6× memory energy. On a bandwidth-constrained
expansion link, this is a losing trade.

The underlying reason is structural rather than a defect in the predictor. The most predictable
successor to any expert access is the same expert again — and that expert is, by definition,
already resident. The prefetchable cases are precisely the unpredictable ones. We report this as
a negative result.

### 11.10 Summary of what would change our conclusions

| Correction | Expected effect |
|---|---|
| Charge migration cost on cache installs | Penalises LRU most; could reverse the Layer 15 result in favour of periodic re-profiling |
| Model true top-2 routing on the real traces | Lowers all real-trace hit rates; relative ordering likely preserved |
| Use real 336 MiB expert size | Scales absolute latencies ~21×; ranking unaffected |
| Re-run static baseline without look-ahead | Lowers static's hit rate; widens the Layer 15 gap, narrows Layer 31 |
| Test a 64- or 128-expert model | Unknown. Flatter routing may weaken every policy studied here |

---

## Section 12. References

1. Jiang, A. Q., Sablayrolles, A., Roux, A., et al. *Mixtral of Experts.* arXiv:2401.04088, 2024.
   — Expert routing behaviour, temporal locality by layer (Table 5), and the architecture our
   traces are drawn from.

2. Fedus, W., Zoph, B., and Shazeer, N. *Switch Transformers: Scaling to Trillion Parameter
   Models with Simple and Efficient Sparsity.* Journal of Machine Learning Research, 23(120),
   2022. — Expert load-balancing loss and the selection-frequency skew our synthetic traces are
   calibrated against.

3. Li, S., Yang, Z., Reddy, D., Srivastava, A., and Jacob, B. *DRAMsim3: A Cycle-Accurate,
   Thermal-Capable DRAM Simulator.* IEEE Computer Architecture Letters, 19(2), 2020. — The
   cycle-accurate simulator used for our burst-level latency and row-buffer validation.

4. CXL Consortium. *Compute Express Link Specification, Revision 2.0.* 2020. — Memory pooling
   and type-3 device semantics underlying our expansion-tier model.

5. Sun, Y., Yuan, Y., Yu, Z., et al. *Demystifying CXL Memory with Genuine CXL-Ready Systems and
   Devices.* MICRO-56, 2023. — Measured CXL latency overheads on real hardware; the source of
   our ~70 ns added-latency figure.

6. Mattson, R. L., Gecsei, J., Slutz, D. R., and Traiger, I. L. *Evaluation Techniques for
   Storage Hierarchies.* IBM Systems Journal, 9(2), 1970. — The stack-distance (inclusion
   property) result that lets us derive LRU hit rates at every cache capacity from a single pass
   over the trace.

7. Belady, L. A. *A Study of Replacement Algorithms for a Virtual-Storage Computer.* IBM Systems
   Journal, 5(2), 1966. — The optimal offline replacement policy, the upper bound against which
   any online policy in this report should ultimately be measured.

8. Allen Institute for AI. *allenai/analysis_mixtral.* HuggingFace Datasets. — The recorded
   per-token, per-layer expert routing decisions used for all real-trace results.

---

# Part B — change-log for the authors

Items the code now contradicts, in the order they matter.

### B1. Numbers that must be regenerated before submission

`src/run_real_benchmark.py` no longer gives the static policy look-ahead. Re-run it once
`data/real_expert_trace*.csv` are present:

```bash
python run_all.py --stage real-trace energy
```

This regenerates `results/real_trace_hit_rates.csv` and `results/energy_metrics.csv`. Both
dashboards pick the new values up automatically; the report's Section 6 and Section 9 tables
must be updated by hand. Expect static's hit rate to fall at both layers.

### B2. The prefetching section needs rewriting

Section 8's positive claim does not survive the two fixes described in §11.9 above. The honest
version is a negative result, and §11.9 is written to be pasted in as its replacement.
Supporting data is in `results/prefetch_multi_dataset.csv` (50 rows).

### B3. Claims that are unchanged and safe to keep

Verified byte-for-byte against regenerated output during the code review — these did not move:

- The full capacity/latency sweep (`tiering_sweep_results.csv`)
- Smart-vs-random placement, peaking at 9.85% at k=6
- The four-policy phase comparison and the 2,000-token re-profiling optimum
- The robustness result, including the hybrid falling from 2nd to 4th
- The 8.03% Layer 15 energy saving

### B4. Formatting and structure (from the doc checklist)

- **References** — Section 12 above is ready to paste.
- **Limitations** — Section 11 above. The draft currently has no limitations section at all,
  which is the single most conspicuous gap for a technical reader.
- **Title page** — suggested block:

  > **CXL-Based Memory Optimisation for Mixture-of-Experts Models**
  > Depth-dependent expert placement across HBM and CXL-attached memory
  >
  > Vedant Kabra · Neil Verma
  > September 2026
  >
  > Code and interactive dashboard: `<repository URL>`

- **Readability** — the most effective single change is to lead each section with its finding in
  one sentence, then support it, rather than building to the result. A reader skimming section
  headings should be able to reconstruct the argument.
- **Highlighted content** — flagged for deletion in the draft; not visible from the repository,
  so left to the authors.

### B5. Repository items from the checklist

- `README.md` rewritten: quick start, findings, file tour, limitations, references.
- `requirements.txt` added; `run_all.py` reproduces every result in `results/` in about 90 seconds.
- Absolute paths from a single author's machine removed from four scripts (they prevented the
  project from running anywhere else); all paths now resolve relative to the repository.
- Adding Neil Verma as a collaborator and the repository visibility settings are GitHub-side
  actions, not repository contents — still outstanding.
