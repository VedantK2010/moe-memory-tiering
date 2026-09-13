"""
Synthetic MoE Expert-Routing Trace Generators
==============================================
Produces the three synthetic traces used in this study, all calibrated
against published Mixtral routing behaviour:

  expert_trace.csv        stationary  -- one fixed popularity distribution
  nonstationary_trace.csv phased      -- hot experts rotate every phase
  dataset2_trace.csv      phased      -- independent seed AND stronger skew,
                                         used for the robustness check

CALIBRATION TARGETS
-------------------
1. Skewed expert popularity (Switch Transformer Sec 2.2, Mixtral Fig. 7):
   routing is non-uniform even with a load-balancing loss. Modelled with a
   Dirichlet draw; lower alpha = stronger hot/cold split.

2. Temporal locality (Mixtral Table 5): consecutive tokens reuse the same
   first-choice expert more often than chance. Reported repeat rates run
   ~14% at layer 0 (near the 1/8 = 12.5% random baseline) up to ~25-30%
   deeper in the stack.

   Our real Mixtral traces measure 23.3% at layer 15 and 23.2% at layer 31,
   which is why P_REPEAT defaults to 0.27 -- close to, and slightly above,
   what the real data shows.

The forced-repeat probability is SOLVED FOR rather than set directly,
because independent draws from a skewed distribution already repeat by
chance at rate sum(p_i^2):

    p_repeat_target = p_forced + (1 - p_forced) * collision_prob
    p_forced        = (p_repeat_target - collision_prob) / (1 - collision_prob)

Also writes a DRAMSim3-format address trace for the stationary case, used
to measure HBM device parameters.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (
    DATA_DIR, RESULTS_DIR, ENCODING,
    NUM_EXPERTS, TOP_K, EXPERT_SIZE_BYTES, SEED,
    SYNTH_NUM_TOKENS, SYNTH_SKEW_ALPHA, SYNTH_P_REPEAT,
    NONSTATIONARY_TOKENS_PER_PHASE, NONSTATIONARY_NUM_PHASES, NONSTATIONARY_SEED,
    DATASET2_SEED, DATASET2_SKEW_ALPHA, DATASET2_P_REPEAT,
)

BASE_ADDRESS = 0x00000000


def _solve_forced_repeat(base_probs, p_repeat_target):
    collision = float(np.sum(base_probs ** 2))
    return max(0.0, (p_repeat_target - collision) / (1 - collision))


def _sample_token(rng, base_probs, prev_first, p_forced, num_experts, top_k):
    """One token's routed experts: a first choice (with forced locality) plus
    top_k-1 distinct others drawn by popularity."""
    if prev_first is not None and rng.random() < p_forced:
        first = prev_first
    else:
        first = int(rng.choice(num_experts, p=base_probs))
    remaining = [e for e in range(num_experts) if e != first]
    probs = base_probs[remaining] / base_probs[remaining].sum()
    rest = rng.choice(remaining, size=top_k - 1, replace=False, p=probs)
    return [first] + list(rest)


def generate_stationary(num_tokens=SYNTH_NUM_TOKENS, num_experts=NUM_EXPERTS,
                        top_k=TOP_K, skew_alpha=SYNTH_SKEW_ALPHA,
                        p_repeat=SYNTH_P_REPEAT, seed=SEED):
    rng = np.random.default_rng(seed)
    base_probs = rng.dirichlet(alpha=[skew_alpha] * num_experts)
    p_forced = _solve_forced_repeat(base_probs, p_repeat)

    rows, prev = [], None
    for _ in range(num_tokens):
        row = _sample_token(rng, base_probs, prev, p_forced, num_experts, top_k)
        rows.append(row)
        prev = row[0]

    df = pd.DataFrame(rows, columns=[f"expert_{i+1}" for i in range(top_k)])
    df.insert(0, "token_id", np.arange(len(df)))
    return df, base_probs


def generate_phased(tokens_per_phase=NONSTATIONARY_TOKENS_PER_PHASE,
                    num_phases=NONSTATIONARY_NUM_PHASES,
                    num_experts=NUM_EXPERTS, top_k=TOP_K,
                    skew_alpha=SYNTH_SKEW_ALPHA, p_repeat=SYNTH_P_REPEAT,
                    seed=NONSTATIONARY_SEED):
    """Back-to-back phases, each with its own hot set.

    The popularity SHAPE is drawn once and then ROTATED each phase, rather
    than redrawn independently. An independent redraw can leave the same
    expert hot by coincidence, which would not test adaptation at all.
    Rotation guarantees the hot set genuinely changes at every boundary.
    """
    rng = np.random.default_rng(seed)
    base_shape = rng.dirichlet(alpha=[skew_alpha] * num_experts)
    roll = max(1, num_experts // num_phases)

    rows, phases, phase_probs = [], [], []
    prev = None
    for phase in range(num_phases):
        base_probs = np.roll(base_shape, shift=phase * roll)
        phase_probs.append(base_probs)
        p_forced = _solve_forced_repeat(base_probs, p_repeat)
        for _ in range(tokens_per_phase):
            row = _sample_token(rng, base_probs, prev, p_forced, num_experts, top_k)
            rows.append(row)
            phases.append(phase)
            prev = row[0]

    df = pd.DataFrame(rows, columns=[f"expert_{i+1}" for i in range(top_k)])
    df.insert(0, "token_id", np.arange(len(df)))
    df["phase"] = phases
    return df, phase_probs


def write_dramsim3_trace(df, out_path, expert_size_bytes=EXPERT_SIZE_BYTES,
                         cycle_step=1, access_bytes=64, accesses_per_expert=1):
    """Write a DRAMSim3 plain-text address trace.

    Each expert occupies its own non-overlapping address region. Every routed
    access emits `accesses_per_expert` reads into that region, continuing from
    where that expert's previous access stopped -- so each expert is streamed
    through its region, as a real 352 MB expert fetch would be.

    WHY THE PER-EXPERT CURSOR: an earlier version restarted every access at the
    region base, so the whole trace touched only 64 distinct addresses
    (8 experts x 8 lines). Two things went wrong in DRAMSim3:
      - Its controller merges a read into an identical one already queued, so
        ~50% of reads never reached DRAM and energy-per-read halved.
      - Expert bases are multiples of 0x15000000 (low 24 bits zero), and the
        first 8 lines never reach the channel bits, so every read went to
        HBM2 channel 0 while channels 1-7 idled.
    Advancing a cursor makes every address unique and walks the channel/bank
    bits naturally.

    cycle_step=1 issues requests back-to-back. The original generator used
    cycle_step=100, which left the memory system ~99% idle and produced an
    idle-latency measurement -- not representative of the bandwidth-bound
    regime this study actually models. Keep it at 1 unless you specifically
    want the unloaded number.
    """
    cols = [c for c in df.columns if c.startswith("expert_")]
    lines_per_expert = expert_size_bytes // access_bytes
    cursor = {}  # expert_id -> next unread line within its region
    # newline="\n": DRAMSim3 reads this on Linux; keep it byte-identical
    # whichever OS generated it.
    with open(out_path, "w", encoding=ENCODING, newline="\n") as f:
        cycle = 0
        for row in df[cols].itertuples(index=False):
            for expert_id in row:
                base = BASE_ADDRESS + int(expert_id) * expert_size_bytes
                start = cursor.get(expert_id, 0)
                for i in range(accesses_per_expert):
                    line = (start + i) % lines_per_expert
                    f.write(f"0x{base + line * access_bytes:010x} READ {cycle}\n")
                    cycle += cycle_step
                cursor[expert_id] = (start + accesses_per_expert) % lines_per_expert


def summarize(df, base_probs, p_repeat_target, out_path, title):
    first = df["expert_1"].to_numpy()
    f_i = np.bincount(first, minlength=NUM_EXPERTS) / len(first)
    repeats = float((first[1:] == first[:-1]).mean())

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(range(NUM_EXPERTS), f_i, color="#4C72B0", label="Empirical $f_i$")
    axes[0].axhline(1 / NUM_EXPERTS, color="gray", ls="--", label="Uniform (1/N)")
    if base_probs is not None:
        axes[0].plot(range(NUM_EXPERTS), base_probs, "ko", label="Target popularity")
    axes[0].set_xlabel("Expert ID")
    axes[0].set_ylabel("Selection proportion")
    axes[0].set_title("Expert popularity ($f_i$)")
    axes[0].legend(fontsize=8)

    axes[1].bar(["Target\n(Mixtral Table 5)", "Empirical\n(generated)"],
                [p_repeat_target, repeats], color=["gray", "#DD8452"])
    axes[1].set_ylabel("Consecutive same-expert rate")
    axes[1].set_title("Temporal locality check")
    axes[1].set_ylim(0, max(p_repeat_target, repeats) * 1.35)

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    return f_i, repeats


def main():
    print("Generating synthetic traces...")

    stationary, base_probs = generate_stationary()
    stationary.to_csv(DATA_DIR / "expert_trace.csv", index=False, encoding=ENCODING)
    f_i, rep = summarize(stationary, base_probs, SYNTH_P_REPEAT,
                         RESULTS_DIR / "trace_summary.png",
                         "Stationary synthetic trace")
    print(f"  expert_trace.csv        {len(stationary):>7,} tokens  "
          f"repeat={rep:.3f}  f_i range {f_i.min():.3f}-{f_i.max():.3f}")

    nonstat, phase_probs = generate_phased()
    nonstat.to_csv(DATA_DIR / "nonstationary_trace.csv", index=False, encoding=ENCODING)
    hot = [int(np.argmax(p)) for p in phase_probs]
    print(f"  nonstationary_trace.csv {len(nonstat):>7,} tokens  hottest per phase: {hot}")

    ds2, ds2_probs = generate_phased(
        skew_alpha=DATASET2_SKEW_ALPHA, p_repeat=DATASET2_P_REPEAT, seed=DATASET2_SEED
    )
    ds2.to_csv(DATA_DIR / "dataset2_trace.csv", index=False, encoding=ENCODING)
    hot2 = [int(np.argmax(p)) for p in ds2_probs]
    print(f"  dataset2_trace.csv      {len(ds2):>7,} tokens  hottest per phase: {hot2}  "
          f"(alpha={DATASET2_SKEW_ALPHA}, stronger skew)")

    # DRAMSim3 address traces.
    #
    # SIZING: 16,000 tokens x TOP_K experts x 8 sequential 64 B reads per
    # expert = 256,000 reads, matching the request count of the original run
    # (32,000 per channel across 8 channels) so the two are comparable.
    #
    # Emitting several sequential reads per expert access is also more
    # faithful than one: a real expert fetch streams 352 MB (~5.5M cache
    # lines) from a contiguous region. Eight is a tractable stand-in that
    # still exercises row-buffer locality honestly. Each access continues
    # where the expert's last one stopped, so no address repeats (see
    # write_dramsim3_trace).
    #
    # cycle_step=1 issues back-to-back, which saturates the memory system --
    # the bandwidth-bound regime this study actually models. The unloaded
    # variant (cycle_step=100) reproduces the original idle measurement, kept
    # only so the two can be compared side by side.
    n_trace_tokens = 16_000
    write_dramsim3_trace(stationary.head(n_trace_tokens), DATA_DIR / "dramsim3_loaded.txt",
                         cycle_step=1, accesses_per_expert=8)
    write_dramsim3_trace(stationary.head(n_trace_tokens), DATA_DIR / "dramsim3_unloaded.txt",
                         cycle_step=100, accesses_per_expert=8)
    n_reads = n_trace_tokens * TOP_K * 8
    print(f"  dramsim3_loaded.txt / dramsim3_unloaded.txt written "
          f"({n_reads:,} reads each; loaded = back-to-back)")


if __name__ == "__main__":
    main()
