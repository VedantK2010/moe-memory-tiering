"""
Synthetic MoE Expert-Access Trace Generator
=============================================
Generates a token-by-token expert routing sequence for a MoE model,
calibrated against two empirical findings from the papers you've read:

1. Skewed expert popularity (Switch Transformer, Sec 2.2 / Mixtral Fig. 7):
   routing is not perfectly uniform across experts even with load-balancing
   loss active during training.

2. Temporal locality (Mixtral, Table 5 / Figure 10): consecutive tokens
   reuse the same expert far more often than random chance would predict.
   Reported "first choice" repeat rates: ~14% at layer 0 (near random,
   since 1/8 = 12.5%), rising to ~25-30% at deeper layers.

This trace is the input you'll feed into DRAMSim3 as a memory access
pattern -- each expert is treated as if it occupies its own region of
memory, and each token "visits" the memory region(s) of its chosen
expert(s) to fetch that expert's weights.

Output:
  - expert_trace.csv       : token_id, expert_1, expert_2 (human-readable)
  - dramsim3_trace.txt     : DRAMSim3-format memory address trace
  - trace_summary.png      : validation plots (popularity + locality)
"""

import matplotlib
matplotlib.use("Agg")           # headless: never try to open a window

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import paths

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
NUM_TOKENS = 50_000       # length of the synthetic token stream
NUM_EXPERTS = 8           # matches Mixtral's 8 experts per layer
TOP_K = 2                 # matches Mixtral's top-2 routing
SEED = 42

# Locality (same-expert repeat probability), calibrated from Mixtral Table 5
# "First choice" repeats. We use the deep-layer number since deeper layers
# show the most exploitable structure -- a reasonable target for tiering.
P_REPEAT = 0.27

# Expert weight size, used only to lay experts out in a fake address space.
# This does NOT need to be a real model's weight size -- it just needs to
# be consistent so DRAMSim3 sees distinct, non-overlapping address regions
# per expert. 16 MiB per expert is a placeholder; adjust later once you
# decide on a real reference model size.
EXPERT_SIZE_BYTES = 16 * 1024 * 1024
BASE_ADDRESS = 0x00000000


def generate_base_popularity(num_experts, skew_alpha, seed):
    """
    Dirichlet distribution gives a smooth, skewed set of probabilities
    that sum to 1. Lower alpha => more skew (more pronounced hot/cold
    split). alpha=6 gives a mild skew similar in spirit to Mixtral's
    Figure 7, where proportions ranged roughly 0.08-0.19 around a
    uniform baseline of 0.125 -- not extreme, but clearly non-uniform.
    """
    rng = np.random.default_rng(seed)
    return rng.dirichlet(alpha=[skew_alpha] * num_experts)


def generate_expert_sequence(num_tokens, num_experts, top_k, p_repeat_target, base_probs, seed):
    """
    Generate the token-by-token expert assignment sequence.

    For each token:
      - with probability p_forced_repeat, the token's first-choice expert
        is forced to be the SAME as the previous token's (models temporal
        locality)
      - otherwise, sample a first-choice expert from the base popularity
        distribution (models overall hot/cold skew) -- this can still
        coincidentally match the previous expert by chance
      - second-choice (and further, if top_k > 2) experts are sampled from
        the remaining experts, weighted by base popularity

    p_repeat_target is the DESIRED overall consecutive-repeat rate (the
    number you're calibrating against, e.g. from Mixtral Table 5). Because
    random sampling can coincidentally repeat the same expert even without
    forcing it, the forced-repeat probability must be solved for:

        p_repeat_target = p_forced + (1 - p_forced) * collision_prob

    where collision_prob = sum(base_probs^2) is the chance two independent
    draws from the popularity distribution land on the same expert. Solving
    for p_forced:

        p_forced = (p_repeat_target - collision_prob) / (1 - collision_prob)
    """
    collision_prob = float(np.sum(base_probs ** 2))
    p_forced = (p_repeat_target - collision_prob) / (1 - collision_prob)
    p_forced = max(0.0, p_forced)  # guard against a target below the natural collision rate

    rng = np.random.default_rng(seed + 1)
    choices = np.zeros((num_tokens, top_k), dtype=int)
    prev_first = int(rng.choice(num_experts, p=base_probs))

    for t in range(num_tokens):
        if t > 0 and rng.random() < p_forced:
            first = prev_first
        else:
            first = int(rng.choice(num_experts, p=base_probs))

        remaining = [e for e in range(num_experts) if e != first]
        remaining_probs = base_probs[remaining] / base_probs[remaining].sum()
        rest = rng.choice(remaining, size=top_k - 1, replace=False, p=remaining_probs)

        choices[t, 0] = first
        choices[t, 1:] = rest
        prev_first = first

    return choices


def write_dramsim3_trace(choices, expert_size_bytes, base_address, out_path):
    """
    Write a DRAMSim3-style memory access trace. Each expert is assigned a
    fixed, non-overlapping address range. Every time a token selects an
    expert, we emit one READ access into that expert's address range
    (simulating fetching that expert's weights from memory).

    DRAMSim3 trace line format (plain text mode):
        <hex address>  <READ|WRITE>  <clock cycle, informational>

    NOTE: Confirm this exact format against your installed DRAMSim3
    version's trace reader before running -- some versions expect a
    slightly different column order or a leading '0x'. This is meant as
    a correct starting point, not gospel.
    """
    with open(out_path, "w") as f:
        cycle = 0
        for row in choices:
            for expert_id in row:
                addr = base_address + expert_id * expert_size_bytes
                f.write(f"0x{addr:010x} READ {cycle}\n")
                cycle += 100  # arbitrary spacing; refine once timing model is decided


def summarize_and_plot(choices, base_probs, num_experts, p_repeat_target, out_path):
    """Sanity-check the generated trace against the two calibration targets."""
    first_choices = choices[:, 0]

    # 1. Empirical selection frequency per expert (this is f_i from the
    #    Switch Transformer paper, Equation 5)
    f_i = np.array([(first_choices == e).mean() for e in range(num_experts)])

    # 2. Empirical repeat rate: fraction of consecutive tokens where the
    #    first-choice expert is unchanged (this is what Mixtral Table 5
    #    reports per layer/domain)
    repeats = (first_choices[1:] == first_choices[:-1]).mean()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].bar(range(num_experts), f_i, color="#4C72B0", label="Empirical $f_i$")
    axes[0].axhline(1 / num_experts, color="gray", linestyle="--", label="Uniform (1/N)")
    axes[0].plot(range(num_experts), base_probs, "ko", label="Target popularity")
    axes[0].set_xlabel("Expert ID")
    axes[0].set_ylabel("Selection proportion")
    axes[0].set_title("Expert popularity ($f_i$)")
    axes[0].legend(fontsize=8)

    axes[1].bar(["Target\n(from Table 5)", "Empirical\n(generated trace)"],
                [p_repeat_target, repeats], color=["gray", "#DD8452"])
    axes[1].set_ylabel("Consecutive same-expert rate")
    axes[1].set_title("Temporal locality check")
    axes[1].set_ylim(0, max(p_repeat_target, repeats) * 1.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)

    print(f"Per-expert selection proportions (f_i): {np.round(f_i, 4).tolist()}")
    print(f"Target repeat rate: {p_repeat_target:.3f}  |  Empirical repeat rate: {repeats:.3f}")


if __name__ == "__main__":
    base_probs = generate_base_popularity(NUM_EXPERTS, skew_alpha=6, seed=SEED)
    print(f"Base (target) popularity distribution: {np.round(base_probs, 4).tolist()}")

    choices = generate_expert_sequence(
        NUM_TOKENS, NUM_EXPERTS, TOP_K, P_REPEAT, base_probs, SEED
    )

    df = pd.DataFrame(
        choices, columns=[f"expert_{i+1}" for i in range(TOP_K)]
    )
    df.insert(0, "token_id", np.arange(NUM_TOKENS))
    trace_path = paths.data("expert_trace.csv")
    df.to_csv(trace_path, index=False)

    dram_path = paths.data("dramsim3_trace.txt")
    write_dramsim3_trace(choices, EXPERT_SIZE_BYTES, BASE_ADDRESS, dram_path)

    plot_path = paths.result("trace_summary.png")
    summarize_and_plot(choices, base_probs, NUM_EXPERTS, P_REPEAT, plot_path)

    print(f"\nWrote: {trace_path}\n       {dram_path}\n       {plot_path}")
