"""
Real Mixtral Routing Traces
===========================
Downloads allenai/analysis_mixtral -- expert routing recorded from real
Mixtral 8x7B inference -- and writes the two trace files every real-trace
experiment reads:

  data/real_expert_trace.csv          layer 15 (middle of the stack)
  data/real_expert_trace_layer31.csv  layer 31 (top of the stack)

Each row is one token: token_id, expert_1 (first choice), expert_2 (second
choice). Tokens from every sequence are concatenated in dataset order.

DATA LAYOUT (checked against the dataset, 14 Sept 2026)
------------------------------------------------------
exp_ids[token][rank][layer] -> expert id, with rank 0 = first choice,
rank 1 = second choice, and 32 layers. An earlier version of this script
assumed one expert per token and wrote expert_1 only; the traces actually
used by the pipeline were produced with both ranks, and this version
reproduces that layout.

Needs internet access once. Reads Hugging Face's parquet conversion of the
dataset directly (about 24 MB), so it only needs pandas + pyarrow -- no
`datasets` package and no Colab:

    python src/convert_real_data.py
    python src/convert_real_data.py --parquet path/to/0000.parquet   # offline copy
"""

import argparse

import numpy as np
import pandas as pd

from config import DATA_DIR, ENCODING, REAL_TRACES

PARQUET_URL = ("https://huggingface.co/datasets/allenai/analysis_mixtral/resolve/"
               "refs%2Fconvert%2Fparquet/default/train/0000.parquet")
LAYERS = {15: REAL_TRACES["Layer 15"], 31: REAL_TRACES["Layer 31"]}


def extract(df, layer):
    """One (token_id, expert_1, expert_2) row per token, all sequences in order."""
    first, second = [], []
    for seq in df["exp_ids"]:
        for tok in seq:
            first.append(int(tok[0][layer]))
            second.append(int(tok[1][layer]))
    return pd.DataFrame({"token_id": np.arange(len(first)),
                         "expert_1": first, "expert_2": second})


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--parquet", default=PARQUET_URL,
                    help="parquet file or URL (default: Hugging Face's conversion)")
    args = ap.parse_args()

    print(f"Reading {args.parquet} ...")
    df = pd.read_parquet(args.parquet, columns=["exp_ids"])
    print(f"  {len(df)} sequences")

    DATA_DIR.mkdir(exist_ok=True)
    for layer, path in LAYERS.items():
        trace = extract(df, layer)
        trace.to_csv(path, index=False, encoding=ENCODING)
        repeat = float((trace["expert_1"].values[1:] == trace["expert_1"].values[:-1]).mean())
        print(f"  layer {layer:>2}: {len(trace):,} tokens -> {path.name} "
              f"(consecutive first-choice repeat rate {repeat:.4f})")


if __name__ == "__main__":
    main()
