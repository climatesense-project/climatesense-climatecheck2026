"""Ensemble multiple reranker predictions via Reciprocal Rank Fusion (RRF).

Usage:
    python 06_ensemble_reranking.py                 # test set
    python 06_ensemble_reranking.py --split train   # train set (for eval)
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import pandas as pd

from task1 import config


def ensemble_rrf(
    prediction_dfs: list[pd.DataFrame], k: int, top_n: int
) -> pd.DataFrame:
    """Reciprocal Rank Fusion: score = Σ 1/(k + rank), then re-rank."""
    parts = []
    for df in prediction_dfs:
        tmp = df[["claim_id", "abstract_id", "rank"]].copy()
        tmp["score"] = 1.0 / (k + tmp["rank"])
        parts.append(tmp[["claim_id", "abstract_id", "score"]])

    combined = pd.concat(parts)
    agg = combined.groupby(["claim_id", "abstract_id"])["score"].sum().reset_index()
    agg["rank"] = (
        agg.groupby("claim_id")["score"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    agg = (
        agg[agg["rank"] <= top_n]
        .sort_values(["claim_id", "rank"])
        .reset_index(drop=True)
    )
    return agg[["claim_id", "abstract_id", "rank"]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--top-n", type=int, default=config.ENSEMBLE_TOP_N)
    args = parser.parse_args()

    pattern = str(config.ARTIFACTS_DIR / f"reranker_preds_*_{args.split}.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        print(
            f"No prediction files found for split '{args.split}'. Run 05_reranker_inference.py first."
        )
        return

    print(f"Ensembling {len(files)} prediction files (RRF k={config.RRF_K})...")
    for f in files:
        print(f"  • {Path(f).name}")

    dfs = [pd.read_csv(f) for f in files]
    result = ensemble_rrf(dfs, k=config.RRF_K, top_n=args.top_n)

    out = config.ARTIFACTS_DIR / f"ensemble_top{args.top_n}_{args.split}.csv"
    result.to_csv(out, index=False)
    print(f"\nEnsemble result ({len(result)} rows) → {out}")


if __name__ == "__main__":
    main()
