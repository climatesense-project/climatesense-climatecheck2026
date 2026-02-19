"""Reranker inference — score BM25 candidates with each fine-tuned reranker.

Usage:
    python 05_reranker_inference.py                  # all models, test set
    python 05_reranker_inference.py --split train    # train set (for eval)
    python 05_reranker_inference.py --config-idx 0   # single model
"""

from __future__ import annotations

import argparse
import glob
import pickle
from pathlib import Path

import pandas as pd
from FlagEmbedding import FlagReranker
from tqdm import tqdm

from task1 import config


def get_model_dirs():
    """Return sorted list of fine-tuned reranker directories."""
    pattern = str(config.MODELS_DIR / "reranker_*")
    dirs = sorted(glob.glob(pattern))
    return [Path(d) for d in dirs if Path(d).is_dir()]


def run_inference(
    model_dir: Path,
    bm25_results: dict,
    claims_df: pd.DataFrame,
    corpus: pd.DataFrame,
    split_name: str,
):
    """Score BM25 candidates with a single reranker; save top-N predictions."""
    abstract_lookup = dict(zip(corpus["abstract_id"], corpus["abstract"], strict=True))
    claim_col = "claim_id"

    print(f"  Loading reranker from {model_dir.name}...")
    reranker = FlagReranker(str(model_dir), use_fp16=True)

    submissions = []
    claim_ids = list(bm25_results.keys())

    for cid in tqdm(claim_ids, desc=f"  {model_dir.name}"):
        claim_text = claims_df.loc[claims_df[claim_col] == cid, "claim"].values[0]
        candidate_aids = bm25_results[cid]

        pairs = [[claim_text, abstract_lookup.get(aid, "")] for aid in candidate_aids]
        scores = reranker.compute_score(pairs)

        scored = sorted(
            zip(candidate_aids, scores, strict=True), key=lambda x: x[1], reverse=True
        )
        top = scored[: config.RERANKER_INFERENCE_TOP_N]

        for rank, (aid, _) in enumerate(top, 1):
            submissions.append({"claim_id": cid, "abstract_id": aid, "rank": rank})

    out_path = (
        config.ARTIFACTS_DIR / f"reranker_preds_{model_dir.name}_{split_name}.csv"
    )
    pd.DataFrame(submissions).to_csv(out_path, index=False)
    print(f"  Saved → {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--config-idx", type=int, default=None)
    args = parser.parse_args()

    corpus = pd.read_parquet(config.CORPUS_PATH)

    if args.split == "test":
        claims_df = pd.read_parquet(config.TEST_PATH)
        with open(config.BM25_TEST_PKL, "rb") as f:
            bm25_results = pickle.load(f)
    else:
        claims_df = pd.read_parquet(config.TRAIN_PATH).drop_duplicates(
            subset=["claim_id"]
        )
        with open(config.BM25_TRAIN_PKL, "rb") as f:
            bm25_results = pickle.load(f)

    model_dirs = get_model_dirs()
    if args.config_idx is not None:
        model_dirs = [model_dirs[args.config_idx]]

    for md in model_dirs:
        print(f"\n{'-' * 50}")
        run_inference(md, bm25_results, claims_df, corpus, args.split)

    print("\nAll done.")


if __name__ == "__main__":
    main()
