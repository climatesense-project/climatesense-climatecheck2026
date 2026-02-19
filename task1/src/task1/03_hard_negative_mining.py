"""Hard-negative mining + training triplet generation for reranker fine-tuning."""

from __future__ import annotations

import pickle
import random

import pandas as pd
from FlagEmbedding import FlagReranker
from tqdm import tqdm

from task1 import config


def main():
    random.seed(config.RANDOM_SEED)

    print("Loading data...")
    df_train = pd.read_parquet(config.TRAIN_PATH)
    corpus = pd.read_parquet(config.CORPUS_PATH)

    with open(config.BM25_TRAIN_PKL, "rb") as f:
        bm25_train = pickle.load(f)

    # Build abstract_id → abstract text lookup
    abstract_lookup = dict(zip(corpus["abstract_id"], corpus["abstract"], strict=True))

    claim_id_col = "claim_id"
    claim_ids = list(bm25_train.keys())

    # Stage 1: Mine hard negatives with pretrained reranker
    print(f"Loading pretrained reranker: {config.HARD_NEG_RERANKER_MODEL}")
    reranker = FlagReranker(config.HARD_NEG_RERANKER_MODEL, use_fp16=True)

    hard_negatives = {}  # claim_id → list of abstract_ids

    for cid in tqdm(claim_ids, desc="Hard-neg mining"):
        claim_text = df_train.loc[df_train[claim_id_col] == cid, "claim"].values[0]
        bm25_candidates = bm25_train[cid]

        # Build pairs for reranker scoring
        pairs = [(claim_text, abstract_lookup.get(aid, "")) for aid in bm25_candidates]
        scores = reranker.compute_score(pairs)

        # Sort by score descending
        scored = sorted(
            zip(bm25_candidates, scores, strict=True), key=lambda x: x[1], reverse=True
        )

        # Gold positives for this claim
        gold_ids = set(
            df_train.loc[df_train[claim_id_col] == cid, "abstract_id"].tolist()
        )

        # Select top non-gold as hard negatives
        hard_negs = []
        for aid, _ in scored[: config.HARD_NEG_TOP_K]:
            if aid not in gold_ids:
                hard_negs.append(aid)
                if len(hard_negs) >= config.HARD_NEG_SELECT:
                    break
        hard_negatives[cid] = hard_negs

    with open(config.HARD_NEG_PKL, "wb") as f:
        pickle.dump(hard_negatives, f)
    print(f"  Hard negatives saved → {config.HARD_NEG_PKL}")

    # Stage 2: Build training triplets
    print("Building training triplets...")
    all_abstracts = df_train["abstract"].tolist()
    triplets = []

    for idx in tqdm(range(len(df_train)), desc="Triplets"):
        row = df_train.iloc[idx]
        cid = row[claim_id_col]
        claim_text = row["claim"]
        pos_text = row["abstract"]

        neg_pool = hard_negatives.get(cid, [])

        for i in range(config.NUM_RANDOM_NEG + config.NUM_HARD_NEG):
            if i < config.NUM_RANDOM_NEG:
                neg_text = random.choice(all_abstracts)
            else:
                if neg_pool:
                    neg_aid = random.choice(neg_pool)
                    neg_text = abstract_lookup.get(neg_aid, "")
                else:
                    neg_text = random.choice(all_abstracts)
            triplets.append({"query": claim_text, "pos": pos_text, "neg": neg_text})

    with open(config.TRAIN_TRIPLETS_PKL, "wb") as f:
        pickle.dump(triplets, f)

    print(f"  {len(triplets):,} triplets saved → {config.TRAIN_TRIPLETS_PKL}")
    print("Done.")


if __name__ == "__main__":
    main()
