"""BM25 initial retrieval - top 5000 candidates per claim."""

from __future__ import annotations

import pickle
import re

import nltk
import pandas as pd
from nltk.corpus import stopwords
from rank_bm25 import BM25Okapi
from tqdm import tqdm

from task1 import config

nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)

STOP_WORDS = set(stopwords.words("english"))


def preprocess(text: str) -> list[str]:
    """Lowercase → remove non-alpha → tokenize → remove stopwords."""
    if not isinstance(text, str):
        return []
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = nltk.word_tokenize(text)
    return [w for w in tokens if w not in STOP_WORDS and len(w) > 1]


def retrieve_top_n(query_tokens, bm25, n):
    scores = bm25.get_scores(query_tokens)
    top_idx = scores.argsort()[::-1][:n]
    return top_idx.tolist()


def main():
    print("Loading data...")
    corpus = pd.read_parquet(config.CORPUS_PATH)
    df_train = pd.read_parquet(config.TRAIN_PATH)
    df_test = pd.read_parquet(config.TEST_PATH)

    # Tokenize corpus
    print("Tokenizing corpus...")
    tokenized_corpus = [
        preprocess(a) for a in tqdm(corpus["abstract"], desc="tokenize")
    ]

    print("Building BM25 index...")
    bm25 = BM25Okapi(tokenized_corpus)

    # Mapping from positional index → abstract_id
    abstract_ids = corpus["abstract_id"].tolist()

    # Retrieve for training claims
    claim_id_col = "claim_id" if "claim_id" in df_train.columns else None
    train_claims = df_train.drop_duplicates(subset=[claim_id_col or "claim"])
    print(f"Retrieving top {config.BM25_TOP_N} for {len(train_claims)} train claims...")

    res_train = {}
    for _, row in tqdm(
        train_claims.iterrows(), total=len(train_claims), desc="BM25 train"
    ):
        cid = row[claim_id_col] if claim_id_col else row["claim"]
        tokens = preprocess(row["claim"])
        top_indices = retrieve_top_n(tokens, bm25, config.BM25_TOP_N)
        res_train[cid] = [abstract_ids[i] for i in top_indices]

    with open(config.BM25_TRAIN_PKL, "wb") as f:
        pickle.dump(res_train, f)
    print(f"  Saved → {config.BM25_TRAIN_PKL}")

    # Retrieve for test claims
    print(f"Retrieving top {config.BM25_TOP_N} for {len(df_test)} test claims...")
    res_test = {}
    for _, row in tqdm(df_test.iterrows(), total=len(df_test), desc="BM25 test"):
        cid = row["claim_id"]
        tokens = preprocess(row["claim"])
        top_indices = retrieve_top_n(tokens, bm25, config.BM25_TOP_N)
        res_test[cid] = [abstract_ids[i] for i in top_indices]

    with open(config.BM25_TEST_PKL, "wb") as f:
        pickle.dump(res_test, f)
    print(f"  Saved → {config.BM25_TEST_PKL}")

    print("Done.")


if __name__ == "__main__":
    main()
