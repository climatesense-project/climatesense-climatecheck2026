"""Analyze ClimateCheck 2026 data and create stratified train/val split."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from task1 import config


def main():
    # Load datasets
    df_train = pd.read_parquet(config.TRAIN_PATH)
    df_test = pd.read_parquet(config.TEST_PATH)
    corpus = pd.read_parquet(config.CORPUS_PATH)

    # Corpus overview
    print(f"\n{'-' * 40}")
    print(f"Publications corpus: {len(corpus):,} abstracts")
    print(f"Columns: {list(corpus.columns)}")
    null_abstracts = corpus["abstract"].isna().sum()
    print(f"Null abstracts: {null_abstracts}")
    print(
        f"Avg abstract length (chars): {corpus['abstract'].dropna().str.len().mean():.0f}"
    )

    # Training set overview
    print(f"\n{'-' * 40}")
    print(f"Training set: {len(df_train):,} claim-abstract pairs")
    print(f"Columns: {list(df_train.columns)}")
    n_unique_claims_train = (
        df_train["claim_id"].nunique()
        if "claim_id" in df_train.columns
        else df_train["claim"].nunique()
    )
    claim_id_col = "claim_id" if "claim_id" in df_train.columns else "claim"
    print(f"Unique claims: {n_unique_claims_train}")

    if "annotation" in df_train.columns:
        print("\nLabel distribution (annotation):")
        label_counts = df_train["annotation"].value_counts()
        for label, cnt in label_counts.items():
            print(f"  {label:30s}: {cnt:5d}  ({cnt / len(df_train) * 100:5.1f}%)")

    if "data_version" in df_train.columns:
        print("\nData version distribution:")
        for ver, cnt in df_train["data_version"].value_counts().items():
            print(f"  {ver!s:30s}: {cnt:5d}")

    # Abstracts per claim
    pairs_per_claim = df_train.groupby(claim_id_col).size()
    print(
        f"\nAbstracts per claim — min: {pairs_per_claim.min()}, "
        f"max: {pairs_per_claim.max()}, "
        f"mean: {pairs_per_claim.mean():.1f}, "
        f"median: {pairs_per_claim.median():.1f}"
    )

    # Label distribution per claim
    if "annotation" in df_train.columns:
        label_per_claim = (
            df_train.groupby(claim_id_col)["annotation"]
            .value_counts()
            .unstack(fill_value=0)
        )
        print("\nPer-claim label stats:")
        for col in label_per_claim.columns:
            vals = label_per_claim[col]
            print(f"  {col:30s}: mean={vals.mean():.2f}, max={vals.max()}")

    # Test set overview
    print(f"\n{'-' * 40}")
    print(f"Test set: {len(df_test):,} claims")
    print(f"Columns: {list(df_test.columns)}")

    # Check overlap
    if "claim" in df_train.columns and "claim" in df_test.columns:
        overlap = set(df_train["claim"].unique()) & set(df_test["claim"].unique())
        print(f"\nClaim text overlap train/test: {len(overlap)}")

    # Stratified train/val split
    # Split at the *claim* level so all pairs for a claim stay together.
    # Stratify by the majority label of each claim.
    print(f"\n{'-' * 40}")
    print("Creating stratified train/val split...")

    if "annotation" in df_train.columns:
        # Determine majority label per claim for stratification
        majority_label = df_train.groupby(claim_id_col)["annotation"].agg(
            lambda x: x.value_counts().index[0]
        )
        unique_claims = majority_label.index.tolist()
        strat_labels = majority_label.values

        train_claims, val_claims = train_test_split(
            unique_claims,
            test_size=config.VAL_FRACTION,
            random_state=config.RANDOM_SEED,
            stratify=strat_labels,
        )
    else:
        unique_claims = df_train[claim_id_col].unique().tolist()
        train_claims, val_claims = train_test_split(
            unique_claims,
            test_size=config.VAL_FRACTION,
            random_state=config.RANDOM_SEED,
        )

    df_tr = df_train[df_train[claim_id_col].isin(train_claims)]
    df_va = df_train[df_train[claim_id_col].isin(val_claims)]

    df_tr.to_parquet(config.TRAIN_SPLIT_PATH, index=False)
    df_va.to_parquet(config.VAL_SPLIT_PATH, index=False)

    print(f"  Train split: {len(df_tr):,} pairs ({len(train_claims)} claims)")
    print(f"  Val   split: {len(df_va):,} pairs ({len(val_claims)} claims)")

    print("\n  Train label distribution:")
    for label, cnt in df_tr["annotation"].value_counts().items():
        print(f"    {label:30s}: {cnt:5d}  ({cnt / len(df_tr) * 100:5.1f}%)")
    print("  Val label distribution:")
    for label, cnt in df_va["annotation"].value_counts().items():
        print(f"    {label:30s}: {cnt:5d}  ({cnt / len(df_va) * 100:5.1f}%)")

    print("\nDone.")


if __name__ == "__main__":
    main()
