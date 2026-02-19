"""Download ClimateCheck 2026 datasets from HuggingFace."""

from __future__ import annotations

from datasets import load_dataset

from task1 import config


def main():
    # Training + Testing dataset
    print("Downloading ClimateCheck train/test dataset...")
    ds = load_dataset(config.HF_DATASET)

    ds["train"].to_parquet(str(config.TRAIN_PATH))
    print(f"  train → {config.TRAIN_PATH}  ({ds['train'].num_rows} rows)")

    ds["test"].to_parquet(str(config.TEST_PATH))
    print(f"  test  → {config.TEST_PATH}  ({ds['test'].num_rows} rows)")

    # Publications corpus
    print("Downloading publications corpus...")
    corpus = load_dataset(config.HF_CORPUS)
    split_name = next(iter(corpus.keys()))  # "train"
    corpus[split_name].to_parquet(str(config.CORPUS_PATH))
    print(f"  corpus → {config.CORPUS_PATH}  ({corpus[split_name].num_rows} rows)")

    print("Done.")


if __name__ == "__main__":
    main()
