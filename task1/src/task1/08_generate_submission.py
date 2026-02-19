"""Generate final submission CSV + ZIP for ClimateCheck 2026 Task 1.

Combines retrieval (top-10) with verification labels and generate
predictions.csv and predictions.zip for submission.

Usage:
    python 08_generate_submission.py
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from task1 import config


class PairDataset(Dataset):
    def __init__(self, claims, abstracts, tokenizer, max_length):
        self.claims = claims
        self.abstracts = abstracts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.claims)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.claims[idx],
            self.abstracts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }


def get_logits(model, loader, device):
    """Run model on loader and return raw logits (N x num_labels)."""
    model.eval()
    all_logits = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Predict"):
            out = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )
            all_logits.append(out.logits.cpu().numpy())
    return np.concatenate(all_logits, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=None)
    args = parser.parse_args()

    # Load retrieval results
    if args.input is not None:
        retrieval_path = args.input
    else:
        retrieval_path = (
            config.ARTIFACTS_DIR / f"ensemble_top{config.ENSEMBLE_TOP_N}_test.csv"
        )
    if not retrieval_path.exists():
        print(f"Retrieval file not found: {retrieval_path}")
        return
    df_retrieval = pd.read_csv(retrieval_path)

    # Get verification labels if available
    vllm_path = (
        config.ARTIFACTS_DIR / f"verification_vllm_top{config.ENSEMBLE_TOP_N}.csv"
    )
    if vllm_path.exists():
        df_vllm = pd.read_csv(vllm_path)
        # Merge on claim_id + abstract_id to align labels correctly
        df_retrieval = df_retrieval.merge(
            df_vllm[["claim_id", "abstract_id", "label"]],
            on=["claim_id", "abstract_id"],
            how="left",
        )
        df_retrieval["label"] = df_retrieval["label"].fillna("Not Enough Information")
    else:
        print(
            f"vLLM verification not found: {vllm_path}. Ignoring verification labels."
        )

    # Format submission
    submission = df_retrieval[["claim_id", "abstract_id", "rank", "label"]].copy()

    csv_path = config.SUBMISSIONS_DIR / "predictions.csv"
    submission.to_csv(csv_path, index=False)

    zip_path = config.SUBMISSIONS_DIR / "predictions.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(csv_path, "predictions.csv")

    print(f"Submission CSV → {csv_path}")
    print(f"Submission ZIP → {zip_path}")
    print(f"\n{len(submission)} rows, {submission['claim_id'].nunique()} claims")
    print("\nLabel distribution:")
    print(submission["label"].value_counts().to_string())


if __name__ == "__main__":
    main()
