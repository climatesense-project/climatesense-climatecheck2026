"""Evaluate a reranker model (pre-trained or fine-tuned) on the validation split.

Usage:
    # Local fine-tuned model:
    python evaluate_reranker.py /path/to/reranker_model
    python evaluate_reranker.py /path/to/reranker_model --margin 0.3

    # Base model from HF:
    python evaluate_reranker.py BAAI/bge-reranker-large
    python evaluate_reranker.py BAAI/bge-reranker-large --margin 0.3
"""

from __future__ import annotations

import argparse
import os
import pickle

import torch
from climatecheck2026_task1 import config
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class TripletDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        pos = self.tokenizer(
            item["query"],
            item["pos"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        neg = self.tokenizer(
            item["query"],
            item["neg"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "pos_input_ids": pos["input_ids"].squeeze(0),
            "pos_attention_mask": pos["attention_mask"].squeeze(0),
            "neg_input_ids": neg["input_ids"].squeeze(0),
            "neg_attention_mask": neg["attention_mask"].squeeze(0),
        }


def is_local_path(model_name_or_dir: str) -> bool:
    """Check if path exists locally, else treat as HF model name."""
    return os.path.isdir(model_name_or_dir)


def evaluate_model(model_name_or_dir: str, margin: float):
    """Evaluate the given model on the validation split of the triplet dataset."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Model: {model_name_or_dir}")

    # Load data and create val split
    with open(config.TRAIN_TRIPLETS_PKL, "rb") as f:
        all_data = pickle.load(f)

    if is_local_path(model_name_or_dir):
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_dir)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_dir)

    if is_local_path(model_name_or_dir):
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_dir
        ).to(device)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_dir
        ).to(device)

    model.eval()

    full_dataset = TripletDataset(all_data, tokenizer, config.RERANKER_MAX_LENGTH)
    val_size = int(len(full_dataset) * config.VAL_FRACTION)
    train_size = len(full_dataset) - val_size
    _, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config.RANDOM_SEED),
    )
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    print(f"Val triplets: {len(val_dataset)}")

    # Evaluation
    correct = 0
    total = 0
    val_loss_total = 0.0

    criterion = torch.nn.MarginRankingLoss(margin=margin)

    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Evaluating")
        for batch in pbar:
            pos_out = model(
                input_ids=batch["pos_input_ids"].to(device),
                attention_mask=batch["pos_attention_mask"].to(device),
            )
            neg_out = model(
                input_ids=batch["neg_input_ids"].to(device),
                attention_mask=batch["neg_attention_mask"].to(device),
            )
            pos_scores = pos_out.logits.squeeze(-1)
            neg_scores = neg_out.logits.squeeze(-1)
            targets = torch.ones(pos_scores.size(0), device=device)

            loss = criterion(pos_scores, neg_scores, targets)
            val_loss_total += loss.item()

            correct += (pos_scores > neg_scores).sum().item()
            total += pos_scores.size(0)

    avg_val_loss = val_loss_total / len(val_loader)
    val_acc = correct / total if total > 0 else 0.0

    print("\nResults:")
    print(f"Validation loss: {avg_val_loss:.4f}")
    print(f"Validation accuracy (pos > neg): {val_acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model_name_or_dir",
        type=str,
        help="HF model name (e.g. 'BAAI/bge-reranker-large') or local path",
    )
    parser.add_argument(
        "--margin", type=float, default=0.2, help="Margin for MarginRankingLoss"
    )
    args = parser.parse_args()
    args = parser.parse_args()
    evaluate_model(args.model_name_or_dir, margin=args.margin)
