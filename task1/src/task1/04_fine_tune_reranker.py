"""Fine-tune BGE-Reranker cross-encoders with triplet margin-ranking loss.

Usage:
    # Train all configs sequentially:
    python 04_fine_tune_reranker.py

    # Train a single config by index (0-based):
    python 04_fine_tune_reranker.py --config-idx 2
"""

from __future__ import annotations

import argparse
import pickle

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from task1 import config


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


def train_one_config(cfg, cfg_idx):
    model_name = cfg["base_model"]
    batch_size = cfg["batch_size"]
    margin = cfg["margin"]
    lr = cfg["lr"]

    save_dir = (
        config.MODELS_DIR
        / f"reranker_{cfg_idx}_{model_name.split('/')[-1]}_bs{batch_size}_m{margin}"
    )
    if save_dir.exists():
        print(f"  [skip] {save_dir} already exists")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    with open(config.TRAIN_TRIPLETS_PKL, "rb") as f:
        all_data = pickle.load(f)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)

    # Stratified split of triplets
    full_dataset = TripletDataset(all_data, tokenizer, config.RERANKER_MAX_LENGTH)
    val_size = int(len(full_dataset) * config.VAL_FRACTION)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config.RANDOM_SEED),
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    print(f"  Train triplets: {train_size} | Val triplets: {val_size}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = torch.nn.MarginRankingLoss(margin=margin)

    best_val_acc = 0.0
    for epoch in range(config.RERANKER_EPOCHS):
        # Training
        model.train()
        total_loss = 0.0
        pbar = tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{config.RERANKER_EPOCHS} [train]"
        )
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

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss_total = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in tqdm(
                val_loader, desc=f"Epoch {epoch + 1}/{config.RERANKER_EPOCHS} [val]"
            ):
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
                val_loss_total += criterion(pos_scores, neg_scores, targets).item()
                correct += (pos_scores > neg_scores).sum().item()
                total += pos_scores.size(0)

        avg_val_loss = val_loss_total / len(val_loader)
        val_acc = correct / total if total > 0 else 0.0
        print(
            f"  Epoch {epoch + 1} — train loss: {avg_train_loss:.4f} | "
            f"val loss: {avg_val_loss:.4f} | val acc (pos>neg): {val_acc:.4f}"
        )

        # Save best model by validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(save_dir))
            tokenizer.save_pretrained(str(save_dir))
            print(f"  ✓ Best model saved (val acc={best_val_acc:.4f})")

    print(f"  Training complete. Best val acc: {best_val_acc:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-idx",
        type=int,
        default=None,
        help="Train only config at this index (0-based).",
    )
    args = parser.parse_args()

    configs = config.RERANKER_CONFIGS
    indices = [args.config_idx] if args.config_idx is not None else range(len(configs))

    for i in indices:
        cfg = configs[i]
        print(f"\n{'=' * 60}")
        print(f"Config {i}: {cfg}")
        print(f"{'=' * 60}")
        train_one_config(cfg, i)

    print("\nAll done.")


if __name__ == "__main__":
    main()
