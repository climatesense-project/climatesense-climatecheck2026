"""Configuration file for ClimateCheck 2026 Task 1 pipeline."""

from __future__ import annotations

import os
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = PROJECT_ROOT / "models"
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"

for d in (DATA_DIR, ARTIFACTS_DIR, MODELS_DIR, SUBMISSIONS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# HuggingFace dataset identifiers
HF_DATASET = "rabuahmad/climatecheck"
HF_CORPUS = "rabuahmad/climatecheck_publications_corpus"

# Local data files
CORPUS_PATH = DATA_DIR / "corpus.parquet"
TRAIN_PATH = DATA_DIR / "train.parquet"
TEST_PATH = DATA_DIR / "test.parquet"
TRAIN_SPLIT_PATH = DATA_DIR / "train_split.parquet"
VAL_SPLIT_PATH = DATA_DIR / "val_split.parquet"

# ── BM25 ─────────────────────────────────────────────────────────────────────
BM25_TOP_N = 5000
BM25_TRAIN_PKL = ARTIFACTS_DIR / "bm25_train.pkl"
BM25_TEST_PKL = ARTIFACTS_DIR / "bm25_test.pkl"

# ── Hard-negative mining ─────────────────────────────────────────────────────
HARD_NEG_RERANKER_MODEL = "BAAI/bge-reranker-large"
HARD_NEG_TOP_K = 1000
HARD_NEG_SELECT = 500
NUM_RANDOM_NEG = 10
NUM_HARD_NEG = 5
HARD_NEG_PKL = ARTIFACTS_DIR / "hard_negatives.pkl"
TRAIN_TRIPLETS_PKL = ARTIFACTS_DIR / "train_triplets.pkl"

# ── Reranker fine-tuning configs ─────────────────────────────────────────────
# Each config produces a separate model for ensemble.
RERANKER_CONFIGS = [
    {
        "base_model": "BAAI/bge-reranker-large",
        "batch_size": 8,
        "margin": 0.20,
        "lr": 1e-5,
    },
    {
        "base_model": "BAAI/bge-reranker-large",
        "batch_size": 16,
        "margin": 0.20,
        "lr": 1e-5,
    },
    {
        "base_model": "BAAI/bge-reranker-large",
        "batch_size": 8,
        "margin": 0.25,
        "lr": 1e-5,
    },
    {
        "base_model": "BAAI/bge-reranker-large",
        "batch_size": 16,
        "margin": 0.25,
        "lr": 1e-5,
    },
    {
        "base_model": "BAAI/bge-reranker-v2-m3",
        "batch_size": 16,
        "margin": 0.20,
        "lr": 1e-5,
    },
]
RERANKER_MAX_LENGTH = 512
RERANKER_EPOCHS = 1
RERANKER_INFERENCE_TOP_N = 500  # keep top-N per claim from each reranker

# ── Ensemble ─────────────────────────────────────────────────────────────────
RRF_K = 6
ENSEMBLE_TOP_N = 10

# ── vLLM (local server) ──────────────────────────────────────────────────────
VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000")
VLLM_MODEL = os.environ.get("VLLM_MODEL", "openai/gpt-oss-120b")

# ── Validation split ─────────────────────────────────────────────────────────
VAL_FRACTION = 0.15
RANDOM_SEED = 42
