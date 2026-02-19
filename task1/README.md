# ClimateSense @ ClimateCheck 2026 - Task 1

[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/climatesense-project/climatecheck2026/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)

This folder contains the code for Task 1 of the [ClimateCheck 2026 competition](https://www.codabench.org/competitions/12213/), including claim–abstract retrieval, reranking, and verification. The code is organized into a pipeline of scripts that download and preprocess the data, perform BM25 retrieval, mine hard negatives, fine-tune reranker models, ensemble their outputs, verify with a local vLLM server, and generate the final submission.

## Requirements 📦

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) for environment management
- CUDA GPU for reranker training & vLLM verification

## Quick start 🚀

1. Clone the repository and navigate to the `task1` folder:

    ```bash
    git clone https://github.com/climatesense-project/climatecheck2026.git
    cd task1
    ```

2. Run the pipeline steps:

    ```bash
    # Step 0: Download HF dataset and publications corpus → `data/*.parquet`
    uv run -m task1.00_download_data

    # Step 1: Preprocess data, print stats, create splits → `data/*_split.parquet`
    uv run -m task1.01_preprocess_data

    # Step 2: BM25 retrieval → `artifacts/bm25_*.pkl`
    uv run -m task1.02_bm25_retrieval

    # Step 3: Hard negative mining + triplet construction → `artifacts/hard_negatives.pkl`,    `train_triplets.pkl`
    uv run -m task1.03_hard_negative_mining

    # Step 4: Fine-tune reranker(s) → `models/`
    uv run -m task1.04_fine_tune_reranker
    # Optional: Evaluate reranker(s) on validation set
    uv run -m task1.utils.evaluate_reranker models/reranker_*

    # Step 5: inference with reranker(s) → `artifacts/reranker_preds_*.csv`
    uv run -m task1.05_reranker_inference

    # Step 6: Ensemble reranker outputs (RRF) → `artifacts/ensemble_top{N}_*.csv`
    uv run -m task1.06_ensemble_reranking --split test

    # Step 7: Claims verifcation with local vLLM server → `artifacts/verification_vllm_*.csv`
    uv run -m task1.07_vllm_verification --top-n 5

    # Step 8: Generate final submission files → `submissions/predictions.csv` (+ `predictions.zip`)
    uv run -m task1.08_generate_submission
    ```

## Configuration 🔧

The file `task1/config.py` contains all the configurable parameters for the pipeline, such as dataset paths, settings, and hyperparameters. You can modify these parameters to experiment with different configurations or to adapt the pipeline to your local environment. The default values are set to reproduce the official challenge submission.

## Data & artifacts 📁

- `data/` - raw and split parquet files
- `artifacts/` - intermediate outputs and predictions
- `models/` - fine-tuned rerankers and any saved models
- `submissions/` - `predictions.csv` and `predictions.zip` for the competition submission

## vLLM local server

The script `07_vllm_verification.py` sends claim–abstract pairs to a locally hosted vLLM server for verification. To reproduce the official submission setup, start the vLLM server using the same model and parameters as specified below. Be sure to adjust GPU allocation and cache volume settings according to your hardware environment. Once running, the server should be accessible at: http://localhost:8000.

```bash
# adjust --gpus and -v <HF_CACHE> to your environment
docker run -d --gpus '"device=<GPU_ID>"' \
  -p 8000:8000 \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai \
  --gpu-memory-utilization 0.95 \
  --max-model-len 8192 \
  --max-num-seqs 128 \
  --dtype auto \
  --enforce-eager \
  --enable-chunked-prefill \
  --async-scheduling \
  openai/gpt-oss-120b
```
