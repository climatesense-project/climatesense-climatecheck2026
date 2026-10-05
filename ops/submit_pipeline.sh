#!/bin/bash
set -e
mkdir -p logs

BASIC_ARGS="-p l40s -N 1 -n 8 --mem=16G -q normal --parsable"
GPU_ARGS="-N 1 -n 8 --mem=16G --parsable"

echo "Submitting pipeline..."

#jid0=$(sbatch $BASIC_ARGS --job-name=cs_00 --output=logs/00_%j.out --error=logs/00_%j.err \
#  --wrap="uv run -m task1.00_download_data")

#jid1=$(sbatch $BASIC_ARGS --job-name=cs_01 --output=logs/01_%j.out --error=logs/01_%j.err \
#  --dependency=afterok:$jid0 \
#  --wrap="uv run -m task1.01_preprocess_data")

#jid2=$(sbatch $BASIC_ARGS --job-name=cs_02 --output=logs/02_%j.out --error=logs/02_%j.err \
#  --dependency=afterok:$jid1 \
#  --wrap="uv run -m task1.02_bm25_retrieval")

#jid3=$(sbatch $GPU_ARGS -p l40s -q long --gpus 1 --job-name=cs_03 --output=logs/03_%j.out --error=logs/03_%j.err \
#  --dependency=afterok:$jid2 \
#  --wrap="uv run -m task1.03_hard_negative_mining")

#jid4=$(sbatch $GPU_ARGS -p h200 -q backfill --gpus h200_4g.71gb:2 --job-name=cs_04 --output=logs/04_%j.out --error=logs/04_%j.err \
#  --dependency=afterok:$jid3 \
#  --wrap="uv run -m task1.04_fine_tune_reranker")

#jid5=$(sbatch $GPU_ARGS -p h200 -q backfill --gpus h200_4g.71gb:2 --job-name=cs_05 --output=logs/05_%j.out --error=logs/05_%j.err \
#  --dependency=afterok:$jid4 \
#  --wrap="uv run -m task1.05_reranker_inference")

#jid6=$(sbatch $GPU_ARGS -p h200 -q backfill --gpus h200_4g.71gb:2 --job-name=cs_06 --output=logs/06_%j.out --error=logs/06_%j.err \
#  --dependency=afterok:$jid5 \
#  --wrap="uv run -m task1.06_ensemble_reranking --split test")

jid8=$(sbatch $BASIC_ARGS --job-name=cs_08 --output=logs/08_%j.out --error=logs/08_%j.err \
  --wrap="uv run -m task1.08_generate_submission")

echo ""
echo "✅ Pipeline submitted successfully!"
echo ""
echo "Monitor with: watch -n 30 'squeue -u \$USER -o \"%.10i %.12j %.8T %.20E\"'"
