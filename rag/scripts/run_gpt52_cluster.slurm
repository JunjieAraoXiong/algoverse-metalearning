#!/bin/bash
#SBATCH --job-name=gpt52-eval
#SBATCH --output=logs/gpt52_eval_%j.out
#SBATCH --error=logs/gpt52_eval_%j.err
#SBATCH --time=4:00:00
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

# CRITICAL: Fix HOME directory
export HOME="/data/junjiexiong"
export HF_HOME="/data/junjiexiong/.cache/huggingface"
export TRANSFORMERS_CACHE="/data/junjiexiong/.cache/transformers"

cd /data/junjiexiong/algoverse/rag

# Pull latest changes (including metadata filter fix)
git pull origin features/metalearning

# Activate conda environment
source /data/junjiexiong/miniconda3/etc/profile.d/conda.sh
conda activate rag

# Create logs directory if needed
mkdir -p logs

echo "========================================"
echo "GPT-5.2 Evaluation on Docling ChromaDB"
echo "========================================"
echo "Started: $(date)"
echo "ChromaDB: /data/junjiexiong/algoverse/rag/chroma_docling"

# Run evaluation with GPT-5.2
python src/bulk_testing.py \
    --dataset financebench \
    --model gpt-5.2 \
    --pipeline hybrid_filter_rerank \
    --top-k 10 \
    --chroma-path /data/junjiexiong/algoverse/rag/chroma_docling

echo "========================================"
echo "Finished: $(date)"
echo "Results in: bulk_runs/"
echo "========================================"
