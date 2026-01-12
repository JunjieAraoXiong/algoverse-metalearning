#!/bin/bash
#SBATCH --job-name=docling-p%a
#SBATCH --partition=production
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --output=logs/ingest_parallel_%A_%a.out
#SBATCH --error=logs/ingest_parallel_%A_%a.err
#SBATCH --array=0-7

# =============================================================================
# Parallel Docling Ingestion - Splits 265 PDFs across 4 GPUs
# =============================================================================
# Usage: sbatch scripts/ingest_parallel.slurm
# =============================================================================

echo "========================================"
echo "Parallel Docling Ingestion - Worker $SLURM_ARRAY_TASK_ID"
echo "========================================"
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo ""

cd /data/junjiexiong/algoverse/rag
source .venv/bin/activate

# Set cache directories
export HF_HOME="/data/junjiexiong/algoverse/rag/.cache/huggingface"
export TRANSFORMERS_CACHE="/data/junjiexiong/algoverse/rag/.cache/transformers"
export TORCH_HOME="/data/junjiexiong/algoverse/rag/.cache/torch"
export XDG_CACHE_HOME="/data/junjiexiong/algoverse/rag/.cache"
# Each worker uses a different GPU based on array task ID
export CUDA_VISIBLE_DEVICES=$SLURM_ARRAY_TASK_ID
export TOKENIZERS_PARALLELISM=false
echo "Using GPU: $CUDA_VISIBLE_DEVICES"

# Calculate file ranges for 4 workers (265 files total, ~66 each)
# Worker 0: files 0-65
# Worker 1: files 66-131
# Worker 2: files 132-197
# Worker 3: files 198-264
TOTAL_FILES=265
NUM_WORKERS=8
FILES_PER_WORKER=$((TOTAL_FILES / NUM_WORKERS))

START_IDX=$((SLURM_ARRAY_TASK_ID * FILES_PER_WORKER))
if [ $SLURM_ARRAY_TASK_ID -eq $((NUM_WORKERS - 1)) ]; then
    # Last worker takes all remaining files
    END_IDX=$TOTAL_FILES
else
    END_IDX=$((START_IDX + FILES_PER_WORKER))
fi

echo "Processing files [$START_IDX:$END_IDX]"
echo ""

PDF_DIR="data/test_files/finance-bench-pdfs"

python src/ingest_docling.py \
    --input-dir "$PDF_DIR" \
    --output-dir chroma_docling \
    --chunk-size 2500 \
    --batch-size 5 \
    --start $START_IDX \
    --end $END_IDX

echo ""
echo "========================================"
echo "Worker $SLURM_ARRAY_TASK_ID Complete: $(date)"
echo "========================================"
