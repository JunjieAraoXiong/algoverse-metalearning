#!/bin/bash
#SBATCH --job-name=rag-eval
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err

# RAG Evaluation Job for Together AI Cluster
# Usage: sbatch scripts/eval_job.sh [--model MODEL] [--pipeline PIPELINE]

set -e

# Parse command line arguments
MODEL="${MODEL:-meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo}"
PIPELINE="${PIPELINE:-hybrid_filter_rerank}"
TOP_K="${TOP_K:-10}"
USE_LLM_JUDGE="${USE_LLM_JUDGE:-false}"

# Print job info
echo "========================================"
echo "RAG Evaluation Job"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Model: $MODEL"
echo "Pipeline: $PIPELINE"
echo "Top-K: $TOP_K"
echo "LLM Judge: $USE_LLM_JUDGE"
echo "========================================"

# Setup environment
cd /data/junjiexiong/algoverse/rag

# Activate virtual environment
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
else
    echo "Warning: Virtual environment not found at .venv/"
    echo "Creating new virtual environment..."
    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
fi

# Set cache directories
export HF_HOME=/data/junjiexiong/.cache/huggingface
export NLTK_DATA=/data/junjiexiong/.cache/nltk_data
export MPLCONFIGDIR=/data/junjiexiong/.cache/matplotlib

# Create logs directory if it doesn't exist
mkdir -p logs

# Verify ChromaDB exists
if [ ! -d "chroma" ]; then
    echo "ERROR: ChromaDB not found at ./chroma/"
    echo "Please upload the ChromaDB first."
    exit 1
fi

# Build command
CMD="python src/bulk_testing.py"
CMD="$CMD --model $MODEL"
CMD="$CMD --pipeline $PIPELINE"
CMD="$CMD --top-k $TOP_K"
CMD="$CMD --dataset financebench"

if [ "$USE_LLM_JUDGE" = "true" ]; then
    CMD="$CMD --use-llm-judge"
fi

# Run evaluation
echo ""
echo "Running: $CMD"
echo ""

$CMD

echo ""
echo "========================================"
echo "Evaluation complete!"
echo "Results saved to: bulk_runs/"
echo "========================================"
