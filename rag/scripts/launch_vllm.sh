#!/bin/bash
#SBATCH --job-name=vllm-server
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=logs/vllm_%j.out
#SBATCH --error=logs/vllm_%j.err

# vLLM Server for RAG Inference on Together AI Cluster
# Usage: sbatch scripts/launch_vllm.sh
#
# This launches an OpenAI-compatible vLLM server running Llama-3.1-70B.
# Once running, the RAG system auto-routes to it via the local-vllm provider.

set -e

# Configuration
MODEL="${MODEL:-meta-llama/Meta-Llama-3.1-70B-Instruct}"
PORT="${PORT:-8000}"
NUM_GPUS=8

# Print job info
echo "========================================"
echo "vLLM Server Launch"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Model: $MODEL"
echo "Port: $PORT"
echo "Tensor Parallel Size: $NUM_GPUS"
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

echo ""
echo "Starting vLLM server..."
echo "Server will be available at: http://localhost:$PORT/v1"
echo ""

# Run vLLM OpenAI-compatible server
python -m vllm.entrypoints.openai.api_server \
    --model $MODEL \
    --tensor-parallel-size $NUM_GPUS \
    --gpu-memory-utilization 0.95 \
    --attention-backend FLASH_ATTN \
    --port $PORT \
    --host 0.0.0.0
