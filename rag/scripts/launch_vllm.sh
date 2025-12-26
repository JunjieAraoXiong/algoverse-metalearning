#!/bin/bash

# Configuration
MODEL="meta-llama/Meta-Llama-3.1-70B-Instruct"
PORT=8000
NUM_GPUS=8

echo "Starting vLLM server with model: $MODEL on $NUM_GPUS GPUs..."

# Ensure we are in the virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Warning: No virtual environment detected. Activating venv..."
    source venv/bin/activate
fi

# Run vLLM
python -m vllm.entrypoints.openai.api_server \
    --model $MODEL \
    --tensor-parallel-size $NUM_GPUS \
    --gpu-memory-utilization 0.95 \
    --attention-backend FLASH_ATTN \
    --port $PORT \
    --host 0.0.0.0
