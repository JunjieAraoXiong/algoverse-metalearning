#!/bin/bash
# Experiment: chunk_standard_2000
# Description: Standard character chunking - 2000 chars
# Generated: 2025-11-23T18:22:13.246486

set -e

cd "/Users/hansonxiong/Desktop/algoverse/shawheen rag"

echo "=========================================="
echo "Running: chunk_standard_2000"
echo "=========================================="

# Run the evaluation
python src/bulk_testing.py \
    --dataset financebench \
    --top-k 10 \
    --temperature 0.0 \
    --max-tokens 512 \
    --use-llm-judge \
    2>&1 | tee "/Users/hansonxiong/Desktop/algoverse/LLM_as_a_Judge/experiments/results/chunk_standard_2000.log"

# Copy results
cp bulk_runs/*.csv "/Users/hansonxiong/Desktop/algoverse/LLM_as_a_Judge/experiments/results/" 2>/dev/null || true
cp bulk_runs/*.json "/Users/hansonxiong/Desktop/algoverse/LLM_as_a_Judge/experiments/results/" 2>/dev/null || true

# Rename with experiment name
for f in "/Users/hansonxiong/Desktop/algoverse/LLM_as_a_Judge/experiments/results"/*.csv; do
    if [[ -f "$f" && ! "$f" == *"chunk_standard_2000"* ]]; then
        mv "$f" "/Users/hansonxiong/Desktop/algoverse/LLM_as_a_Judge/experiments/results/chunk_standard_2000_$(basename $f)"
    fi
done

echo "Completed: chunk_standard_2000"
