#!/bin/bash
# Experiment: optimal_hypothesis_2
# Description: Medium chunks (1500) + k=10 + all features
# Generated: 2025-11-23T18:22:13.247652

set -e

cd "/Users/hansonxiong/Desktop/algoverse/shawheen rag"

echo "=========================================="
echo "Running: optimal_hypothesis_2"
echo "=========================================="

# Run the evaluation
python src/bulk_testing.py \
    --dataset financebench \
    --top-k 10 \
    --temperature 0.0 \
    --max-tokens 512 \
    --use-llm-judge \
    2>&1 | tee "/Users/hansonxiong/Desktop/algoverse/LLM_as_a_Judge/experiments/results/optimal_hypothesis_2.log"

# Copy results
cp bulk_runs/*.csv "/Users/hansonxiong/Desktop/algoverse/LLM_as_a_Judge/experiments/results/" 2>/dev/null || true
cp bulk_runs/*.json "/Users/hansonxiong/Desktop/algoverse/LLM_as_a_Judge/experiments/results/" 2>/dev/null || true

# Rename with experiment name
for f in "/Users/hansonxiong/Desktop/algoverse/LLM_as_a_Judge/experiments/results"/*.csv; do
    if [[ -f "$f" && ! "$f" == *"optimal_hypothesis_2"* ]]; then
        mv "$f" "/Users/hansonxiong/Desktop/algoverse/LLM_as_a_Judge/experiments/results/optimal_hypothesis_2_$(basename $f)"
    fi
done

echo "Completed: optimal_hypothesis_2"
