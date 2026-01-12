#!/bin/bash
# Full evaluation suite for Together AI and GPT-4o-mini
# Records all results to evaluation/together_api_results.json

set -e
cd /Users/hansonxiong/Desktop/algoverse/rag

# Load API keys
source .env

# Models to test
LLAMA="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"
GPT="gpt-4o-mini"

# Pipelines and k values
PIPELINES=("semantic" "hybrid" "hybrid_filter_rerank")
K_VALUES=(5 10)

echo "============================================"
echo "Full RAG Evaluation Suite"
echo "============================================"

# Function to run test and extract results
run_test() {
    local model=$1
    local pipeline=$2
    local k=$3

    echo ""
    echo ">>> Testing: $model | $pipeline | k=$k"
    echo "============================================"

    python src/bulk_testing.py \
        --dataset financebench \
        --model "$model" \
        --pipeline "$pipeline" \
        --top-k "$k"
}

# Run all Llama 70B tests
echo ""
echo "========== LLAMA 70B TESTS =========="
for pipeline in "${PIPELINES[@]}"; do
    for k in "${K_VALUES[@]}"; do
        run_test "$LLAMA" "$pipeline" "$k"
    done
done

# Run all GPT-4o-mini tests (for comparison)
echo ""
echo "========== GPT-4O-MINI TESTS =========="
for pipeline in "${PIPELINES[@]}"; do
    for k in "${K_VALUES[@]}"; do
        run_test "$GPT" "$pipeline" "$k"
    done
done

echo ""
echo "============================================"
echo "All tests complete!"
echo "Results saved in: bulk_runs/"
echo "============================================"
