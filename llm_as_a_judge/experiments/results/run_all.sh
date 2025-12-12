#!/bin/bash
# Master Experiment Runner
# Generated: 2025-11-23T18:22:13.247735
# Total experiments: 25

set -e

OUTPUT_DIR="/Users/hansonxiong/Desktop/algoverse/LLM_as_a_Judge/experiments/results"
mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "RAG Experiment Suite - Publication Quality"
echo "Total experiments: 25"
echo "=========================================="

# Track progress
COMPLETED=0
FAILED=0


echo ""
echo "[1/25] Starting: baseline_semantic_only"
echo "Description: Pure semantic search baseline - no enhancements"

if bash "/Users/hansonxiong/Desktop/algoverse/LLM_as_a_Judge/experiments/results/scripts/baseline_semantic_only.sh"; then
    COMPLETED=$((COMPLETED + 1))
    echo "✓ Completed: baseline_semantic_only"
else
    FAILED=$((FAILED + 1))
    echo "✗ Failed: baseline_semantic_only"
fi


echo ""
echo "[2/25] Starting: ablation_hybrid"
echo "Description: Baseline + BM25 hybrid search"

if bash "/Users/hansonxiong/Desktop/algoverse/LLM_as_a_Judge/experiments/results/scripts/ablation_hybrid.sh"; then
    COMPLETED=$((COMPLETED + 1))
    echo "✓ Completed: ablation_hybrid"
else
    FAILED=$((FAILED + 1))
    echo "✗ Failed: ablation_hybrid"
fi


echo ""
echo "[3/25] Starting: ablation_metadata"
echo "Description: Baseline + metadata filtering"

if bash "/Users/hansonxiong/Desktop/algoverse/LLM_as_a_Judge/experiments/results/scripts/ablation_metadata.sh"; then
    COMPLETED=$((COMPLETED + 1))
    echo "✓ Completed: ablation_metadata"
else
    FAILED=$((FAILED + 1))
    echo "✗ Failed: ablation_metadata"
fi


echo ""
echo "[4/25] Starting: ablation_reranking"
echo "Description: Baseline + cross-encoder reranking"

if bash "/Users/hansonxiong/Desktop/algoverse/LLM_as_a_Judge/experiments/results/scripts/ablation_reranking.sh"; then
    COMPLETED=$((COMPLETED + 1))
    echo "✓ Completed: ablation_reranking"
else
    FAILED=$((FAILED + 1))
    echo "✗ Failed: ablation_reranking"
fi


echo ""
echo "[5/25] Starting: ablation_all_features"
echo "Description: All retrieval enhancements enabled"

if bash "/Users/hansonxiong/Desktop/algoverse/LLM_as_a_Judge/experiments/results/scripts/ablation_all_features.sh"; then
    COMPLETED=$((COMPLETED + 1))
    echo "✓ Completed: ablation_all_features"
else
    FAILED=$((FAILED + 1))
    echo "✗ Failed: ablation_all_features"
fi


echo ""
echo "[6/25] Starting: chunk_standard_500"
echo "Description: Standard character chunking - 500 chars"

if bash "/Users/hansonxiong/Desktop/algoverse/LLM_as_a_Judge/experiments/results/scripts/chunk_standard_500.sh"; then
    COMPLETED=$((COMPLETED + 1))
    echo "✓ Completed: chunk_standard_500"
else
    FAILED=$((FAILED + 1))
    echo "✗ Failed: chunk_standard_500"
fi


echo ""
echo "[7/25] Starting: chunk_standard_1000"
echo "Description: Standard character chunking - 1000 chars"

if bash "/Users/hansonxiong/Desktop/algoverse/LLM_as_a_Judge/experiments/results/scripts/chunk_standard_1000.sh"; then
    COMPLETED=$((COMPLETED + 1))
    echo "✓ Completed: chunk_standard_1000"
else
    FAILED=$((FAILED + 1))
    echo "✗ Failed: chunk_standard_1000"
fi


echo ""
echo "[8/25] Starting: chunk_element_1000"
echo "Description: Element-based chunking - 1000 max chars"

if bash "/Users/hansonxiong/Desktop/algoverse/LLM_as_a_Judge/experiments/results/scripts/chunk_element_1000.sh"; then
    COMPLETED=$((COMPLETED + 1))
    echo "✓ Completed: chunk_element_1000"
else
    FAILED=$((FAILED + 1))
    echo "✗ Failed: chunk_element_1000"
fi


echo ""
echo "[9/25] Starting: chunk_standard_1500"
echo "Description: Standard character chunking - 1500 chars"

if bash "/Users/hansonxiong/Desktop/algoverse/LLM_as_a_Judge/experiments/results/scripts/chunk_standard_1500.sh"; then
    COMPLETED=$((COMPLETED + 1))
    echo "✓ Completed: chunk_standard_1500"
else
    FAILED=$((FAILED + 1))
    echo "✗ Failed: chunk_standard_1500"
fi


echo ""
echo "[10/25] Starting: chunk_element_1500"
echo "Description: Element-based chunking - 1500 max chars"

if bash "/Users/hansonxiong/Desktop/algoverse/LLM_as_a_Judge/experiments/results/scripts/chunk_element_1500.sh"; then
    COMPLETED=$((COMPLETED + 1))
    echo "✓ Completed: chunk_element_1500"
else
    FAILED=$((FAILED + 1))
    echo "✗ Failed: chunk_element_1500"
fi


echo ""
echo "[11/25] Starting: chunk_standard_2000"
echo "Description: Standard character chunking - 2000 chars"

if bash "/Users/hansonxiong/Desktop/algoverse/LLM_as_a_Judge/experiments/results/scripts/chunk_standard_2000.sh"; then
    COMPLETED=$((COMPLETED + 1))
    echo "✓ Completed: chunk_standard_2000"
else
    FAILED=$((FAILED + 1))
    echo "✗ Failed: chunk_standard_2000"
fi


echo ""
echo "[12/25] Starting: chunk_element_2000"
echo "Description: Element-based chunking - 2000 max chars"

if bash "/Users/hansonxiong/Desktop/algoverse/LLM_as_a_Judge/experiments/results/scripts/chunk_element_2000.sh"; then
    COMPLETED=$((COMPLETED + 1))
    echo "✓ Completed: chunk_element_2000"
else
    FAILED=$((FAILED + 1))
    echo "✗ Failed: chunk_element_2000"
fi


echo ""
echo "[13/25] Starting: chunk_standard_3000"
echo "Description: Standard character chunking - 3000 chars"

if bash "/Users/hansonxiong/Desktop/algoverse/LLM_as_a_Judge/experiments/results/scripts/chunk_standard_3000.sh"; then
    COMPLETED=$((COMPLETED + 1))
    echo "✓ Completed: chunk_standard_3000"
else
    FAILED=$((FAILED + 1))
    echo "✗ Failed: chunk_standard_3000"
fi


echo ""
echo "[14/25] Starting: chunk_element_3000"
echo "Description: Element-based chunking - 3000 max chars"

if bash "/Users/hansonxiong/Desktop/algoverse/LLM_as_a_Judge/experiments/results/scripts/chunk_element_3000.sh"; then
    COMPLETED=$((COMPLETED + 1))
    echo "✓ Completed: chunk_element_3000"
else
    FAILED=$((FAILED + 1))
    echo "✗ Failed: chunk_element_3000"
fi


echo ""
echo "[15/25] Starting: k_value_3"
echo "Description: Top-k retrieval sensitivity - k=3"

if bash "/Users/hansonxiong/Desktop/algoverse/LLM_as_a_Judge/experiments/results/scripts/k_value_3.sh"; then
    COMPLETED=$((COMPLETED + 1))
    echo "✓ Completed: k_value_3"
else
    FAILED=$((FAILED + 1))
    echo "✗ Failed: k_value_3"
fi


echo ""
echo "[16/25] Starting: k_value_5"
echo "Description: Top-k retrieval sensitivity - k=5"

if bash "/Users/hansonxiong/Desktop/algoverse/LLM_as_a_Judge/experiments/results/scripts/k_value_5.sh"; then
    COMPLETED=$((COMPLETED + 1))
    echo "✓ Completed: k_value_5"
else
    FAILED=$((FAILED + 1))
    echo "✗ Failed: k_value_5"
fi


echo ""
echo "[17/25] Starting: k_value_10"
echo "Description: Top-k retrieval sensitivity - k=10"

if bash "/Users/hansonxiong/Desktop/algoverse/LLM_as_a_Judge/experiments/results/scripts/k_value_10.sh"; then
    COMPLETED=$((COMPLETED + 1))
    echo "✓ Completed: k_value_10"
else
    FAILED=$((FAILED + 1))
    echo "✗ Failed: k_value_10"
fi


echo ""
echo "[18/25] Starting: k_value_15"
echo "Description: Top-k retrieval sensitivity - k=15"

if bash "/Users/hansonxiong/Desktop/algoverse/LLM_as_a_Judge/experiments/results/scripts/k_value_15.sh"; then
    COMPLETED=$((COMPLETED + 1))
    echo "✓ Completed: k_value_15"
else
    FAILED=$((FAILED + 1))
    echo "✗ Failed: k_value_15"
fi


echo ""
echo "[19/25] Starting: k_value_20"
echo "Description: Top-k retrieval sensitivity - k=20"

if bash "/Users/hansonxiong/Desktop/algoverse/LLM_as_a_Judge/experiments/results/scripts/k_value_20.sh"; then
    COMPLETED=$((COMPLETED + 1))
    echo "✓ Completed: k_value_20"
else
    FAILED=$((FAILED + 1))
    echo "✗ Failed: k_value_20"
fi


echo ""
echo "[20/25] Starting: k_value_30"
echo "Description: Top-k retrieval sensitivity - k=30"

if bash "/Users/hansonxiong/Desktop/algoverse/LLM_as_a_Judge/experiments/results/scripts/k_value_30.sh"; then
    COMPLETED=$((COMPLETED + 1))
    echo "✓ Completed: k_value_30"
else
    FAILED=$((FAILED + 1))
    echo "✗ Failed: k_value_30"
fi


echo ""
echo "[21/25] Starting: embedding_small"
echo "Description: OpenAI text-embedding-3-small"

if bash "/Users/hansonxiong/Desktop/algoverse/LLM_as_a_Judge/experiments/results/scripts/embedding_small.sh"; then
    COMPLETED=$((COMPLETED + 1))
    echo "✓ Completed: embedding_small"
else
    FAILED=$((FAILED + 1))
    echo "✗ Failed: embedding_small"
fi


echo ""
echo "[22/25] Starting: embedding_large"
echo "Description: OpenAI text-embedding-3-large"

if bash "/Users/hansonxiong/Desktop/algoverse/LLM_as_a_Judge/experiments/results/scripts/embedding_large.sh"; then
    COMPLETED=$((COMPLETED + 1))
    echo "✓ Completed: embedding_large"
else
    FAILED=$((FAILED + 1))
    echo "✗ Failed: embedding_large"
fi


echo ""
echo "[23/25] Starting: prompt_forced_answer"
echo "Description: Forced answer prompting strategy"

if bash "/Users/hansonxiong/Desktop/algoverse/LLM_as_a_Judge/experiments/results/scripts/prompt_forced_answer.sh"; then
    COMPLETED=$((COMPLETED + 1))
    echo "✓ Completed: prompt_forced_answer"
else
    FAILED=$((FAILED + 1))
    echo "✗ Failed: prompt_forced_answer"
fi


echo ""
echo "[24/25] Starting: optimal_hypothesis_1"
echo "Description: Large chunks (3000) + k=20 + all features"

if bash "/Users/hansonxiong/Desktop/algoverse/LLM_as_a_Judge/experiments/results/scripts/optimal_hypothesis_1.sh"; then
    COMPLETED=$((COMPLETED + 1))
    echo "✓ Completed: optimal_hypothesis_1"
else
    FAILED=$((FAILED + 1))
    echo "✗ Failed: optimal_hypothesis_1"
fi


echo ""
echo "[25/25] Starting: optimal_hypothesis_2"
echo "Description: Medium chunks (1500) + k=10 + all features"

if bash "/Users/hansonxiong/Desktop/algoverse/LLM_as_a_Judge/experiments/results/scripts/optimal_hypothesis_2.sh"; then
    COMPLETED=$((COMPLETED + 1))
    echo "✓ Completed: optimal_hypothesis_2"
else
    FAILED=$((FAILED + 1))
    echo "✗ Failed: optimal_hypothesis_2"
fi


echo ""
echo "=========================================="
echo "Experiment Suite Complete"
echo "Completed: $COMPLETED"
echo "Failed: $FAILED"
echo "=========================================="
