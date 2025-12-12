#!/bin/bash
# Priority Experiment Runner
# Runs the most important experiments first for quick publication results
# Total: 12 high-priority experiments (vs 25 total)

set -e

OUTPUT_DIR="$(dirname "$0")"
SCRIPTS_DIR="$OUTPUT_DIR/scripts"

echo "=========================================="
echo "Priority Experiment Suite"
echo "12 high-priority experiments"
echo "=========================================="

# Track progress
COMPLETED=0
FAILED=0
TOTAL=12

run_experiment() {
    local name=$1
    local num=$2

    echo ""
    echo "[$num/$TOTAL] Running: $name"

    if bash "$SCRIPTS_DIR/$name.sh"; then
        COMPLETED=$((COMPLETED + 1))
        echo "✓ Completed: $name"
    else
        FAILED=$((FAILED + 1))
        echo "✗ Failed: $name"
    fi
}

# ==========================================
# PRIORITY 1: Core Ablation Study (5 experiments)
# ==========================================
echo ""
echo "=== PRIORITY 1: Ablation Study ==="

run_experiment "baseline_semantic_only" 1
run_experiment "ablation_hybrid" 2
run_experiment "ablation_metadata" 3
run_experiment "ablation_reranking" 4
run_experiment "ablation_all_features" 5

# ==========================================
# PRIORITY 2: K-Value Extremes (3 experiments)
# ==========================================
echo ""
echo "=== PRIORITY 2: K-Value Analysis ==="

run_experiment "k_value_5" 6
run_experiment "k_value_10" 7
run_experiment "k_value_20" 8

# ==========================================
# PRIORITY 3: Chunk Size Extremes (3 experiments)
# ==========================================
echo ""
echo "=== PRIORITY 3: Chunk Size Analysis ==="

run_experiment "chunk_element_1000" 9
run_experiment "chunk_element_2000" 10
run_experiment "chunk_element_3000" 11

# ==========================================
# PRIORITY 4: Best Hypothesis (1 experiment)
# ==========================================
echo ""
echo "=== PRIORITY 4: Optimal Configuration ==="

run_experiment "optimal_hypothesis_1" 12

# ==========================================
# Summary
# ==========================================
echo ""
echo "=========================================="
echo "Priority Suite Complete"
echo "=========================================="
echo "Completed: $COMPLETED / $TOTAL"
echo "Failed: $FAILED"
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Next steps:"
echo "  1. Open analysis.ipynb to generate figures"
echo "  2. Run 'bash run_all.sh' for remaining experiments"
echo "=========================================="
