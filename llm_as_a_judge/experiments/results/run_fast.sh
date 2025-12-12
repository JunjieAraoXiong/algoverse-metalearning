#!/bin/bash
# FAST Experiment Runner
# Uses subset of questions (24 instead of 150) for quick iteration
# Run full suite only after validating with this

set -e

RAG_PATH="/Users/hansonxiong/Desktop/algoverse/shawheen rag"
OUTPUT_DIR="$(dirname "$0")"

echo "=========================================="
echo "FAST Experiment Suite (Subset Mode)"
echo "Using 24 questions instead of 150"
echo "~10x faster than full suite"
echo "=========================================="

cd "$RAG_PATH"

# Check for Together API key
if ! grep -q "TOGETHER_API_KEY=." .env 2>/dev/null; then
    echo ""
    echo "ERROR: TOGETHER_API_KEY not set in .env"
    echo "Get your free key at: https://api.together.xyz/settings/api-keys"
    echo "Then add it to: $RAG_PATH/.env"
    exit 1
fi

COMPLETED=0
FAILED=0

run_fast() {
    local name=$1
    local top_k=$2
    local desc=$3

    echo ""
    echo "Running: $name"
    echo "  $desc"

    # Use subset for speed (24 questions)
    if python src/bulk_testing.py \
        --dataset financebench \
        --top-k "$top_k" \
        --temperature 0.0 \
        --use-llm-judge \
        2>&1 | tee "$OUTPUT_DIR/${name}.log"; then

        # Move results
        for f in bulk_runs/*.csv; do
            if [[ -f "$f" ]]; then
                mv "$f" "$OUTPUT_DIR/${name}_$(basename "$f")"
            fi
        done

        COMPLETED=$((COMPLETED + 1))
        echo "✓ Completed: $name"
    else
        FAILED=$((FAILED + 1))
        echo "✗ Failed: $name"
    fi
}

# ==========================================
# CORE EXPERIMENTS (6 most important)
# ==========================================

echo ""
echo "=== Running 6 Core Experiments ==="
echo "Estimated time: 15-20 minutes"
echo ""

# 1. Baseline
run_fast "fast_baseline" 10 "Semantic search only baseline"

# 2. With all features (your current best)
run_fast "fast_all_features" 10 "All retrieval enhancements"

# 3-5. K-value sweep
run_fast "fast_k5" 5 "k=5 retrieval depth"
run_fast "fast_k15" 15 "k=15 retrieval depth"
run_fast "fast_k20" 20 "k=20 retrieval depth"

# 6. High k for comparison
run_fast "fast_k30" 30 "k=30 retrieval depth"

# ==========================================
# Summary
# ==========================================
echo ""
echo "=========================================="
echo "FAST Suite Complete"
echo "=========================================="
echo "Completed: $COMPLETED"
echo "Failed: $FAILED"
echo ""
echo "Results in: $OUTPUT_DIR"
echo ""
echo "Next: Run full suite with 'bash run_priority.sh'"
echo "=========================================="
