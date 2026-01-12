#!/bin/bash
# Run GPT-5.2 test on the locally-ingested ChromaDB
# Use this after ingestion completes

set -e

echo "========================================"
echo "GPT-5.2 FinanceBench Test"
echo "========================================"

# Check ChromaDB status
python -c "
import chromadb
client = chromadb.PersistentClient(path='chroma_financebench')
col = client.get_collection('langchain')
print(f'ChromaDB chunks: {col.count()}')
"

echo ""
echo "Starting evaluation..."

# Run with the new ChromaDB
python src/bulk_testing.py \
    --dataset financebench \
    --model gpt-5.2 \
    --pipeline hybrid_filter_rerank \
    --top-k 10 \
    --chroma-path chroma_financebench

echo ""
echo "Results saved to bulk_runs/"
echo "Check the latest JSON file for metrics."
