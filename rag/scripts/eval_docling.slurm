#!/bin/bash
#SBATCH --job-name=eval-docling
#SBATCH --partition=production
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --output=logs/eval_docling_%j.out
#SBATCH --error=logs/eval_docling_%j.err

# =============================================================================
# RAG Evaluation on Docling-ingested ChromaDB
# =============================================================================

echo "========================================"
echo "RAG Evaluation Starting"
echo "========================================"
echo "Date: $(date)"
echo "Node: $(hostname)"

cd /data/junjiexiong/algoverse/rag

# Activate virtual environment
source .venv/bin/activate

# Set cache directories
export HF_HOME="/data/junjiexiong/algoverse/rag/.cache/huggingface"
export TRANSFORMERS_CACHE="/data/junjiexiong/algoverse/rag/.cache/transformers"
export XDG_CACHE_HOME="/data/junjiexiong/algoverse/rag/.cache"

# Load API keys from .env if exists
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
    echo "Loaded API keys from .env"
fi

# Check ChromaDB status (using SQLite to avoid HNSW segfaults)
echo ""
echo "Checking ChromaDB status..."
python -c "
import sqlite3
conn = sqlite3.connect('chroma_docling/chroma.sqlite3')
c = conn.cursor()
c.execute('SELECT COUNT(*) FROM embeddings')
chunks = c.fetchone()[0]
c.execute(\"SELECT COUNT(DISTINCT string_value) FROM embedding_metadata WHERE key='source_file'\")
pdfs = c.fetchone()[0]
print(f'Unique PDFs: {pdfs}')
print(f'Total chunks: {chunks}')
conn.close()
"

# Point to Docling-ingested database
# Backup existing chroma if it exists, then symlink
if [ -d "chroma" ] && [ ! -L "chroma" ]; then
    mv chroma chroma_backup_$(date +%s)
fi
ln -sf chroma_docling chroma
echo "Using ChromaDB: chroma -> chroma_docling"

# Run evaluation
echo ""
echo "Running RAG evaluation..."
python src/bulk_testing.py \
    --dataset financebench \
    --model gpt-4o-mini \
    --pipeline hybrid_filter_rerank \
    --top-k 5 \
    --use-numeric-verify

echo ""
echo "========================================"
echo "Evaluation Complete: $(date)"
echo "========================================"
