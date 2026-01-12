#!/bin/bash
#SBATCH --job-name=docling-ingest
#SBATCH --partition=production
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/ingest_docling_%j.out
#SBATCH --error=logs/ingest_docling_%j.err

# =============================================================================
# Docling PDF Ingestion Job
# =============================================================================
# Processes FinanceBench PDFs with 97.9% table accuracy using TableFormer
#
# Usage:
#   sbatch scripts/ingest_docling.slurm
#
# Monitor:
#   squeue -u $USER
#   tail -f logs/ingest_docling_*.out
# =============================================================================

echo "========================================"
echo "Docling Ingestion Job Starting"
echo "========================================"
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'No GPU')"
echo ""

# Setup environment
cd /data/junjiexiong/algoverse/rag

# Create logs directory
mkdir -p logs

# Activate virtual environment (created via srun earlier)
source /data/junjiexiong/algoverse/rag/.venv/bin/activate

# Set cache directories to writable location (cluster has restricted home permissions)
export HF_HOME="/data/junjiexiong/algoverse/rag/.cache/huggingface"
export TRANSFORMERS_CACHE="/data/junjiexiong/algoverse/rag/.cache/transformers"
export TORCH_HOME="/data/junjiexiong/algoverse/rag/.cache/torch"
export XDG_CACHE_HOME="/data/junjiexiong/algoverse/rag/.cache"
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$TORCH_HOME"

echo "Verifying installations..."
python -c "from docling.document_converter import DocumentConverter; print('✓ Docling OK')" || { echo "ERROR: Docling import failed"; exit 1; }
python -c "from langchain_core.documents import Document; print('✓ LangChain OK')" || { echo "ERROR: LangChain import failed"; exit 1; }

# Set environment variables for GPU
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

# Check if PDFs exist
PDF_DIR="data/test_files/finance-bench-pdfs"
if [ ! -d "$PDF_DIR" ] || [ -z "$(ls -A $PDF_DIR/*.pdf 2>/dev/null)" ]; then
    echo "ERROR: No PDFs found in $PDF_DIR"
    echo "Please transfer PDFs to cluster first:"
    echo "  rsync -avz data/test_files/finance-bench-pdfs/ junjiexiong@cluster:$PWD/data/test_files/finance-bench-pdfs/"
    exit 1
fi

PDF_COUNT=$(ls -1 $PDF_DIR/*.pdf 2>/dev/null | wc -l)
echo "Found $PDF_COUNT PDFs to process"
echo ""

# Run ingestion
echo "Starting Docling ingestion..."
python src/ingest_docling.py \
    --input-dir "$PDF_DIR" \
    --output-dir chroma_docling \
    --chunk-size 2500 \
    --batch-size 10

# Verify results
echo ""
echo "========================================"
echo "Verification"
echo "========================================"
python -c "
from langchain_chroma import Chroma
from src.config import get_embedding_model

db = Chroma(persist_directory='chroma_docling', embedding_function=get_embedding_model('bge-large'))
total = db._collection.count()

# Count tables
result = db.get(where={'element_type': 'table'}, limit=1000)
table_count = len(result['documents'])

print(f'Total chunks:  {total}')
print(f'Table chunks:  {table_count}')
print(f'Table ratio:   {table_count/total*100:.1f}%')
"

echo ""
echo "========================================"
echo "Job Complete: $(date)"
echo "========================================"
