#!/bin/bash
#SBATCH --job-name=dsrag-finance-parallel
#SBATCH --partition=production
#SBATCH --gres=gpu:6
#SBATCH --time=24:00:00
#SBATCH --output=logs/dsrag_finance_%j.out
#SBATCH --error=logs/dsrag_finance_%j.err

# =============================================================================
# dsRAG-Style Finance Ingestion - 6-GPU PARALLEL VERSION
# =============================================================================
# Builds ChromaDB with:
#   - Cohere embed-v3 (asymmetric query/document embeddings)
#   - LLM-generated AutoContext headers (batched for 5x speed)
#   - Docling table extraction (97.9% accuracy)
#   - 6-GPU parallel processing (6x speedup!)
#
# Resources: 6 GPUs, 128GB RAM, 16 CPUs
# Expected: ~90K chunks from 267 PDFs (3 skipped)
# Time: ~4-5 hours (vs ~24 hours sequential)
# Cost: ~$15-25 (Cohere + GPT-4o-mini)
#
# Skip list (problematic PDFs for manual investigation):
#   - AMD_2015_10K.pdf: Corrupted (PDFium data format error)
#   - AMCOR_2022_8K_dated-2022-04-26.pdf: Font error, worker hung
#   - CVSHEALTH_2022_10K.pdf: Worker hung during processing
#
# Usage:
#   sbatch scripts/ingest_dsrag_finance.slurm
#
# Monitor:
#   squeue -u $USER
#   tail -f logs/dsrag_finance_*.out
# =============================================================================

echo "========================================"
echo "dsRAG Finance Ingestion Starting (6-GPU PARALLEL)"
echo "========================================"
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "GPUs: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'No GPU')"
echo "GPU Count: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l || echo '0')"
echo ""

# Setup environment (CRITICAL: cluster has restricted home permissions)
export HOME="/data/junjiexiong"
export HF_HOME="/data/junjiexiong/.cache/huggingface"
export TRANSFORMERS_CACHE="/data/junjiexiong/.cache/transformers"
export TORCH_HOME="/data/junjiexiong/.cache/torch"
export XDG_CACHE_HOME="/data/junjiexiong/.cache"
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$TORCH_HOME"

cd /data/junjiexiong/algoverse/rag
mkdir -p logs

# Activate virtual environment
source /data/junjiexiong/algoverse/rag/.venv/bin/activate

# Verify environment
echo "Verifying installations..."
python -c "from docling.document_converter import DocumentConverter; print('Docling OK')" || exit 1
python -c "import cohere; print('Cohere SDK OK')" || exit 1
python -c "from openai import OpenAI; print('OpenAI SDK OK')" || exit 1

# Check API keys
echo "Checking API keys..."
python -c "
import os
cohere_key = os.environ.get('COHERE_API_KEY') or os.environ.get('CO_API_KEY')
openai_key = os.environ.get('OPENAI_API_KEY')
if not cohere_key: raise ValueError('COHERE_API_KEY not set')
if not openai_key: raise ValueError('OPENAI_API_KEY not set')
print(f'  COHERE_API_KEY: {cohere_key[:8]}...')
print(f'  OPENAI_API_KEY: {openai_key[:8]}...')
"

# Check input PDFs
PDF_DIR="data/test_files/finance-bench-pdfs"
if [ ! -d "$PDF_DIR" ] || [ -z "$(ls -A $PDF_DIR/*.pdf 2>/dev/null)" ]; then
    echo "ERROR: No PDFs found in $PDF_DIR"
    echo "Please transfer PDFs first:"
    echo "  rsync -avz data/test_files/finance-bench-pdfs/ cluster:$PWD/$PDF_DIR/"
    exit 1
fi

PDF_COUNT=$(ls -1 $PDF_DIR/*.pdf 2>/dev/null | wc -l)
echo "Found $PDF_COUNT PDFs to process"
echo ""

# GPU settings - let workers manage their own CUDA_VISIBLE_DEVICES
export TOKENIZERS_PARALLELISM=false

# Enable multiprocessing spawn for CUDA safety
export CUDA_LAUNCH_BLOCKING=0

# Run dsRAG PARALLEL ingestion
echo "========================================"
echo "Starting dsRAG Finance Ingestion (6-GPU PARALLEL)"
echo "========================================"

python scripts/build_dsrag_parallel.py \
    --domain finance \
    --input-dir "$PDF_DIR" \
    --output-dir chroma_dsrag_finance \
    --num-workers 6 \
    --embedding-model cohere-v3 \
    --autocontext-model gpt-4o-mini \
    --chunk-size 2500 \
    --batch-size 500 \
    --skip-files "AMD_2015_10K.pdf,AMCOR_2022_8K_dated-2022-04-26.pdf,CVSHEALTH_2022_10K.pdf"

# Verify results
echo ""
echo "========================================"
echo "Verification"
echo "========================================"
python -c "
from langchain_chroma import Chroma
from src.config import get_embedding_model

db = Chroma(
    persist_directory='chroma_dsrag_finance',
    embedding_function=get_embedding_model('cohere-v3')
)
total = db._collection.count()

# Sample a chunk to verify AutoContext
sample = db.get(limit=1, include=['documents', 'metadatas'])
has_autocontext = sample['metadatas'][0].get('has_autocontext', False)

print(f'Total chunks:     {total}')
print(f'Has AutoContext:  {has_autocontext}')

# Count by type
result = db.get(where={'element_type': 'table'}, limit=10000)
table_count = len(result['documents'])
print(f'Table chunks:     {table_count}')
print(f'Table ratio:      {table_count/total*100:.1f}%')
"

echo ""
echo "========================================"
echo "Job Complete: $(date)"
echo "========================================"
