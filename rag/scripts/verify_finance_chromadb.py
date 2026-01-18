#!/usr/bin/env python3
"""
Verify Finance ChromaDB Integrity.

Run this script on the cluster BEFORE re-ingesting to check if:
1. ChromaDB exists and is accessible
2. Expected chunk count is reasonable (85K-95K for 267 PDFs)
3. Checkpoints are consistent with ChromaDB
4. Sample queries work correctly

Usage:
    python scripts/verify_finance_chromadb.py

Exit codes:
    0 - All checks passed, data appears healthy
    1 - Corruption detected, re-ingestion recommended
    2 - ChromaDB not found (needs fresh ingestion)
"""

import sys
from pathlib import Path

# Setup paths
BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

CHROMADB_PATH = BASE_DIR / "chroma_dsrag_finance"
CHECKPOINT_DIR = BASE_DIR / "chroma_dsrag_finance_checkpoints"

# Expected ranges
EXPECTED_PDF_COUNT = 267  # 270 total - 3 skipped
MIN_EXPECTED_CHUNKS = 75000  # Conservative lower bound
MAX_EXPECTED_CHUNKS = 100000  # Upper bound
CHUNKS_PER_PDF_MIN = 200  # Minimum expected chunks per PDF
CHUNKS_PER_PDF_MAX = 500  # Maximum expected chunks per PDF


def check_chromadb_exists():
    """Check if ChromaDB directory exists."""
    print("\n" + "=" * 60)
    print("CHECK 1: ChromaDB Directory Existence")
    print("=" * 60)

    if not CHROMADB_PATH.exists():
        print(f"  ❌ ChromaDB not found at: {CHROMADB_PATH}")
        return False

    # Check for essential ChromaDB files
    chroma_files = list(CHROMADB_PATH.glob("*"))
    print(f"  ✅ ChromaDB directory exists: {CHROMADB_PATH}")
    print(f"     Files in directory: {len(chroma_files)}")

    return True


def check_chromadb_count():
    """Check ChromaDB chunk count and verify it's within expected range."""
    print("\n" + "=" * 60)
    print("CHECK 2: ChromaDB Chunk Count")
    print("=" * 60)

    try:
        import chromadb
        client = chromadb.PersistentClient(path=str(CHROMADB_PATH))

        # Try both possible collection names
        collection_names = ['langchain', 'finance']
        collection = None

        for name in collection_names:
            try:
                collection = client.get_collection(name)
                print(f"  Found collection: '{name}'")
                break
            except:
                continue

        if collection is None:
            print(f"  ❌ No collection found. Tried: {collection_names}")
            available = [c.name for c in client.list_collections()]
            print(f"     Available collections: {available}")
            return False, 0

        chunk_count = collection.count()
        print(f"  Chunk count: {chunk_count:,}")

        # Check if count is in expected range
        if chunk_count < MIN_EXPECTED_CHUNKS:
            print(f"  ❌ Count too LOW (expected >{MIN_EXPECTED_CHUNKS:,})")
            print(f"     This suggests incomplete ingestion")
            return False, chunk_count
        elif chunk_count > MAX_EXPECTED_CHUNKS:
            print(f"  ⚠️ Count unusually HIGH (expected <{MAX_EXPECTED_CHUNKS:,})")
            print(f"     Possible duplicate chunks")
            return False, chunk_count
        else:
            print(f"  ✅ Count within expected range ({MIN_EXPECTED_CHUNKS:,}-{MAX_EXPECTED_CHUNKS:,})")
            return True, chunk_count

    except Exception as e:
        print(f"  ❌ Error accessing ChromaDB: {e}")
        return False, 0


def check_checkpoints():
    """Check checkpoint files for processed PDFs."""
    print("\n" + "=" * 60)
    print("CHECK 3: Checkpoint Files")
    print("=" * 60)

    if not CHECKPOINT_DIR.exists():
        print(f"  ⚠️ Checkpoint directory not found: {CHECKPOINT_DIR}")
        print(f"     This is OK if ChromaDB was built without checkpoints")
        return True, 0

    checkpoint_files = list(CHECKPOINT_DIR.glob("*.json"))
    failed_pdfs_file = CHECKPOINT_DIR / "failed_pdfs.jsonl"

    print(f"  Checkpoint files: {len(checkpoint_files)}")
    print(f"  Expected: ~{EXPECTED_PDF_COUNT}")

    # Check failed PDFs
    failed_count = 0
    if failed_pdfs_file.exists():
        with open(failed_pdfs_file, 'r') as f:
            failed_count = sum(1 for line in f if line.strip())
        print(f"  Failed PDFs logged: {failed_count}")

    # Verify checkpoint consistency
    if len(checkpoint_files) < EXPECTED_PDF_COUNT - 10:
        print(f"  ❌ Too few checkpoints ({len(checkpoint_files)} < {EXPECTED_PDF_COUNT - 10})")
        return False, len(checkpoint_files)
    else:
        print(f"  ✅ Checkpoint count looks reasonable")
        return True, len(checkpoint_files)


def check_sample_query():
    """Test a sample retrieval query."""
    print("\n" + "=" * 60)
    print("CHECK 4: Sample Query Test")
    print("=" * 60)

    try:
        from langchain_chroma import Chroma
        from src.config import get_embedding_model

        print("  Loading embedding model...")
        embeddings = get_embedding_model("cohere-v3")

        db = Chroma(
            persist_directory=str(CHROMADB_PATH),
            embedding_function=embeddings,
            collection_name='langchain'
        )

        # Test query
        test_query = "What was the total revenue?"
        print(f"  Testing query: '{test_query}'")

        results = db.similarity_search(test_query, k=3)

        if len(results) == 0:
            print("  ❌ Query returned no results")
            return False

        print(f"  ✅ Query returned {len(results)} results")

        # Check if results have AutoContext
        sample = results[0]
        has_autocontext = sample.metadata.get('has_autocontext', False)
        print(f"  AutoContext present: {has_autocontext}")

        # Show a snippet of the first result
        snippet = sample.page_content[:200].replace('\n', ' ')
        print(f"  Sample snippet: {snippet}...")

        return True

    except Exception as e:
        print(f"  ❌ Query test failed: {e}")
        return False


def check_data_consistency(chunk_count):
    """Check if chunk count is consistent with expected PDFs."""
    print("\n" + "=" * 60)
    print("CHECK 5: Data Consistency")
    print("=" * 60)

    avg_chunks_per_pdf = chunk_count / EXPECTED_PDF_COUNT
    print(f"  Average chunks per PDF: {avg_chunks_per_pdf:.1f}")
    print(f"  Expected range: {CHUNKS_PER_PDF_MIN}-{CHUNKS_PER_PDF_MAX}")

    if avg_chunks_per_pdf < CHUNKS_PER_PDF_MIN:
        print(f"  ❌ Too few chunks per PDF - possible parsing failures")
        return False
    elif avg_chunks_per_pdf > CHUNKS_PER_PDF_MAX:
        print(f"  ⚠️ More chunks than expected - possible duplicates")
        return False
    else:
        print(f"  ✅ Chunk distribution looks healthy")
        return True


def main():
    print("=" * 60)
    print("FINANCE CHROMADB INTEGRITY VERIFICATION")
    print("=" * 60)
    print(f"ChromaDB Path: {CHROMADB_PATH}")
    print(f"Checkpoint Path: {CHECKPOINT_DIR}")

    all_passed = True

    # Check 1: ChromaDB exists
    if not check_chromadb_exists():
        print("\n" + "=" * 60)
        print("RESULT: ChromaDB NOT FOUND")
        print("=" * 60)
        print("Action: Run fresh ingestion")
        return 2

    # Check 2: Chunk count
    count_ok, chunk_count = check_chromadb_count()
    all_passed = all_passed and count_ok

    # Check 3: Checkpoints
    checkpoint_ok, checkpoint_count = check_checkpoints()
    all_passed = all_passed and checkpoint_ok

    # Check 4: Sample query
    query_ok = check_sample_query()
    all_passed = all_passed and query_ok

    # Check 5: Data consistency (only if we have a chunk count)
    if chunk_count > 0:
        consistency_ok = check_data_consistency(chunk_count)
        all_passed = all_passed and consistency_ok

    # Summary
    print("\n" + "=" * 60)
    if all_passed:
        print("RESULT: ALL CHECKS PASSED")
        print("=" * 60)
        print("ChromaDB appears healthy. Re-ingestion may NOT be needed.")
        print("\nIf you still want to re-ingest, run:")
        print("  rm -rf chroma_dsrag_finance/")
        print("  rm -rf chroma_dsrag_finance_checkpoints/")
        print("  sbatch scripts/ingest_dsrag_finance.slurm")
        return 0
    else:
        print("RESULT: CORRUPTION DETECTED")
        print("=" * 60)
        print("Re-ingestion is RECOMMENDED.")
        print("\nRun these commands to re-ingest:")
        print("  rm -rf chroma_dsrag_finance/")
        print("  rm -rf chroma_dsrag_finance_checkpoints/")
        print("  sbatch scripts/ingest_dsrag_finance.slurm")
        return 1


if __name__ == "__main__":
    sys.exit(main())
