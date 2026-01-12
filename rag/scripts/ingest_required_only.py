#!/usr/bin/env python3
"""
Targeted ingestion - only ingest the documents required by FinanceBench questions.
Uses MPS (Apple Metal) acceleration on Mac for faster processing.
"""

import json
import sys
import time
from pathlib import Path

# Setup paths
BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import get_embedding_model, DEFAULTS
from src.metadata_utils import parse_filename

# Try PyMuPDF (faster)
try:
    import fitz
    USE_PYMUPDF = True
except ImportError:
    USE_PYMUPDF = False
    from pypdf import PdfReader


def extract_text(pdf_path: Path) -> str:
    """Extract text from PDF."""
    if USE_PYMUPDF:
        doc = fitz.open(str(pdf_path))
        text_parts = [page.get_text() for page in doc]
        doc.close()
        return "\n\n".join(text_parts)
    else:
        reader = PdfReader(str(pdf_path))
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)


def process_pdf(pdf_path: Path, chunk_size: int = 2000) -> list[Document]:
    """Process a single PDF."""
    file_meta = parse_filename(pdf_path.name)
    file_meta_dict = file_meta.to_dict() if file_meta else {"source_file": pdf_path.name}

    try:
        text = extract_text(pdf_path)
        if not text.strip():
            return []

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        chunks = []
        for i, chunk_text in enumerate(splitter.split_text(text)):
            if chunk_text.strip():
                chunks.append(Document(
                    page_content=chunk_text.strip(),
                    metadata={**file_meta_dict, 'source': str(pdf_path), 'chunk_index': i}
                ))
        return chunks
    except Exception as e:
        print(f"  Error: {e}")
        return []


def main():
    # Load questions
    with open(BASE_DIR / 'data/question_sets/financebench_open_source.jsonl') as f:
        questions = [json.loads(line) for line in f]

    # Get required docs
    required = set(q.get('doc_name') for q in questions[:150] if q.get('doc_name'))

    # Find available PDFs
    pdf_dir = BASE_DIR / 'data/test_files/finance-bench-pdfs'
    available = {p.stem: p for p in pdf_dir.glob('*.pdf')}

    to_process = [available[doc] for doc in required if doc in available]
    missing = [doc for doc in required if doc not in available]

    print("=" * 70)
    print("TARGETED INGESTION - Required Documents Only")
    print("=" * 70)
    print(f"Questions analyzed:  150")
    print(f"Required documents:  {len(required)}")
    print(f"Available locally:   {len(to_process)}")
    print(f"Missing PDFs:        {len(missing)}")
    print("=" * 70)

    # Initialize ChromaDB with MPS-enabled embeddings
    embeddings = get_embedding_model(DEFAULTS.embedding_model)
    output_dir = str(BASE_DIR / 'chroma_financebench')
    db = Chroma(persist_directory=output_dir, embedding_function=embeddings)

    # Check existing
    try:
        existing = db.get(include=['metadatas'])
        processed = set(m.get('source_file') for m in existing['metadatas'] if m)
        to_process = [p for p in to_process if p.name not in processed]
        if len(processed) > 0:
            print(f"Skipping {len(processed)} already-processed files")
    except:
        pass

    if not to_process:
        print("All required documents already ingested!")
        return

    print(f"\nProcessing {len(to_process)} documents...")

    start_time = time.time()
    total_chunks = 0
    batch_size = 5  # Small batches for faster progress

    for i in range(0, len(to_process), batch_size):
        batch = to_process[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = (len(to_process) + batch_size - 1) // batch_size

        print(f"\n[Batch {batch_num}/{total_batches}]")

        batch_chunks = []
        for pdf_path in batch:
            chunks = process_pdf(pdf_path)
            batch_chunks.extend(chunks)
            print(f"  ✓ {pdf_path.name} ({len(chunks)} chunks)")

        if batch_chunks:
            print(f"  Embedding {len(batch_chunks)} chunks...", end=" ", flush=True)
            db.add_documents(batch_chunks)
            print("Done")
            total_chunks += len(batch_chunks)

        elapsed = time.time() - start_time
        rate = (i + len(batch)) / elapsed if elapsed > 0 else 0
        eta = (len(to_process) - i - len(batch)) / rate if rate > 0 else 0
        print(f"  Progress: {i + len(batch)}/{len(to_process)}, ETA: {eta/60:.1f} min")

    print("\n" + "=" * 70)
    print("INGESTION COMPLETE")
    print("=" * 70)
    print(f"Documents processed: {len(to_process)}")
    print(f"Total chunks:        {total_chunks}")
    print(f"Time elapsed:        {(time.time() - start_time)/60:.1f} minutes")
    print(f"Output:              {output_dir}")
    print(f"Final count:         {db._collection.count()}")

    if missing:
        print(f"\n⚠️  {len(missing)} documents missing - questions using these will fail:")
        for doc in sorted(missing)[:10]:
            print(f"    - {doc}")
        if len(missing) > 10:
            print(f"    ... and {len(missing) - 10} more")


if __name__ == "__main__":
    main()
