#!/usr/bin/env python3
"""
Fast PDF ingestion using PyPDF2/PyMuPDF for quick local testing.

This provides basic text extraction without GPU requirements.
Table extraction quality is lower than Docling/Marker, but this
allows for quick testing on local machines.

Usage:
    python src/ingest_pypdf.py --input-dir data/test_files/finance-bench-pdfs --output-dir chroma_local
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

# Setup paths
BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import get_embedding_model, DEFAULTS
from src.metadata_utils import parse_filename

# Try PyMuPDF (faster, better quality), fall back to PyPDF2
try:
    import fitz  # PyMuPDF
    USE_PYMUPDF = True
except ImportError:
    USE_PYMUPDF = False
    try:
        from pypdf import PdfReader
    except ImportError:
        from PyPDF2 import PdfReader


def extract_text_pymupdf(pdf_path: Path) -> str:
    """Extract text using PyMuPDF (fitz)."""
    doc = fitz.open(str(pdf_path))
    text_parts = []
    for page in doc:
        text_parts.append(page.get_text())
    doc.close()
    return "\n\n".join(text_parts)


def extract_text_pypdf(pdf_path: Path) -> str:
    """Extract text using PyPDF2/pypdf."""
    reader = PdfReader(str(pdf_path))
    text_parts = []
    for page in reader.pages:
        text_parts.append(page.extract_text() or "")
    return "\n\n".join(text_parts)


def process_pdf(pdf_path: Path, chunk_size: int = 2000, chunk_overlap: int = 200) -> List[Document]:
    """
    Process a single PDF with basic text extraction.

    Args:
        pdf_path: Path to PDF file
        chunk_size: Characters per chunk
        chunk_overlap: Overlap between chunks

    Returns:
        List of Document objects
    """
    chunks = []

    # Parse filename for metadata
    file_meta = parse_filename(pdf_path.name)
    if file_meta:
        file_meta_dict = file_meta.to_dict()
    else:
        file_meta_dict = {"source_file": pdf_path.name}

    try:
        # Extract text
        if USE_PYMUPDF:
            text = extract_text_pymupdf(pdf_path)
        else:
            text = extract_text_pypdf(pdf_path)

        if not text.strip():
            print(f"  ✗ No text extracted from {pdf_path.name}")
            return []

        # Split into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        text_chunks = splitter.split_text(text)

        for i, chunk_text in enumerate(text_chunks):
            if chunk_text.strip():
                chunks.append(Document(
                    page_content=chunk_text.strip(),
                    metadata={
                        **file_meta_dict,
                        'source': str(pdf_path),
                        'chunk_index': i
                    }
                ))

    except Exception as e:
        print(f"  ✗ Error processing {pdf_path.name}: {e}")
        return []

    return chunks


def get_processed_files(chroma_path: str) -> set:
    """Get set of already processed filenames from ChromaDB."""
    if not os.path.exists(chroma_path):
        return set()

    try:
        embeddings = get_embedding_model(DEFAULTS.embedding_model)
        db = Chroma(persist_directory=chroma_path, embedding_function=embeddings)
        result = db.get(include=['metadatas'])

        processed = set()
        for meta in result['metadatas']:
            if meta and 'source_file' in meta:
                processed.add(meta['source_file'])

        return processed
    except Exception as e:
        print(f"Warning: Could not check existing database: {e}")
        return set()


def run_ingestion(
    input_dir: str,
    output_dir: str,
    chunk_size: int = 2000,
    batch_size: int = 20,
    sample: Optional[int] = None,
):
    """
    Run PyPDF-based ingestion with batch processing.
    """
    # Get list of PDF files
    pdf_files = sorted(Path(input_dir).glob("*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in {input_dir}")
        return

    if sample:
        pdf_files = pdf_files[:sample]

    total_files = len(pdf_files)

    print("=" * 70)
    print(f"PYPDF INGESTION ({'PyMuPDF' if USE_PYMUPDF else 'PyPDF2'})")
    print("=" * 70)
    print(f"Input Dir:    {input_dir}")
    print(f"Output Dir:   {output_dir}")
    print(f"Total Files:  {total_files}")
    print(f"Chunk Size:   {chunk_size}")
    print(f"Batch Size:   {batch_size}")
    print("=" * 70)

    # Check for already processed files
    processed_files = get_processed_files(output_dir)
    files_to_process = [p for p in pdf_files if p.name not in processed_files]
    skipped = total_files - len(files_to_process)

    if skipped > 0:
        print(f"\nSkipping {skipped} already-ingested files")

    if not files_to_process:
        print("All files already processed!")
        return

    # Initialize ChromaDB
    embeddings = get_embedding_model(DEFAULTS.embedding_model)
    db = Chroma(persist_directory=output_dir, embedding_function=embeddings)

    # Process in batches
    start_time = time.time()
    total_chunks = 0

    for i in range(0, len(files_to_process), batch_size):
        batch = files_to_process[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = (len(files_to_process) + batch_size - 1) // batch_size

        print(f"\n[Batch {batch_num}/{total_batches}] Processing {len(batch)} files...")

        batch_chunks = []

        for pdf_path in batch:
            try:
                chunks = process_pdf(pdf_path, chunk_size)
                batch_chunks.extend(chunks)
                print(f"  ✓ {pdf_path.name} ({len(chunks)} chunks)")
            except Exception as e:
                print(f"  ✗ FAILED {pdf_path.name}: {str(e)[:80]}")

        # Save batch to ChromaDB
        if batch_chunks:
            print(f"  -> Saving {len(batch_chunks)} chunks to ChromaDB...", end=" ", flush=True)
            try:
                CHROMA_MAX_BATCH = 5000
                for k in range(0, len(batch_chunks), CHROMA_MAX_BATCH):
                    sub_batch = batch_chunks[k:k + CHROMA_MAX_BATCH]
                    db.add_documents(sub_batch)
                    print(".", end="", flush=True)
                print(" Done.")
                total_chunks += len(batch_chunks)
            except Exception as e:
                print(f" ERROR: {e}")

        # Progress update
        elapsed = time.time() - start_time
        files_done = min(i + batch_size, len(files_to_process))
        rate = files_done / elapsed if elapsed > 0 else 0
        eta = (len(files_to_process) - files_done) / rate if rate > 0 else 0
        print(f"  Progress: {files_done}/{len(files_to_process)} files, ETA: {eta/60:.1f} min")

    # Summary
    print("\n" + "=" * 70)
    print("INGESTION COMPLETE")
    print("=" * 70)
    print(f"Total files processed: {len(files_to_process)}")
    print(f"Total chunks created:  {total_chunks}")
    print(f"Time elapsed:          {(time.time() - start_time)/60:.1f} minutes")
    print(f"ChromaDB location:     {output_dir}")

    # Verify
    final_count = db._collection.count()
    print(f"Final ChromaDB count:  {final_count}")


def main():
    parser = argparse.ArgumentParser(description="Fast PyPDF-based PDF Ingestion")
    parser.add_argument("--input-dir", type=str, required=True, help="Directory with PDFs")
    parser.add_argument("--output-dir", type=str, default="chroma_local", help="ChromaDB output")
    parser.add_argument("--chunk-size", type=int, default=2000, help="Max chars per chunk")
    parser.add_argument("--batch-size", type=int, default=20, help="Files per batch")
    parser.add_argument("--sample", type=int, help="Only process N files (for testing)")

    args = parser.parse_args()

    run_ingestion(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        chunk_size=args.chunk_size,
        batch_size=args.batch_size,
        sample=args.sample,
    )


if __name__ == "__main__":
    main()
