#!/usr/bin/env python3
"""
PDF ingestion using Marker for GPU-accelerated table extraction.

Marker achieves ~92% table accuracy with simpler dependencies than Docling.
It's more reliable to install on cluster environments.

Usage:
    # Local (GPU recommended)
    python src/ingest_marker.py --input-dir data/pdfs --output-dir chroma

    # On cluster
    sbatch scripts/ingest_docling.slurm
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

# Add custom package directory if on cluster
PKGDIR = "/data/junjiexiong/algoverse/rag/.packages"
if os.path.exists(PKGDIR):
    sys.path.insert(0, PKGDIR)

from langchain_core.documents import Document
from langchain_chroma import Chroma

from src.config import get_embedding_model, DEFAULTS
from src.metadata_utils import parse_filename


def process_pdf_marker(pdf_path: Path, chunk_size: int = 2500) -> List[Document]:
    """
    Process a single PDF with Marker for table extraction.

    Args:
        pdf_path: Path to PDF file
        chunk_size: Maximum characters per prose chunk (tables are never split)

    Returns:
        List of Document objects with metadata
    """
    from marker.converters.pdf import PdfConverter

    chunks = []

    # Parse filename for metadata
    file_meta = parse_filename(pdf_path.name)
    if file_meta:
        file_meta_dict = file_meta.to_dict()
    else:
        # Fallback for non-standard filenames
        file_meta_dict = {"source_file": pdf_path.name}

    try:
        # Convert PDF with Marker (GPU accelerated)
        converter = PdfConverter()
        rendered = converter(str(pdf_path))

        # Get markdown output
        markdown = rendered.markdown if hasattr(rendered, 'markdown') else str(rendered)

    except Exception as e:
        print(f"  ✗ Marker error on {pdf_path.name}: {e}")
        return []

    # Table-aware chunking
    # Tables (lines starting with |) are kept as single chunks
    # Prose is chunked at chunk_size limit
    current_chunk = ""
    current_type = "prose"
    in_table = False

    for line in markdown.split('\n'):
        line_stripped = line.strip()

        # Detect table start
        if line_stripped.startswith('|') and not in_table:
            # Save any accumulated prose first
            if current_chunk.strip():
                chunks.append(Document(
                    page_content=current_chunk.strip(),
                    metadata={**file_meta_dict, 'element_type': 'prose', 'source': str(pdf_path)}
                ))
            # Start new table
            current_chunk = line + '\n'
            current_type = "table"
            in_table = True

        elif in_table:
            # Continue table if line starts with | or is separator (---)
            if line_stripped.startswith('|') or (line_stripped.startswith('-') and '|' in current_chunk):
                current_chunk += line + '\n'
            else:
                # Table ended - save as single chunk (never split)
                if current_chunk.strip():
                    chunks.append(Document(
                        page_content=current_chunk.strip(),
                        metadata={**file_meta_dict, 'element_type': 'table', 'source': str(pdf_path)}
                    ))
                # Start prose chunk
                current_chunk = line + '\n'
                current_type = "prose"
                in_table = False

        else:
            # Prose - accumulate and chunk at size limit
            current_chunk += line + '\n'

            if len(current_chunk) > chunk_size:
                # Find a good break point (end of sentence or paragraph)
                break_point = current_chunk.rfind('\n\n', 0, chunk_size)
                if break_point == -1:
                    break_point = current_chunk.rfind('. ', 0, chunk_size)
                if break_point == -1:
                    break_point = chunk_size

                chunk_text = current_chunk[:break_point].strip()
                if chunk_text:
                    chunks.append(Document(
                        page_content=chunk_text,
                        metadata={**file_meta_dict, 'element_type': 'prose', 'source': str(pdf_path)}
                    ))
                current_chunk = current_chunk[break_point:].strip() + '\n'

    # Don't forget the last chunk
    if current_chunk.strip():
        chunks.append(Document(
            page_content=current_chunk.strip(),
            metadata={**file_meta_dict, 'element_type': current_type, 'source': str(pdf_path)}
        ))

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
    chunk_size: int = 2500,
    batch_size: int = 10,
    sample: Optional[int] = None,
):
    """
    Run Marker-based ingestion with batch processing.

    Args:
        input_dir: Directory containing PDF files
        output_dir: ChromaDB output directory
        chunk_size: Max characters per prose chunk
        batch_size: Files to process before saving to ChromaDB
        sample: Optional limit on number of files to process
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
    print("MARKER INGESTION (GPU-Accelerated Table Extraction)")
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
                chunks = process_pdf_marker(pdf_path, chunk_size)
                batch_chunks.extend(chunks)
                print(f"  ✓ {pdf_path.name} ({len(chunks)} chunks)")
            except Exception as e:
                print(f"  ✗ FAILED {pdf_path.name}: {str(e)[:80]}")

        # Save batch to ChromaDB
        if batch_chunks:
            print(f"  -> Saving {len(batch_chunks)} chunks to ChromaDB...", end=" ", flush=True)
            try:
                # ChromaDB batch size limit
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
    parser = argparse.ArgumentParser(description="Marker-based PDF Ingestion")
    parser.add_argument("--input-dir", type=str, required=True, help="Directory with PDFs")
    parser.add_argument("--output-dir", type=str, default="chroma", help="ChromaDB output")
    parser.add_argument("--chunk-size", type=int, default=2500, help="Max chars per chunk")
    parser.add_argument("--batch-size", type=int, default=10, help="Files per batch")
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
