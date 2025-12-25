"""
Improved database creation with:
1. Filename metadata extraction (company, year, doc_type)
2. Table preservation as markdown
3. Element type tagging (table vs prose)

Usage:
    python src/create_database_v2.py [--sample N] [--output-dir DIR]
"""

import warnings
warnings.filterwarnings("ignore", message=".*max_size.*deprecated.*")

import argparse
import os
import shutil
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_chroma import Chroma
from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title

# Load environment variables
load_dotenv()

# Paths
BASE_DIR = Path(__file__).parent.parent
DEFAULT_CHROMA_PATH = str(BASE_DIR / "chroma")
DATA_PATH = str(BASE_DIR / "data/test_files/finance-bench-pdfs")

# Add parent to path for imports
import sys
sys.path.insert(0, str(BASE_DIR))

# Import config and metadata utils
from src.config import get_embedding_model, DEFAULTS
from src.metadata_utils import parse_filename, normalize_company_name


def html_table_to_markdown(html: str) -> str:
    """
    Convert HTML table to markdown format.
    Preserves structure for better semantic understanding.
    """
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, 'html.parser')
        table = soup.find('table')
        if not table:
            return html

        rows = []
        for tr in table.find_all('tr'):
            cells = []
            for cell in tr.find_all(['th', 'td']):
                # Clean cell text
                text = cell.get_text(strip=True)
                # Replace problematic characters
                text = text.replace('|', '\\|').replace('\n', ' ')
                cells.append(text)
            if cells:
                rows.append('| ' + ' | '.join(cells) + ' |')

        if len(rows) >= 1:
            # Add header separator after first row
            num_cols = rows[0].count('|') - 1
            separator = '|' + '---|' * num_cols
            rows.insert(1, separator)

        return '\n'.join(rows)
    except ImportError:
        # BeautifulSoup not available, return original
        return html
    except Exception:
        return html


def get_element_type(element) -> str:
    """Get the type of an unstructured element."""
    type_name = type(element).__name__

    # Map to simpler categories
    type_mapping = {
        'Table': 'table',
        'Title': 'title',
        'Header': 'header',
        'NarrativeText': 'prose',
        'Text': 'prose',
        'ListItem': 'list',
        'FigureCaption': 'figure',
        'Image': 'image',
        'PageBreak': 'page_break',
    }

    return type_mapping.get(type_name, 'other')


def process_pdf(pdf_path: Path, strategy: str = "hi_res", chunk_size: int = 1000) -> List[Document]:
    """
    Process a single PDF with improved parsing.
    """
    chunks = []

    # Parse filename for metadata
    file_meta = parse_filename(pdf_path.name)
    if not file_meta:
        file_meta_dict = {"source_file": pdf_path.name}
    else:
        file_meta_dict = file_meta.to_dict()

    # Determine partition kwargs based on strategy
    partition_kwargs = {
        "filename": str(pdf_path),
        "strategy": strategy,
        "extract_images_in_pdf": False,
        "include_page_breaks": True,
        "max_characters": 4000,
        "new_after_n_chars": 3000,
        "combine_text_under_n_chars": 500,
    }

    if strategy == "hi_res":
        # Slow but accurate (OCR + Table Detection)
        partition_kwargs.update({
            "infer_table_structure": True,
            "ocr_languages": "eng",
        })
    else:
        # Fast (Text only)
        # No extra args needed for fast strategy
        pass

    # Parse PDF into elements
    try:
        elements = partition_pdf(**partition_kwargs)
    except Exception as e:
        print(f"Error partitioning {pdf_path}: {e}")
        return []

    # Chunk by title sections
    # chunk_by_title defaults: max=500, new_after=None, combine=None
    # We want customizable sizing
    
    # Calculate overlap buffer (roughly 10-20% less than max)
    new_after = int(chunk_size * 0.8)
    combine_under = int(chunk_size * 0.25)

    chunked_elements = chunk_by_title(
        elements,
        max_characters=chunk_size,
        combine_text_under_n_chars=combine_under,
        new_after_n_chars=new_after,
    )

    for chunk in chunked_elements:
        elem_meta = chunk.metadata.to_dict() if hasattr(chunk.metadata, 'to_dict') else {}
        element_type = get_element_type(chunk)
        text = chunk.text
        
        # Only process tables if we are in hi_res mode and have HTML
        if strategy == "hi_res" and element_type == 'table':
            html = elem_meta.get('text_as_html', '')
            if html:
                text = html_table_to_markdown(html)

        metadata = {
            **file_meta_dict,
            'element_type': element_type,
            'page_number': elem_meta.get('page_number', 0),
            'source': str(pdf_path),
        }

        cleaned_metadata = {}
        for key, value in metadata.items():
            if isinstance(value, list):
                cleaned_metadata[key] = ','.join(str(v) for v in value)
            elif isinstance(value, (str, int, float, bool)) or value is None:
                cleaned_metadata[key] = value
            else:
                cleaned_metadata[key] = str(value)

        if not text or not text.strip():
            continue

        chunks.append(Document(page_content=text, metadata=cleaned_metadata))

    return chunks


def get_processed_files(chroma_path: str) -> set:
    """
    Get set of source filenames already in ChromaDB.
    """
    if not os.path.exists(chroma_path):
        return set()

    try:
        embeddings = get_embedding_model(DEFAULTS.embedding_model)
        db = Chroma(persist_directory=chroma_path, embedding_function=embeddings)
        
        # Fetch all metadata (limit to avoid OOM on huge DBs, but 300 files is fine)
        # Note: Chroma get() without ids returns all if collection is small enough
        result = db.get(include=['metadatas'])
        
        processed = set()
        for meta in result['metadatas']:
            if meta and 'source_file' in meta:
                processed.add(meta['source_file'])
        
        print(f"Found {len(processed)} files already processed in {chroma_path}")
        return processed
    except Exception as e:
        print(f"Warning: Could not check existing database: {e}")
        return set()


def run_ingestion(
    data_path: str, 
    output_dir: str, 
    sample: Optional[int] = None, 
    batch_size: int = 10,
    strategy: str = "hi_res",
    chunk_size: int = 1000
):
    """
    Run ingestion with batch processing and resume capability.
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed

    # 1. Get list of files
    pdf_files = sorted(Path(data_path).glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {data_path}")
        return

    if sample:
        pdf_files = pdf_files[:sample]
        print(f"Sampling first {sample} files.")

    total_files = len(pdf_files)
    print(f"\n{'='*80}")
    print(f"INGESTION START")
    print(f"Total Files: {total_files}")
    print(f"Batch Size:  {batch_size}")
    print(f"Strategy:    {strategy}")
    print(f"Chunk Size:  {chunk_size}")
    print(f"Output Dir:  {output_dir}")
    print(f"{'='*80}\n")

    # 2. Check existing work
    processed_files = get_processed_files(output_dir)
    
    # Filter out already done
    files_to_process = [p for p in pdf_files if p.name not in processed_files]
    skipped_count = total_files - len(files_to_process)
    
    if skipped_count > 0:
        print(f"Skipping {skipped_count} files (already ingested).")
    
    if not files_to_process:
        print("All files processed! Exiting.")
        return

    # 3. Process in batches
    # Initialize DB connection once to ensure directory exists
    embeddings = get_embedding_model(DEFAULTS.embedding_model)
    db = Chroma(persist_directory=output_dir, embedding_function=embeddings)
    
    max_workers = min(16, os.cpu_count() or 1)
    print(f"Starting processing with {max_workers} workers...")

    for i in range(0, len(files_to_process), batch_size):
        batch = files_to_process[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = (len(files_to_process) + batch_size - 1) // batch_size
        
        print(f"\n[Batch {batch_num}/{total_batches}] Processing {len(batch)} files...")
        
        batch_chunks = []
        failed_in_batch = 0

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Pass new args to process_pdf
            future_to_file = {
                executor.submit(process_pdf, pdf_path, strategy, chunk_size): pdf_path 
                for pdf_path in batch
            }
            
            for future in as_completed(future_to_file):
                pdf_path = future_to_file[future]
                try:
                    chunks = future.result()
                    batch_chunks.extend(chunks)
                    print(f"  ✓ {pdf_path.name} ({len(chunks)} chunks)")
                except Exception as e:
                    print(f"  ✗ FAILED {pdf_path.name}: {str(e)[:100]}")
                    failed_in_batch += 1

        if batch_chunks:
            print(f"  -> Saving {len(batch_chunks)} chunks to ChromaDB...", end=" ", flush=True)
            try:
                # ChromaDB has a max batch size (usually 5461). We use 5000 to be safe.
                CHROMA_MAX_BATCH = 5000
                for k in range(0, len(batch_chunks), CHROMA_MAX_BATCH):
                    sub_batch = batch_chunks[k:k + CHROMA_MAX_BATCH]
                    db.add_documents(sub_batch)
                    print(f".", end="", flush=True)
                print(" Done.")
            except Exception as e:
                print(f"ERROR Saving Log: {e}")
        
        # Clear memory
        del batch_chunks


def main():
    parser = argparse.ArgumentParser(description="Finance RAG Ingestion (Resumable)")
    parser.add_argument("--sample", type=int, help="Only process N PDFs")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_CHROMA_PATH)
    parser.add_argument("--data-dir", type=str, default=DATA_PATH)
    parser.add_argument("--batch-size", type=int, default=10, help="Number of files to save at once")
    
    # New arguments
    parser.add_argument("--fast", action="store_true", help="Use fast strategy (no OCR/Table inference)")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Max characters per chunk")

    args = parser.parse_args()
    
    strategy = "fast" if args.fast else "hi_res"

    run_ingestion(
        data_path=args.data_dir, 
        output_dir=args.output_dir, 
        sample=args.sample, 
        batch_size=args.batch_size,
        strategy=strategy,
        chunk_size=args.chunk_size
    )


if __name__ == "__main__":
    main()