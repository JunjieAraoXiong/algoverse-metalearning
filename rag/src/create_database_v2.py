"""
Improved database creation with:
1. Filename metadata extraction (company, year, doc_type)
2. Table preservation as markdown
3. Element type tagging (table vs prose)

Usage:
    python src/create_database_v2.py [--sample N] [--output-dir DIR]
"""

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


def process_pdf(pdf_path: Path) -> List[Document]:
    """
    Process a single PDF with improved parsing.

    Returns list of Document objects with rich metadata.
    """
    chunks = []

    # Parse filename for metadata
    file_meta = parse_filename(pdf_path.name)
    if not file_meta:
        print(f"  Warning: Could not parse filename {pdf_path.name}")
        file_meta_dict = {"source_file": pdf_path.name}
    else:
        file_meta_dict = file_meta.to_dict()

    # Parse PDF into elements
    elements = partition_pdf(
        filename=str(pdf_path),
        strategy="hi_res",  # High resolution for better table detection
        infer_table_structure=True,  # Extract tables as structured data
        ocr_languages="eng",  # Explicitly set language to silence warnings
        extract_images_in_pdf=False,
        include_page_breaks=True,
        max_characters=4000,
        new_after_n_chars=3000,
        combine_text_under_n_chars=500,
    )

    # Chunk by title sections, keeping tables intact
    chunked_elements = chunk_by_title(
        elements,
        max_characters=2000,
        combine_text_under_n_chars=1000,
        new_after_n_chars=1500,
    )

    for chunk in chunked_elements:
        # Get element metadata
        elem_meta = chunk.metadata.to_dict() if hasattr(chunk.metadata, 'to_dict') else {}

        # Determine element type
        element_type = get_element_type(chunk)

        # Get text content
        text = chunk.text

        # Convert tables to markdown if we have HTML
        if element_type == 'table':
            html = elem_meta.get('text_as_html', '')
            if html:
                text = html_table_to_markdown(html)

        # Build comprehensive metadata
        metadata = {
            # From filename
            **file_meta_dict,
            # Element info
            'element_type': element_type,
            'page_number': elem_meta.get('page_number', 0),
            # Source path for reference
            'source': str(pdf_path),
        }

        # Clean metadata values (ChromaDB only accepts str, int, float, bool, None)
        cleaned_metadata = {}
        for key, value in metadata.items():
            if isinstance(value, list):
                cleaned_metadata[key] = ','.join(str(v) for v in value)
            elif isinstance(value, (str, int, float, bool)) or value is None:
                cleaned_metadata[key] = value
            else:
                cleaned_metadata[key] = str(value)

        # Skip empty chunks
        if not text or not text.strip():
            continue

        chunks.append(Document(
            page_content=text,
            metadata=cleaned_metadata
        ))

    return chunks


    # Process in batches (increased for H100s)
    batch_size = 5000
    total_batches = (len(chunks) + batch_size - 1) // batch_size

    db = None
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        batch_num = (i // batch_size) + 1

        print(f"[Batch {batch_num}/{total_batches}] Embedding {len(batch)} chunks...", end=" ", flush=True)

        try:
            if db is None:
                db = Chroma.from_documents(
                    batch, embeddings, persist_directory=chroma_path
                )
            else:
                db.add_documents(batch)
            print(f"✓ ({i + len(batch)}/{len(chunks)} total)")
        except Exception as e:
            print(f"✗ FAILED: {str(e)[:80]}")
            continue

    print(f"\n{'='*80}")
    print(f"✓ Database created at {chroma_path}")
    print(f"✓ Total chunks: {len(chunks)}")
    print(f"{'='*80}\n")


def load_and_process_pdfs_parallel(data_path: str, sample: Optional[int] = None) -> List[Document]:
    """
    Load and process all PDFs in parallel.
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import multiprocessing

    pdf_files = sorted(Path(data_path).glob("*.pdf"))

    if sample:
        pdf_files = pdf_files[:sample]

    total_files = len(pdf_files)
    all_chunks = []
    failed_files = []

    # Stats
    table_count = 0
    prose_count = 0
    
    # Determine optimal workers
    # Hardcoded to 16 to prevent OOM on 200GB nodes (16 workers * ~4GB/worker = 64GB)
    max_workers = 16

    print(f"\n{'='*80}")
    print("IMPROVED INGESTION V2 (PARALLEL)")
    print(f"Features: Filename metadata, Table preservation, Element tagging")
    print(f"Parallelism: {max_workers} workers")
    print(f"{'='*80}\n")
    print(f"Processing {total_files} PDF files...")
    print()

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_file = {executor.submit(process_pdf, pdf_path): pdf_path for pdf_path in pdf_files}
        
        completed_count = 0
        for future in as_completed(future_to_file):
            pdf_path = future_to_file[future]
            completed_count += 1
            
            try:
                chunks = future.result()
                
                # Update stats
                file_tables = 0
                file_prose = 0
                for c in chunks:
                    if c.metadata.get('element_type') == 'table':
                        file_tables += 1
                    else:
                        file_prose += 1
                
                table_count += file_tables
                prose_count += file_prose
                
                all_chunks.extend(chunks)
                print(f"[{completed_count}/{total_files}] {pdf_path.name}... ✓ {len(chunks)} chunks")
                
            except Exception as e:
                print(f"[{completed_count}/{total_files}] {pdf_path.name}... ✗ FAILED: {str(e)[:60]}")
                failed_files.append({'file': pdf_path.name, 'error': str(e)})

    # Summary
    print(f"\n{'='*80}")
    print("PROCESSING COMPLETE")
    print(f"{'='*80}")
    print(f"Files processed: {total_files - len(failed_files)}/{total_files}")
    print(f"Total chunks: {len(all_chunks)}")
    print(f"  - Tables: {table_count}")
    print(f"  - Prose/Other: {prose_count}")

    if failed_files:
        print(f"\nFailed files ({len(failed_files)}):")
        for f in failed_files[:10]:
            print(f"  - {f['file']}")

    if all_chunks:
        print(f"\n{'='*80}")
        print("SAMPLE CHUNK")
        print(f"{'='*80}")
        sample_chunk = all_chunks[0]
        print(f"Metadata: {sample_chunk.metadata}")
        print(f"Content preview: {sample_chunk.page_content[:300]}...")

    print()
    return all_chunks


def main():
    parser = argparse.ArgumentParser(description="Improved PDF ingestion for FinanceBench")
    parser.add_argument("--sample", type=int, help="Only process N PDFs (for testing)")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_CHROMA_PATH,
                        help="ChromaDB output directory")
    parser.add_argument("--data-dir", type=str, default=DATA_PATH,
                        help="Directory containing PDFs")
    args = parser.parse_args()

    # Load and process PDFs (Parallel)
    chunks = load_and_process_pdfs_parallel(args.data_dir, sample=args.sample)

    if not chunks:
        print("No chunks created. Exiting.")
        return

    # Save to ChromaDB
    save_to_chroma(chunks, args.output_dir)


if __name__ == "__main__":
    main()
