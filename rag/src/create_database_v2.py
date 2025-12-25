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


def process_pdf(pdf_path: Path) -> List[Document]:
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

    # Parse PDF into elements
    # Using hi_res for final build, user can change to fast via sed
    elements = partition_pdf(
        filename=str(pdf_path),
        strategy="hi_res",
        infer_table_structure=True,
        ocr_languages="eng",
        extract_images_in_pdf=False,
        include_page_breaks=True,
        max_characters=4000,
        new_after_n_chars=3000,
        combine_text_under_n_chars=500,
    )

    # Chunk by title sections
    chunked_elements = chunk_by_title(
        elements,
        max_characters=2000,
        combine_text_under_n_chars=1000,
        new_after_n_chars=1500,
    )

    for chunk in chunked_elements:
        elem_meta = chunk.metadata.to_dict() if hasattr(chunk.metadata, 'to_dict') else {}
        element_type = get_element_type(chunk)
        text = chunk.text

        if element_type == 'table':
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


def save_to_chroma(chunks: List[Document], chroma_path: str):
    """Save chunks to ChromaDB with batched embedding."""
    # Get embedding model
    embeddings = get_embedding_model(DEFAULTS.embedding_model)

    # Process in batches
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
    print(f"{ '='*80}\n")


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
    table_count = 0
    prose_count = 0
    
    # Safely set workers
    max_workers = 16

    print(f"\n{'='*80}")
    print("INGESTION START")
    print(f"Parallelism: {max_workers} workers")
    print(f"{ '='*80}\n")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {executor.submit(process_pdf, pdf_path): pdf_path for pdf_path in pdf_files}
        completed_count = 0
        for future in as_completed(future_to_file):
            pdf_path = future_to_file[future]
            completed_count += 1
            try:
                chunks = future.result()
                for c in chunks:
                    if c.metadata.get('element_type') == 'table':
                        table_count += 1
                    else:
                        prose_count += 1
                all_chunks.extend(chunks)
                print(f"[{completed_count}/{total_files}] {pdf_path.name}... ✓ {len(chunks)} chunks")
            except Exception as e:
                print(f"[{completed_count}/{total_files}] {pdf_path.name}... ✗ FAILED: {str(e)[:60]}")
                failed_files.append({'file': pdf_path.name, 'error': str(e)})

    return all_chunks


def main():
    parser = argparse.ArgumentParser(description="Finance RAG Ingestion")
    parser.add_argument("--sample", type=int, help="Only process N PDFs")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_CHROMA_PATH)
    parser.add_argument("--data-dir", type=str, default=DATA_PATH)
    args = parser.parse_args()

    chunks = load_and_process_pdfs_parallel(args.data_dir, sample=args.sample)
    if not chunks:
        print("No chunks created.")
        return

    save_to_chroma(chunks, args.output_dir)


if __name__ == "__main__":
    main()