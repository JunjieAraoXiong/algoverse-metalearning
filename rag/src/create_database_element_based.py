"""
Element-based database creation using Unstructured.io for better table handling.
This version preserves table structure and creates semantically coherent chunks.
"""

from langchain_core.documents import Document
from langchain_chroma import Chroma
from dotenv import load_dotenv
from pathlib import Path
from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title
import os
import shutil

# Load environment variables
load_dotenv()

# Paths
BASE_DIR = Path(__file__).parent.parent
CHROMA_PATH = str(BASE_DIR / "chroma")
DATA_PATH = str(BASE_DIR / "data/test_files/finance-bench-pdfs")

# Import config for embeddings
from src.config import get_embedding_model, DEFAULTS

# Use FREE local embeddings by default
embeddings = get_embedding_model(DEFAULTS.embedding_model)


def main():
    """Main entry point for element-based database generation."""
    generate_data_store()


def generate_data_store():
    """Load documents with element parsing, chunk semantically, and save to ChromaDB."""
    print("\n" + "="*80)
    print("ELEMENT-BASED DATABASE CREATION")
    print("Using Unstructured.io for table-aware parsing")
    print("="*80 + "\n")

    chunks = load_and_chunk_documents()
    save_to_chroma(chunks)


def load_and_chunk_documents():
    """
    Load PDFs with element-based parsing and create semantic chunks.
    Tables are kept as complete units to preserve financial data structure.
    """
    pdf_files = list(Path(DATA_PATH).glob("*.pdf"))
    total_files = len(pdf_files)
    all_chunks = []
    failed_files = []

    print(f"{'='*80}")
    print(f"Processing {total_files} PDF files with element-based parsing")
    print(f"{'='*80}\n")

    for idx, pdf_path in enumerate(pdf_files, 1):
        try:
            print(f"[{idx}/{total_files}] Parsing: {pdf_path.name}...", end=" ", flush=True)

            # Parse PDF into elements (titles, tables, text, etc.)
            elements = partition_pdf(
                filename=str(pdf_path),
                strategy="hi_res",  # High resolution for better table detection
                infer_table_structure=True,  # Extract tables as structured data
                extract_images_in_pdf=False,  # Skip images for now
                include_page_breaks=True,
                max_characters=4000,  # Max size for individual elements
                new_after_n_chars=3000,  # Soft max before creating new element
                combine_text_under_n_chars=500,  # Combine small elements
            )

            print(f"({len(elements)} elements) ", end="", flush=True)

            # Chunk elements by title sections, keeping tables intact
            chunks = chunk_by_title(
                elements,
                max_characters=2000,  # Max chunk size
                combine_text_under_n_chars=1000,  # Combine small sections
                new_after_n_chars=1500,  # Soft max
            )

            # Convert to LangChain Documents
            for chunk in chunks:
                # Get metadata from chunk
                metadata = chunk.metadata.to_dict() if hasattr(chunk.metadata, 'to_dict') else {}

                # Add source file to metadata
                metadata['source'] = str(pdf_path)

                # Clean metadata: ChromaDB only accepts str, int, float, bool, or None
                cleaned_metadata = {}
                for key, value in metadata.items():
                    if isinstance(value, list):
                        # Convert lists to comma-separated strings
                        cleaned_metadata[key] = ','.join(str(v) for v in value)
                    elif isinstance(value, (str, int, float, bool)) or value is None:
                        cleaned_metadata[key] = value
                    else:
                        # Convert other types to string
                        cleaned_metadata[key] = str(value)

                # Create Document
                doc = Document(
                    page_content=chunk.text,
                    metadata=cleaned_metadata
                )
                all_chunks.append(doc)

            print(f"✓ Created {len(chunks)} chunks")

        except Exception as e:
            print(f"✗ FAILED")
            failed_files.append({
                'file': pdf_path.name,
                'error': str(e)
            })
            continue

    print(f"\n{'='*80}")
    print(f"Successfully processed: {total_files - len(failed_files)}/{total_files} files")
    print(f"Total chunks created: {len(all_chunks)}")

    if failed_files:
        print(f"\nFailed files: {len(failed_files)}")
        for fail in failed_files:
            print(f"  - {fail['file']}: {fail['error'][:100]}")

    print(f"{'='*80}\n")

    # Show sample chunks
    if all_chunks:
        print("\nSample chunk (showing first chunk):")
        print("-" * 80)
        print(all_chunks[0].page_content[:500] + "...")
        print("-" * 80)
        print(f"Metadata: {all_chunks[0].metadata}")
        print()

    return all_chunks


def save_to_chroma(chunks: list[Document]):
    """Save chunks to ChromaDB with batched processing."""
    # Remove existing database
    if os.path.exists(CHROMA_PATH):
        print(f"Removing existing database at {CHROMA_PATH}...")
        shutil.rmtree(CHROMA_PATH)

    print(f"\n{'='*80}")
    print(f"Embedding and saving {len(chunks)} chunks to ChromaDB")
    print(f"{'='*80}\n")

    # Process in batches
    batch_size = 500
    total_batches = (len(chunks) + batch_size - 1) // batch_size

    db = None
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        batch_num = (i // batch_size) + 1

        print(f"[Batch {batch_num}/{total_batches}] Embedding {len(batch)} chunks...", end=" ", flush=True)

        try:
            if db is None:
                # Create database with first batch
                db = Chroma.from_documents(
                    batch, embeddings, persist_directory=CHROMA_PATH
                )
            else:
                # Add subsequent batches
                db.add_documents(batch)
            print(f"✓ ({i + len(batch)}/{len(chunks)} total)")
        except Exception as e:
            print(f"✗ FAILED: {str(e)[:100]}")
            continue

    print(f"\n{'='*80}")
    print(f"✓ Database created successfully at {CHROMA_PATH}")
    print(f"✓ Total chunks: {len(chunks)}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
