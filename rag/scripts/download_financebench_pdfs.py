"""
Download all FinanceBench PDFs from GitHub repository.

Usage:
    python scripts/download_financebench_pdfs.py

This will download ~368 PDFs (~2-3 GB total) to data/test_files/finance-bench-pdfs/
"""

import os
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Configuration
GITHUB_API_URL = "https://api.github.com/repos/patronus-ai/financebench/contents/pdfs"
RAW_URL_BASE = "https://raw.githubusercontent.com/patronus-ai/financebench/main/pdfs"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "test_files" / "finance-bench-pdfs"
MAX_WORKERS = 5  # Concurrent downloads


def get_pdf_list():
    """Fetch list of PDF files from GitHub API."""
    print("Fetching PDF list from GitHub...")

    response = requests.get(GITHUB_API_URL)
    if response.status_code != 200:
        print(f"Error fetching file list: {response.status_code}")
        print("Trying alternative method...")
        return get_pdf_list_from_dataset()

    files = response.json()
    pdf_files = [f['name'] for f in files if f['name'].endswith('.pdf')]
    print(f"Found {len(pdf_files)} PDF files")
    return pdf_files


def get_pdf_list_from_dataset():
    """Alternative: Get PDF list from HuggingFace dataset."""
    try:
        from datasets import load_dataset
        print("Loading dataset from HuggingFace...")
        dataset = load_dataset("PatronusAI/financebench", split="train")

        # Extract unique doc names
        doc_names = set()
        for item in dataset:
            doc_name = item.get('doc_name', '')
            if doc_name:
                doc_names.add(f"{doc_name}.pdf")

        print(f"Found {len(doc_names)} unique documents in dataset")
        return list(doc_names)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return []


def download_pdf(filename: str, output_dir: Path) -> tuple:
    """Download a single PDF file."""
    url = f"{RAW_URL_BASE}/{filename}"
    output_path = output_dir / filename

    # Skip if already exists
    if output_path.exists():
        return filename, "skipped", 0

    try:
        response = requests.get(url, timeout=60)
        if response.status_code == 200:
            output_path.write_bytes(response.content)
            return filename, "success", len(response.content)
        else:
            return filename, f"error:{response.status_code}", 0
    except Exception as e:
        return filename, f"error:{str(e)[:50]}", 0


def main():
    """Main download function."""
    print("=" * 60)
    print("FinanceBench PDF Downloader")
    print("=" * 60)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")

    # Get list of PDFs
    pdf_files = get_pdf_list()

    if not pdf_files:
        print("No PDF files found. Exiting.")
        return

    # Check existing files
    existing = set(f.name for f in OUTPUT_DIR.glob("*.pdf"))
    to_download = [f for f in pdf_files if f not in existing]

    print(f"\nTotal PDFs: {len(pdf_files)}")
    print(f"Already downloaded: {len(existing)}")
    print(f"To download: {len(to_download)}")

    if not to_download:
        print("\nAll files already downloaded!")
        return

    # Download with progress bar
    print(f"\nDownloading {len(to_download)} files with {MAX_WORKERS} workers...")

    success_count = 0
    error_count = 0
    total_bytes = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(download_pdf, filename, OUTPUT_DIR): filename
            for filename in to_download
        }

        with tqdm(total=len(to_download), desc="Downloading") as pbar:
            for future in as_completed(futures):
                filename, status, size = future.result()

                if status == "success":
                    success_count += 1
                    total_bytes += size
                elif status == "skipped":
                    pass
                else:
                    error_count += 1
                    tqdm.write(f"  Failed: {filename} ({status})")

                pbar.update(1)

    # Summary
    print("\n" + "=" * 60)
    print("DOWNLOAD COMPLETE")
    print("=" * 60)
    print(f"Successfully downloaded: {success_count}")
    print(f"Failed: {error_count}")
    print(f"Total size: {total_bytes / 1024 / 1024:.1f} MB")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
