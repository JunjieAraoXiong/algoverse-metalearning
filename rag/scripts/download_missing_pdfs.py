#!/usr/bin/env python3
"""
Download missing PDFs for FinanceBench evaluation.
These 23 PDFs affect 36 questions (24% of the test set).
"""

import json
import os
import sys
import time
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_DIR = Path(__file__).parent.parent
PDF_DIR = BASE_DIR / 'data/test_files/finance-bench-pdfs'

# Load doc info
doc_links = {}
with open(BASE_DIR / 'data/question_sets/financebench_document_information.jsonl') as f:
    for line in f:
        doc = json.loads(line)
        doc_links[doc['doc_name']] = doc.get('doc_link', '')

# Missing docs that affect questions
MISSING_DOCS = [
    "ACTIVISIONBLIZZARD_2019_10K",
    "ADOBE_2015_10K",
    "ADOBE_2016_10K",
    "ADOBE_2017_10K",
    "ADOBE_2022_10K",
    "AMD_2015_10K",
    "FOOTLOCKER_2022_8K_dated-2022-05-20",
    "FOOTLOCKER_2022_8K_dated_2022-08-19",
    "JOHNSON_JOHNSON_2022Q4_EARNINGS",
    "JOHNSON_JOHNSON_2022_10K",
    "JOHNSON_JOHNSON_2023Q2_EARNINGS",
    "JOHNSON_JOHNSON_2023_8K_dated-2023-08-30",
    "KRAFTHEINZ_2019_10K",
    "LOCKHEEDMARTIN_2020_10K",
    "LOCKHEEDMARTIN_2021_10K",
    "LOCKHEEDMARTIN_2022_10K",
    "MICROSOFT_2016_10K",
    "MICROSOFT_2023_10K",
    "PEPSICO_2021_10K",
    "PEPSICO_2022_10K",
    "PEPSICO_2023Q1_EARNINGS",
    "PEPSICO_2023_8K_dated-2023-05-05",
    "PEPSICO_2023_8K_dated-2023-05-30",
]


def download_pdf(doc_name: str) -> tuple[str, bool, str]:
    """Download a single PDF. Returns (name, success, message)."""
    url = doc_links.get(doc_name, '')
    if not url:
        return doc_name, False, "No download URL"

    output_path = PDF_DIR / f"{doc_name}.pdf"
    if output_path.exists():
        return doc_name, True, "Already exists"

    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=60, allow_redirects=True)

        # Check if we got a PDF
        content_type = response.headers.get('Content-Type', '')
        if response.status_code == 200:
            if 'pdf' in content_type.lower() or url.endswith('.pdf') or response.content[:4] == b'%PDF':
                with open(output_path, 'wb') as f:
                    f.write(response.content)
                return doc_name, True, f"Downloaded ({len(response.content)/1024/1024:.1f} MB)"
            else:
                return doc_name, False, f"Not a PDF (got {content_type[:50]})"
        else:
            return doc_name, False, f"HTTP {response.status_code}"
    except Exception as e:
        return doc_name, False, str(e)[:50]


def main():
    print("=" * 70)
    print("DOWNLOADING MISSING PDFs")
    print("=" * 70)
    print(f"Missing documents: {len(MISSING_DOCS)}")
    print(f"Output directory:  {PDF_DIR}")
    print("=" * 70)

    start = time.time()
    success = 0
    failed = []

    # Download with thread pool
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(download_pdf, doc): doc for doc in MISSING_DOCS}

        for future in as_completed(futures):
            doc_name, ok, msg = future.result()
            status = "✓" if ok else "✗"
            print(f"  {status} {doc_name}: {msg}")

            if ok:
                success += 1
            else:
                failed.append((doc_name, msg))

    print()
    print("=" * 70)
    print("DOWNLOAD COMPLETE")
    print("=" * 70)
    print(f"Success: {success}/{len(MISSING_DOCS)}")
    print(f"Failed:  {len(failed)}")
    print(f"Time:    {time.time() - start:.1f}s")

    if failed:
        print()
        print("Failed downloads (may need manual download):")
        for doc, msg in failed:
            url = doc_links.get(doc, '')
            print(f"  - {doc}")
            print(f"    URL: {url}")
            print(f"    Error: {msg}")


if __name__ == "__main__":
    main()
