#!/usr/bin/env python3
"""ChromaDB Cleanup Script.

Removes duplicates, empty chunks, and other data quality issues
identified by inspect_chromadb.py.

Usage:
    # Dry run (show what would be deleted)
    python scripts/cleanup_chromadb.py --dry-run

    # Actually clean up
    python scripts/cleanup_chromadb.py --execute
"""

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_chroma import Chroma
from src.config import get_embedding_model


def load_chromadb(persist_directory: str = "chroma") -> Chroma:
    """Load ChromaDB without embedding function (for deletion operations)."""
    # For deletion, we don't need embedding function
    return Chroma(persist_directory=persist_directory)


def get_all_chunks(db: Chroma) -> list[dict]:
    """Get all chunks from the database."""
    collection = db._collection
    results = collection.get(include=["documents", "metadatas"])

    chunks = []
    for i, (doc, meta) in enumerate(zip(results["documents"], results["metadatas"])):
        chunks.append({
            "id": results["ids"][i],
            "content": doc,
            "metadata": meta or {},
        })
    return chunks


def find_exact_duplicates(chunks: list[dict]) -> list[str]:
    """Find IDs of exact duplicate chunks (keeps one copy)."""
    content_to_ids = {}
    for chunk in chunks:
        content = chunk["content"]
        if content not in content_to_ids:
            content_to_ids[content] = []
        content_to_ids[content].append(chunk["id"])

    # For each duplicate group, keep the first and mark rest for deletion
    duplicates_to_remove = []
    for content, ids in content_to_ids.items():
        if len(ids) > 1:
            # Keep the first, remove the rest
            duplicates_to_remove.extend(ids[1:])

    return duplicates_to_remove


def find_empty_chunks(chunks: list[dict]) -> list[str]:
    """Find IDs of empty or whitespace-only chunks."""
    return [c["id"] for c in chunks if not c["content"].strip()]


def find_very_short_chunks(chunks: list[dict], min_length: int = 50) -> list[str]:
    """Find IDs of very short chunks (likely noise)."""
    return [c["id"] for c in chunks if len(c["content"].strip()) < min_length]


def delete_chunks(db: Chroma, ids: list[str], batch_size: int = 5000) -> int:
    """Delete chunks by ID in batches."""
    collection = db._collection
    deleted = 0

    for i in range(0, len(ids), batch_size):
        batch = ids[i:i + batch_size]
        collection.delete(ids=batch)
        deleted += len(batch)
        print(f"  Deleted {deleted}/{len(ids)} chunks...")

    return deleted


def main():
    parser = argparse.ArgumentParser(description="Clean up ChromaDB")
    parser.add_argument(
        "--chroma-path",
        default="chroma",
        help="Path to ChromaDB directory (default: chroma)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without actually deleting",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually execute the deletions",
    )
    parser.add_argument(
        "--remove-duplicates",
        action="store_true",
        default=True,
        help="Remove exact duplicate chunks (default: True)",
    )
    parser.add_argument(
        "--remove-empty",
        action="store_true",
        default=True,
        help="Remove empty chunks (default: True)",
    )
    parser.add_argument(
        "--remove-short",
        action="store_true",
        default=False,
        help="Remove very short chunks (<50 chars) (default: False)",
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=50,
        help="Minimum chunk length when --remove-short is used (default: 50)",
    )
    parser.add_argument(
        "--output",
        help="Output file for cleanup report (optional)",
    )

    args = parser.parse_args()

    if not args.dry_run and not args.execute:
        print("ERROR: Must specify either --dry-run or --execute")
        print("  --dry-run: Show what would be deleted without actually deleting")
        print("  --execute: Actually execute the deletions")
        sys.exit(1)

    print(f"Loading ChromaDB from {args.chroma_path}...")
    db = load_chromadb(args.chroma_path)

    print("Fetching all chunks...")
    chunks = get_all_chunks(db)
    print(f"Found {len(chunks):,} chunks")

    # Track what to delete
    to_delete = set()
    report = {
        "timestamp": datetime.now().isoformat(),
        "mode": "dry_run" if args.dry_run else "execute",
        "initial_count": len(chunks),
        "removals": {},
    }

    # Find duplicates
    if args.remove_duplicates:
        print("\nFinding exact duplicates...")
        duplicates = find_exact_duplicates(chunks)
        print(f"  Found {len(duplicates):,} duplicate chunks")
        to_delete.update(duplicates)
        report["removals"]["duplicates"] = len(duplicates)

    # Find empty chunks
    if args.remove_empty:
        print("\nFinding empty chunks...")
        empty = find_empty_chunks(chunks)
        print(f"  Found {len(empty):,} empty chunks")
        to_delete.update(empty)
        report["removals"]["empty"] = len(empty)

    # Find very short chunks
    if args.remove_short:
        print(f"\nFinding very short chunks (<{args.min_length} chars)...")
        short = find_very_short_chunks(chunks, args.min_length)
        print(f"  Found {len(short):,} very short chunks")
        to_delete.update(short)
        report["removals"]["short"] = len(short)

    # Summary
    print("\n" + "=" * 50)
    print("CLEANUP SUMMARY")
    print("=" * 50)
    print(f"Initial chunks:  {len(chunks):,}")
    print(f"To remove:       {len(to_delete):,}")
    print(f"Final chunks:    {len(chunks) - len(to_delete):,}")
    print(f"Reduction:       {len(to_delete) / len(chunks) * 100:.1f}%")

    report["to_remove"] = len(to_delete)
    report["final_count"] = len(chunks) - len(to_delete)

    if args.dry_run:
        print("\n[DRY RUN] No changes made. Use --execute to apply.")
    elif args.execute:
        if to_delete:
            print(f"\nDeleting {len(to_delete):,} chunks...")
            deleted = delete_chunks(db, list(to_delete))
            print(f"Successfully deleted {deleted:,} chunks")
            report["deleted"] = deleted
        else:
            print("\nNo chunks to delete.")
            report["deleted"] = 0

    # Save report
    if args.output:
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nReport saved to: {args.output}")

    return report


if __name__ == "__main__":
    main()
