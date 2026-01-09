#!/usr/bin/env python3
"""ChromaDB Quality Inspection Script.

Comprehensive quality checks for the ChromaDB vector database:
- Table OCR verification
- Chunking & metadata sanity
- Database hygiene (duplicates, empty chunks, garbled text)
- Retrieval tests

Usage:
    python scripts/inspect_chromadb.py --sample-size 20 --output inspection_report.json
"""

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_chroma import Chroma


def load_chromadb(persist_directory: str = "chroma", embedding_name: str = "openai-large") -> Chroma:
    """Load the ChromaDB instance.

    Args:
        persist_directory: Path to ChromaDB
        embedding_name: Embedding model to use. Default "openai-large" matches
                       how the DB was built. Use "bge-large" for free local model
                       (but will fail retrieval if DB was built with OpenAI).
    """
    from src.config import get_embedding_model
    embeddings = get_embedding_model(embedding_name)
    return Chroma(persist_directory=persist_directory, embedding_function=embeddings)


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


def check_table_ocr(chunks: list[dict], sample_size: int = 20) -> dict[str, Any]:
    """Check table OCR quality.

    Looks for:
    - Table-like structures (pipes, dashes, aligned columns)
    - Common OCR errors (0 vs O, 1 vs l, etc.)
    - Flattened tables (columns merged into single lines)
    """
    results = {
        "total_table_chunks": 0,
        "readable_tables": 0,
        "flattened_tables": 0,
        "ocr_errors_detected": 0,
        "samples": [],
        "issues": [],
    }

    # Patterns for table detection
    table_patterns = [
        r"\|.*\|",  # Markdown tables
        r"\$[\d,]+\.?\d*",  # Dollar amounts
        r"\d{1,3}(,\d{3})+",  # Large numbers with commas
        r"^\s*\d+\s+\d+\s+\d+",  # Aligned numeric columns
    ]

    # Common OCR error patterns
    ocr_error_patterns = [
        (r"[0O][0O]", "Possible 0/O confusion"),
        (r"[1Il][1Il]", "Possible 1/l/I confusion"),
        (r"\bI\d+\b", "Possible l→I substitution in numbers"),
        (r"[^\w\s]{3,}", "Possible garbled characters"),
    ]

    table_chunks = []
    for chunk in chunks:
        content = chunk["content"]
        is_table = any(re.search(p, content, re.MULTILINE) for p in table_patterns)

        if is_table:
            table_chunks.append(chunk)

    results["total_table_chunks"] = len(table_chunks)

    # Sample and analyze table chunks
    import random
    sampled = random.sample(table_chunks, min(sample_size, len(table_chunks)))

    for chunk in sampled:
        content = chunk["content"]
        analysis = {
            "id": chunk["id"],
            "preview": content[:300] + "..." if len(content) > 300 else content,
            "length": len(content),
            "has_pipe_tables": bool(re.search(r"\|.*\|.*\|", content)),
            "has_numbers": bool(re.search(r"\d+", content)),
            "issues": [],
        }

        # Check for OCR errors
        for pattern, desc in ocr_error_patterns:
            if re.search(pattern, content):
                analysis["issues"].append(desc)
                results["ocr_errors_detected"] += 1

        # Check for flattened tables (multiple numbers on same line without structure)
        lines = content.split("\n")
        flat_lines = sum(1 for line in lines if len(re.findall(r"\d+", line)) > 5 and "|" not in line)
        if flat_lines > 3:
            analysis["issues"].append("Possible flattened table structure")
            results["flattened_tables"] += 1
        else:
            results["readable_tables"] += 1

        results["samples"].append(analysis)

    return results


def check_chunking_metadata(chunks: list[dict]) -> dict[str, Any]:
    """Check chunking quality and metadata completeness."""
    results = {
        "total_chunks": len(chunks),
        "chunk_size_stats": {},
        "metadata_completeness": {},
        "boundary_issues": 0,
        "empty_chunks": 0,
        "issues": [],
    }

    # Chunk size analysis
    sizes = [len(c["content"]) for c in chunks]
    results["chunk_size_stats"] = {
        "min": min(sizes) if sizes else 0,
        "max": max(sizes) if sizes else 0,
        "mean": sum(sizes) / len(sizes) if sizes else 0,
        "median": sorted(sizes)[len(sizes) // 2] if sizes else 0,
    }

    # Check for too short or too long chunks
    too_short = sum(1 for s in sizes if s < 100)
    too_long = sum(1 for s in sizes if s > 4000)

    if too_short > 0:
        results["issues"].append(f"{too_short} chunks are very short (<100 chars)")
    if too_long > 0:
        results["issues"].append(f"{too_long} chunks are very long (>4000 chars)")

    # Metadata completeness
    metadata_fields = Counter()
    for chunk in chunks:
        for key in chunk["metadata"].keys():
            metadata_fields[key] += 1

    results["metadata_completeness"] = {
        field: {
            "count": count,
            "percentage": round(count / len(chunks) * 100, 2),
        }
        for field, count in metadata_fields.most_common()
    }

    # Check for empty chunks
    results["empty_chunks"] = sum(1 for c in chunks if not c["content"].strip())

    # Check for boundary issues (mid-sentence splits)
    boundary_issues = 0
    for chunk in chunks:
        content = chunk["content"].strip()
        # Check if chunk starts mid-sentence (lowercase, no capital)
        if content and content[0].islower() and not content.startswith(("e.g.", "i.e.")):
            boundary_issues += 1
        # Check if chunk ends mid-sentence
        if content and not content.endswith((".", "!", "?", ":", '"', "'", ")", "]")):
            boundary_issues += 1

    results["boundary_issues"] = boundary_issues
    if boundary_issues > len(chunks) * 0.3:
        results["issues"].append(f"High rate of boundary issues ({boundary_issues} occurrences)")

    return results


def check_database_hygiene(chunks: list[dict]) -> dict[str, Any]:
    """Check for duplicates, garbled text, and boilerplate."""
    results = {
        "duplicates": {"exact": 0, "near": 0, "examples": []},
        "garbled_chunks": 0,
        "boilerplate_chunks": 0,
        "document_coverage": {},
        "issues": [],
    }

    # Check for exact duplicates
    content_hashes = Counter()
    for chunk in chunks:
        content_hash = hash(chunk["content"])
        content_hashes[content_hash] += 1

    exact_duplicates = sum(count - 1 for count in content_hashes.values() if count > 1)
    results["duplicates"]["exact"] = exact_duplicates

    if exact_duplicates > 0:
        results["issues"].append(f"{exact_duplicates} exact duplicate chunks found")

    # Check for garbled text
    garbled_pattern = re.compile(r"[^\x00-\x7F]{5,}|[\x00-\x1F]{3,}")
    garbled_chunks = []
    for chunk in chunks:
        if garbled_pattern.search(chunk["content"]):
            garbled_chunks.append(chunk["id"])

    results["garbled_chunks"] = len(garbled_chunks)
    if garbled_chunks:
        results["issues"].append(f"{len(garbled_chunks)} chunks with garbled text")

    # Check for boilerplate (repeated headers, footers, etc.)
    first_100_chars = Counter()
    for chunk in chunks:
        prefix = chunk["content"][:100]
        first_100_chars[prefix] += 1

    boilerplate_prefixes = {k: v for k, v in first_100_chars.items() if v > 10}
    results["boilerplate_chunks"] = sum(boilerplate_prefixes.values())

    if boilerplate_prefixes:
        results["issues"].append(f"{len(boilerplate_prefixes)} repeated prefixes (possible boilerplate)")
        results["duplicates"]["examples"] = [
            {"prefix": k[:50] + "...", "count": v}
            for k, v in sorted(boilerplate_prefixes.items(), key=lambda x: -x[1])[:5]
        ]

    # Document coverage
    source_field = None
    for field in ["source", "doc_name", "filename", "file"]:
        if any(field in c["metadata"] for c in chunks):
            source_field = field
            break

    if source_field:
        doc_counts = Counter()
        for chunk in chunks:
            doc = chunk["metadata"].get(source_field, "unknown")
            # Extract just the filename if it's a path
            if "/" in str(doc):
                doc = str(doc).split("/")[-1]
            doc_counts[doc] += 1

        results["document_coverage"] = {
            "total_documents": len(doc_counts),
            "chunks_per_doc": {
                "min": min(doc_counts.values()),
                "max": max(doc_counts.values()),
                "mean": sum(doc_counts.values()) / len(doc_counts),
            },
            "top_10": dict(doc_counts.most_common(10)),
            "bottom_10": dict(doc_counts.most_common()[-10:]),
        }

    return results


def run_retrieval_tests(db: Chroma) -> dict[str, Any]:
    """Run sample retrieval queries to verify database functionality."""
    results = {
        "tests": [],
        "passed": 0,
        "failed": 0,
    }

    # Test queries covering different types
    test_queries = [
        {
            "name": "Table query - revenue",
            "query": "What was the total revenue for fiscal year 2022?",
            "expected_patterns": [r"\$", r"revenue", r"\d+"],
        },
        {
            "name": "Table query - specific company",
            "query": "3M operating income 2018",
            "expected_patterns": [r"3M|3m", r"operat", r"\d+"],
        },
        {
            "name": "Text query - business description",
            "query": "Describe the company's main business operations and segments",
            "expected_patterns": [r"business|operation|segment"],
        },
        {
            "name": "Numeric query - percentage",
            "query": "What was the gross profit margin percentage?",
            "expected_patterns": [r"%|percent|margin"],
        },
        {
            "name": "Time-specific query",
            "query": "Q4 2021 quarterly results",
            "expected_patterns": [r"Q4|quarter|2021"],
        },
    ]

    for test in test_queries:
        try:
            docs = db.similarity_search(test["query"], k=5)

            # Check if results match expected patterns
            combined_text = " ".join(d.page_content for d in docs)
            matches = sum(
                1 for pattern in test["expected_patterns"]
                if re.search(pattern, combined_text, re.IGNORECASE)
            )

            passed = matches >= len(test["expected_patterns"]) // 2 + 1

            results["tests"].append({
                "name": test["name"],
                "query": test["query"],
                "passed": passed,
                "results_count": len(docs),
                "pattern_matches": matches,
                "expected_patterns": len(test["expected_patterns"]),
                "top_result_preview": docs[0].page_content[:200] if docs else "No results",
            })

            if passed:
                results["passed"] += 1
            else:
                results["failed"] += 1

        except Exception as e:
            results["tests"].append({
                "name": test["name"],
                "query": test["query"],
                "passed": False,
                "error": str(e),
            })
            results["failed"] += 1

    return results


def generate_report(
    table_results: dict,
    chunking_results: dict,
    hygiene_results: dict,
    retrieval_results: dict,
) -> dict[str, Any]:
    """Generate comprehensive inspection report."""

    # Collect all issues
    all_issues = []
    all_issues.extend(table_results.get("issues", []))
    all_issues.extend(chunking_results.get("issues", []))
    all_issues.extend(hygiene_results.get("issues", []))

    if retrieval_results["failed"] > 0:
        all_issues.append(f"{retrieval_results['failed']} retrieval tests failed")

    # Calculate overall health score
    total_checks = 10
    issues_count = len(all_issues)
    health_score = max(0, (total_checks - issues_count) / total_checks * 100)

    return {
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "total_chunks": chunking_results["total_chunks"],
            "health_score": round(health_score, 1),
            "issues_found": len(all_issues),
            "retrieval_tests": f"{retrieval_results['passed']}/{retrieval_results['passed'] + retrieval_results['failed']} passed",
        },
        "issues": all_issues,
        "table_ocr": table_results,
        "chunking_metadata": chunking_results,
        "database_hygiene": hygiene_results,
        "retrieval_tests": retrieval_results,
        "recommendations": generate_recommendations(all_issues, chunking_results, hygiene_results),
    }


def generate_recommendations(
    issues: list[str],
    chunking_results: dict,
    hygiene_results: dict,
) -> list[str]:
    """Generate actionable recommendations based on findings."""
    recommendations = []

    if hygiene_results["duplicates"]["exact"] > 0:
        recommendations.append(
            f"Remove {hygiene_results['duplicates']['exact']} duplicate chunks to improve retrieval quality"
        )

    if chunking_results["empty_chunks"] > 0:
        recommendations.append(
            f"Filter out {chunking_results['empty_chunks']} empty chunks"
        )

    if hygiene_results["garbled_chunks"] > 0:
        recommendations.append(
            f"Review and potentially re-ingest documents with garbled text ({hygiene_results['garbled_chunks']} chunks affected)"
        )

    if hygiene_results["boilerplate_chunks"] > 100:
        recommendations.append(
            "Consider filtering repetitive boilerplate content (headers, footers, disclaimers)"
        )

    if chunking_results["chunk_size_stats"]["max"] > 4000:
        recommendations.append(
            "Some chunks are very large (>4000 chars). Consider re-chunking with smaller max size."
        )

    if not recommendations:
        recommendations.append("Database appears healthy. Ready for evaluation.")

    return recommendations


def print_summary(report: dict) -> None:
    """Print a human-readable summary."""
    print("\n" + "=" * 60)
    print("CHROMADB INSPECTION REPORT")
    print("=" * 60)

    summary = report["summary"]
    print(f"\nTotal Chunks: {summary['total_chunks']:,}")
    print(f"Health Score: {summary['health_score']}%")
    print(f"Issues Found: {summary['issues_found']}")
    print(f"Retrieval Tests: {summary['retrieval_tests']}")

    if report["issues"]:
        print("\n" + "-" * 40)
        print("ISSUES DETECTED:")
        for i, issue in enumerate(report["issues"], 1):
            print(f"  {i}. {issue}")

    print("\n" + "-" * 40)
    print("RECOMMENDATIONS:")
    for i, rec in enumerate(report["recommendations"], 1):
        print(f"  {i}. {rec}")

    # Chunk size stats
    stats = report["chunking_metadata"]["chunk_size_stats"]
    print("\n" + "-" * 40)
    print("CHUNK SIZE STATISTICS:")
    print(f"  Min: {stats['min']:,} chars")
    print(f"  Max: {stats['max']:,} chars")
    print(f"  Mean: {stats['mean']:,.0f} chars")
    print(f"  Median: {stats['median']:,} chars")

    # Metadata coverage
    print("\n" + "-" * 40)
    print("METADATA FIELDS:")
    for field, info in list(report["chunking_metadata"]["metadata_completeness"].items())[:5]:
        print(f"  {field}: {info['percentage']}% coverage")

    # Document coverage
    if report["database_hygiene"]["document_coverage"]:
        coverage = report["database_hygiene"]["document_coverage"]
        print("\n" + "-" * 40)
        print("DOCUMENT COVERAGE:")
        print(f"  Total Documents: {coverage['total_documents']}")
        print(f"  Chunks per Doc: {coverage['chunks_per_doc']['min']}-{coverage['chunks_per_doc']['max']} (mean: {coverage['chunks_per_doc']['mean']:.0f})")

    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Inspect ChromaDB quality")
    parser.add_argument(
        "--chroma-path",
        default="chroma",
        help="Path to ChromaDB directory (default: chroma)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=20,
        help="Number of chunks to sample for detailed analysis (default: 20)",
    )
    parser.add_argument(
        "--output",
        help="Output file for JSON report (optional)",
    )
    parser.add_argument(
        "--skip-retrieval",
        action="store_true",
        help="Skip retrieval tests (faster)",
    )
    parser.add_argument(
        "--embedding",
        default="openai-large",
        help="Embedding model to use for retrieval tests (default: openai-large). "
             "Must match what ChromaDB was built with.",
    )

    args = parser.parse_args()

    print(f"Loading ChromaDB from {args.chroma_path}...")
    print(f"Using embedding model: {args.embedding}")
    db = load_chromadb(args.chroma_path, args.embedding)

    print("Fetching all chunks...")
    chunks = get_all_chunks(db)
    print(f"Found {len(chunks):,} chunks")

    print("\nRunning inspections...")

    print("  - Checking table OCR quality...")
    table_results = check_table_ocr(chunks, args.sample_size)

    print("  - Checking chunking and metadata...")
    chunking_results = check_chunking_metadata(chunks)

    print("  - Checking database hygiene...")
    hygiene_results = check_database_hygiene(chunks)

    if not args.skip_retrieval:
        print("  - Running retrieval tests...")
        retrieval_results = run_retrieval_tests(db)
    else:
        retrieval_results = {"tests": [], "passed": 0, "failed": 0}

    print("\nGenerating report...")
    report = generate_report(table_results, chunking_results, hygiene_results, retrieval_results)

    # Print summary
    print_summary(report)

    # Save JSON report if requested
    if args.output:
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\nFull report saved to: {args.output}")

    return report


if __name__ == "__main__":
    main()
