#!/usr/bin/env python3
"""
dsRAG-Style Parallel ChromaDB Builder (Multi-GPU Support).

This is an optimized version of build_dsrag_style_chromadb.py that processes
PDFs in parallel across multiple GPUs. Each GPU handles a subset of PDFs,
with all chunks collected and written to ChromaDB at the end.

Key optimizations:
1. Multi-GPU parallelism (8x speedup on Docling PDF parsing with 8 GPUs)
2. H100 optimizations (TF32, BF16 for faster inference)
3. Process-level isolation (no CUDA conflicts)
4. Skip-file support for recovery runs (skip corrupted/problematic PDFs)

Usage:
    python scripts/build_dsrag_parallel.py \
        --domain finance \
        --input-dir data/test_files/finance-bench-pdfs \
        --output-dir chroma_dsrag_finance \
        --num-workers 8

    # Recovery run with skip list:
    python scripts/build_dsrag_parallel.py \
        --domain finance \
        --input-dir data/test_files/finance-bench-pdfs \
        --output-dir chroma_dsrag_finance \
        --num-workers 8 \
        --skip-files "AMD_2015_10K.pdf,AMCOR_2022_8K_dated-2022-04-26.pdf"

Expected time: ~3 hours with 8 GPUs (vs ~24 hours sequential)
"""

import argparse
import gc
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from multiprocessing import Pool, Manager, get_context
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Setup paths
BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

from dotenv import load_dotenv
load_dotenv()


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class DsRAGConfig:
    """Configuration for dsRAG-style ingestion."""
    embedding_model: str = "cohere-v3"
    autocontext_model: str = "gpt-4o-mini"
    autocontext_temperature: float = 0.0
    chunk_size: int = 2500
    chunk_overlap: int = 200
    batch_size: int = 10
    chroma_batch_size: int = 500
    skip_autocontext: bool = False


# =============================================================================
# H100 GPU Optimizations (Hopper Architecture)
# =============================================================================

def setup_h100_optimizations():
    """
    Enable H100-specific optimizations for faster inference.

    H100 Hopper architecture supports:
    - TF32: 2x faster than FP32 for matrix ops
    - FP8: 4x faster (not used here due to model compatibility)
    - Flash Attention 2: Memory-efficient attention
    - CUDA Graphs: Reduced kernel launch overhead
    """
    try:
        import torch

        # ========================================
        # 1. TF32 Precision (2x speedup on matmul)
        # ========================================
        # H100 native support for TensorFloat-32
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # NOTE: Do NOT set default dtype to bfloat16 here!
        # Docling's RT-DETR model requires consistent dtypes

        # ========================================
        # 2. Disable Gradient Computation
        # ========================================
        torch.set_grad_enabled(False)

        # ========================================
        # 3. cuDNN Optimizations
        # ========================================
        # Enable cuDNN autotuner for convolutions
        torch.backends.cudnn.benchmark = True
        # Use deterministic algorithms where possible
        torch.backends.cudnn.deterministic = False

        # ========================================
        # 4. Memory Optimizations
        # ========================================
        # Enable memory-efficient attention if available (Flash Attention)
        if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
            torch.backends.cuda.enable_flash_sdp(True)
        if hasattr(torch.backends.cuda, 'enable_mem_efficient_sdp'):
            torch.backends.cuda.enable_mem_efficient_sdp(True)

        # Set memory allocator for better fragmentation handling
        if hasattr(torch.cuda, 'memory'):
            # Use expandable segments for less fragmentation
            os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

        # ========================================
        # 5. CUDA Stream Optimization
        # ========================================
        # Ensure CUDA operations are properly synchronized
        if torch.cuda.is_available():
            torch.cuda.set_device(0)  # Use first visible GPU
            # Warm up CUDA
            _ = torch.zeros(1, device='cuda')

        return True

    except Exception as e:
        print(f"  Note: Could not enable all H100 optimizations: {e}")
        return False


# =============================================================================
# LLM AutoContext Generation (ASYNC PARALLEL VERSION for speed)
# =============================================================================

import asyncio

def get_llm_client():
    """Get OpenAI client for AutoContext generation."""
    from openai import OpenAI
    return OpenAI()


def get_async_llm_client():
    """Get async OpenAI client for parallel AutoContext generation."""
    from openai import AsyncOpenAI
    return AsyncOpenAI()


def generate_document_context(
    first_chunks: List[str],
    metadata: Dict[str, Any],
    model: str = "gpt-4o-mini"
) -> Tuple[str, str]:
    """Generate document title and summary using LLM."""
    context = "\n\n".join(first_chunks[:3])[:4000]

    meta_hint = ""
    if metadata.get("company"):
        meta_hint += f"Company: {metadata['company']}\n"
    if metadata.get("year"):
        meta_hint += f"Year: {metadata['year']}\n"
    if metadata.get("doc_type"):
        meta_hint += f"Document Type: {metadata['doc_type']}\n"

    prompt = f"""You are analyzing a financial/legal document. Based on the metadata and content below, generate:
1. A concise document title (e.g., "Apple Inc. 2023 Annual Report (10-K)")
2. A one-sentence summary of what this document covers

Metadata:
{meta_hint}

Beginning of document:
{context}

Respond in JSON format:
{{"title": "...", "summary": "..."}}"""

    client = get_llm_client()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=200,
    )

    try:
        result = json.loads(response.choices[0].message.content)
        return result.get("title", "Unknown Document"), result.get("summary", "")
    except (json.JSONDecodeError, KeyError):
        fallback_title = f"{metadata.get('company', 'Unknown')} {metadata.get('doc_type', 'Document')} {metadata.get('year', '')}"
        return fallback_title.strip(), ""


async def _process_single_batch_async(
    client,
    batch: List[str],
    batch_idx: int,
    doc_title: str,
    model: str,
    semaphore: asyncio.Semaphore
) -> Tuple[int, List[str]]:
    """Process a single batch of chunks asynchronously."""
    async with semaphore:  # Limit concurrent API calls
        excerpts = []
        for j, content in enumerate(batch):
            truncated = content[:1500]
            excerpts.append(f"[CHUNK {j+1}]\n{truncated}")

        batch_prompt = f"""You are analyzing excerpts from "{doc_title}".

For each chunk below, write a single brief sentence describing what that section is about.
Focus on the specific topic or metric being discussed.
Return exactly {len(batch)} numbered summaries, one per line.

{chr(10).join(excerpts)}

Summaries (one sentence each):"""

        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": batch_prompt}],
                temperature=0.0,
                max_tokens=50 * len(batch),
            )

            response_text = response.choices[0].message.content.strip()
            lines = [line.strip() for line in response_text.split('\n') if line.strip()]

            summaries = []
            for line in lines:
                cleaned = re.sub(r'^[\d]+[\.\):\s]+', '', line).strip()
                if cleaned:
                    summaries.append(cleaned)

            # Pad or trim to exact batch size
            while len(summaries) < len(batch):
                summaries.append("General content section.")
            summaries = summaries[:len(batch)]

            return (batch_idx, summaries)

        except Exception as e:
            print(f"    Batch {batch_idx} failed: {e}, using fallback")
            return (batch_idx, ["General content section."] * len(batch))


async def _generate_section_contexts_async(
    chunks: List[str],
    doc_title: str,
    model: str = "gpt-4o-mini",
    batch_size: int = 10,
    max_concurrent: int = 5  # Limit concurrent API calls to avoid rate limits
) -> List[str]:
    """Generate section summaries using PARALLEL async LLM calls."""
    if not chunks:
        return []

    client = get_async_llm_client()
    semaphore = asyncio.Semaphore(max_concurrent)

    # Create batches
    batches = []
    for i in range(0, len(chunks), batch_size):
        batches.append(chunks[i:i + batch_size])

    # Process all batches in parallel
    tasks = [
        _process_single_batch_async(client, batch, idx, doc_title, model, semaphore)
        for idx, batch in enumerate(batches)
    ]

    results = await asyncio.gather(*tasks)

    # Sort by batch index and flatten
    results.sort(key=lambda x: x[0])
    all_summaries = []
    for _, summaries in results:
        all_summaries.extend(summaries)

    return all_summaries


def generate_section_contexts_batch(
    chunks: List[str],
    doc_title: str,
    model: str = "gpt-4o-mini",
    batch_size: int = 10
) -> List[str]:
    """
    Generate section summaries for multiple chunks using PARALLEL async LLM calls.

    This is 5-10x faster than sequential calls while maintaining the same quality
    (batch_size=10 is preserved for quality).
    """
    if not chunks:
        return []

    # Run async function from sync context
    return asyncio.run(_generate_section_contexts_async(
        chunks, doc_title, model, batch_size, max_concurrent=5
    ))


def build_autocontext_header(
    doc_title: str,
    doc_summary: str,
    section_summary: str,
    element_type: str = "prose"
) -> str:
    """Build the dsRAG-style AutoContext header."""
    parts = []
    parts.append(f"Document context: the following excerpt is from a document titled '{doc_title}'.")
    if doc_summary:
        parts.append(f"This document is about: {doc_summary}")
    if section_summary:
        parts.append(f"\nSection context: {section_summary}")
    if element_type == "table":
        parts.append("\nContent type: This excerpt contains tabular data.")
    return "\n".join(parts)


def build_rule_based_header(metadata: Dict[str, Any], chunk_content: str) -> str:
    """Build rule-based AutoContext header (fallback, no LLM calls)."""
    company = metadata.get('company', 'Unknown')
    doc_type = metadata.get('doc_type', 'Document')
    year = metadata.get('year', '')
    fiscal_period = metadata.get('fiscal_period', '')

    if fiscal_period:
        doc_title = f"{company} {doc_type} {fiscal_period}"
    elif year:
        doc_title = f"{company} {doc_type} {year}"
    else:
        doc_title = f"{company} {doc_type}"

    headers = []
    for line in chunk_content.split('\n')[:10]:
        line = line.strip()
        if line.startswith('##'):
            header = re.sub(r'^#+\s*', '', line).strip()
            if header and len(header) > 2:
                headers.append(header)
                if len(headers) >= 2:
                    break

    parts = [f"Document: {doc_title}"]
    if headers:
        parts.append(f"Section: {' > '.join(headers)}")

    element_type = metadata.get('element_type', 'prose')
    if element_type == 'table':
        parts.append("Content Type: Table")

    return "\n".join(parts)


# =============================================================================
# Error Logging for Failed PDFs
# =============================================================================

def log_failed_pdf(error_log_path: Path, pdf_name: str, error: str, gpu_id: int):
    """
    Append failed PDF info to JSONL error log.

    This enables tracking problematic PDFs for later review and re-ingestion.
    Using JSONL format allows safe concurrent appends from multiple workers.

    Args:
        error_log_path: Path to the error log file
        pdf_name: Name of the failed PDF file
        error: Error message describing the failure
        gpu_id: GPU worker ID that encountered the error
    """
    from datetime import datetime

    entry = {
        "filename": pdf_name,
        "error": str(error),
        "gpu_id": gpu_id,
        "timestamp": datetime.now().isoformat(),
    }

    # Use 'a' mode for atomic append - safe for concurrent writes
    with open(error_log_path, 'a') as f:
        f.write(json.dumps(entry) + '\n')


def get_failed_pdfs_count(checkpoint_dir: Path) -> int:
    """Get count of failed PDFs from error log."""
    error_log = checkpoint_dir / "failed_pdfs.jsonl"
    if not error_log.exists():
        return 0

    count = 0
    with open(error_log, 'r') as f:
        for line in f:
            if line.strip():
                count += 1
    return count


# =============================================================================
# PDF Processing (worker function)
# =============================================================================

def process_single_pdf(args: Tuple) -> List[Dict]:
    """
    Process a single PDF on the assigned GPU with incremental checkpointing.

    This is the worker function that runs in a separate process.
    Each worker is assigned a specific GPU via CUDA_VISIBLE_DEVICES.

    IMPORTANT: Chunks are saved to a checkpoint file immediately after processing.
    This ensures no work is lost if the job crashes or hangs.

    Args:
        args: Tuple of (pdf_path, gpu_id, config_dict, worker_id, total_for_worker, checkpoint_dir)

    Returns:
        List of chunk dictionaries (serializable for multiprocessing)
    """
    pdf_path_str, gpu_id, config_dict, worker_id, total_for_worker, checkpoint_dir = args
    pdf_path = Path(pdf_path_str)
    checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None

    # Set GPU for this worker
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    # Enable H100 optimizations
    setup_h100_optimizations()

    # Reconstruct config
    config = DsRAGConfig(**config_dict)

    # Import here to avoid CUDA initialization issues
    from langchain_core.documents import Document

    try:
        # Docling parsing (uses GPU)
        from docling.document_converter import DocumentConverter
        from src.metadata_utils import parse_filename

        # Parse filename for metadata
        file_meta = parse_filename(pdf_path.name)
        if file_meta:
            file_meta_dict = file_meta.to_dict()
        else:
            file_meta_dict = {"source_file": pdf_path.name}

        # Convert PDF with Docling
        converter = DocumentConverter()
        result = converter.convert(str(pdf_path))
        markdown = result.document.export_to_markdown()

    except Exception as e:
        print(f"  [GPU{gpu_id}] Docling error on {pdf_path.name}: {e}")
        return []

    # Table-aware chunking with chunk_index tracking for RSE adjacency detection
    chunks = []
    current_chunk = ""
    current_type = "prose"
    in_table = False
    chunk_size = config.chunk_size
    chunk_index = 0  # Sequential index for RSE to detect adjacency

    for line in markdown.split('\n'):
        line_stripped = line.strip()

        if line_stripped.startswith('|') and not in_table:
            if current_chunk.strip():
                chunks.append({
                    'content': current_chunk.strip(),
                    'metadata': {
                        **file_meta_dict,
                        'element_type': 'prose',
                        'source': str(pdf_path),
                        'chunk_index': chunk_index,
                    }
                })
                chunk_index += 1
            current_chunk = line + '\n'
            current_type = "table"
            in_table = True

        elif in_table:
            if line_stripped.startswith('|') or (line_stripped.startswith('-') and '|' in current_chunk):
                current_chunk += line + '\n'
            else:
                if current_chunk.strip():
                    chunks.append({
                        'content': current_chunk.strip(),
                        'metadata': {
                            **file_meta_dict,
                            'element_type': 'table',
                            'source': str(pdf_path),
                            'chunk_index': chunk_index,
                        }
                    })
                    chunk_index += 1
                current_chunk = line + '\n'
                current_type = "prose"
                in_table = False

        else:
            current_chunk += line + '\n'
            if len(current_chunk) > chunk_size:
                break_point = current_chunk.rfind('\n\n', 0, chunk_size)
                if break_point == -1:
                    break_point = current_chunk.rfind('. ', 0, chunk_size)
                if break_point == -1:
                    break_point = chunk_size

                chunk_text = current_chunk[:break_point].strip()
                if chunk_text:
                    chunks.append({
                        'content': chunk_text,
                        'metadata': {
                            **file_meta_dict,
                            'element_type': 'prose',
                            'source': str(pdf_path),
                            'chunk_index': chunk_index,
                        }
                    })
                    chunk_index += 1
                current_chunk = current_chunk[break_point:].strip() + '\n'

    if current_chunk.strip():
        chunks.append({
            'content': current_chunk.strip(),
            'metadata': {
                **file_meta_dict,
                'element_type': current_type,
                'source': str(pdf_path),
                'chunk_index': chunk_index,
            }
        })

    # Add AutoContext
    if config.skip_autocontext:
        # Fast path: rule-based headers
        for chunk in chunks:
            header = build_rule_based_header(chunk['metadata'], chunk['content'])
            chunk['content'] = f"{header}\n\n{chunk['content']}"
            chunk['metadata']['has_autocontext'] = True
    else:
        # LLM-generated AutoContext
        if chunks:
            # Generate document context (once)
            first_contents = [c['content'] for c in chunks[:5]]
            doc_title, doc_summary = generate_document_context(
                first_contents,
                file_meta_dict,
                model=config.autocontext_model
            )

            # Generate section summaries in batches
            chunk_contents = [c['content'] for c in chunks]
            section_summaries = generate_section_contexts_batch(
                chunk_contents,
                doc_title,
                model=config.autocontext_model,
                batch_size=10
            )

            # Add headers
            for chunk, section_summary in zip(chunks, section_summaries):
                header = build_autocontext_header(
                    doc_title=doc_title,
                    doc_summary=doc_summary,
                    section_summary=section_summary,
                    element_type=chunk['metadata'].get('element_type', 'prose')
                )
                chunk['content'] = f"{header}\n\n{chunk['content']}"
                chunk['metadata']['has_autocontext'] = True
                chunk['metadata']['doc_title'] = doc_title

    print(f"  [GPU{gpu_id}] {pdf_path.name}: {len(chunks)} chunks ({worker_id}/{total_for_worker})")

    # CRITICAL: Save checkpoint immediately after processing
    # This ensures no work is lost if job crashes or hangs
    if checkpoint_dir and chunks:
        checkpoint_file = checkpoint_dir / f"{pdf_path.stem}.json"
        try:
            with open(checkpoint_file, 'w') as f:
                json.dump(chunks, f)
            print(f"  [GPU{gpu_id}] ✓ Checkpoint saved: {checkpoint_file.name}")
        except Exception as e:
            print(f"  [GPU{gpu_id}] ⚠ Checkpoint save failed: {e}")

    # Clean up GPU memory
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except:
        pass

    return chunks


# =============================================================================
# Main Parallel Ingestion
# =============================================================================

def get_checkpoint_dir(output_dir: str) -> Path:
    """Get the checkpoint directory path for a given output directory."""
    return Path(output_dir).parent / f"{Path(output_dir).name}_checkpoints"


def get_processed_files_from_checkpoints(checkpoint_dir: Path) -> set:
    """
    Get set of already-processed PDF filenames from checkpoint files.

    This enables resume capability - we check which PDFs have checkpoint files.

    Returns:
        Set of filenames (e.g., {"APPLE_2015_10K.pdf", "AMAZON_2015_10K.pdf"})
    """
    if not checkpoint_dir.exists():
        return set()

    # Find all checkpoint JSON files
    checkpoint_files = list(checkpoint_dir.glob("*.json"))

    # Convert checkpoint names back to PDF names
    # Checkpoint: APPLE_2015_10K.json -> PDF: APPLE_2015_10K.pdf
    processed = set()
    for cp_file in checkpoint_files:
        pdf_name = f"{cp_file.stem}.pdf"
        processed.add(pdf_name)

    return processed


def load_chunks_from_checkpoints(checkpoint_dir: Path) -> List[Dict]:
    """
    Load all chunks from checkpoint files.

    Returns:
        List of all chunk dictionaries from all checkpoint files
    """
    all_chunks = []
    checkpoint_files = sorted(checkpoint_dir.glob("*.json"))

    print(f"Loading chunks from {len(checkpoint_files)} checkpoint files...")

    for cp_file in checkpoint_files:
        try:
            with open(cp_file, 'r') as f:
                chunks = json.load(f)
                all_chunks.extend(chunks)
        except Exception as e:
            print(f"  Warning: Could not load checkpoint {cp_file.name}: {e}")

    return all_chunks


def run_parallel_ingestion(
    domain: str,
    output_dir: str,
    input_dir: Optional[str] = None,
    config: Optional[DsRAGConfig] = None,
    num_workers: int = 8,
    limit: Optional[int] = None,
    skip_files: Optional[List[str]] = None,
    resume: bool = True,
):
    """
    Run dsRAG-style ingestion in parallel across multiple GPUs.

    Phase 1: Process PDFs in parallel (each worker uses its own GPU)
    Phase 2: Write all chunks to ChromaDB (single-threaded, safe)

    Args:
        domain: "finance", "medical", or "legal"
        output_dir: ChromaDB output directory
        input_dir: Input directory for PDFs
        config: DsRAG configuration
        num_workers: Number of parallel workers (typically 4 for 4 GPUs)
        limit: Limit number of files for testing
    """
    config = config or DsRAGConfig()

    # Setup checkpoint directory for incremental saves
    checkpoint_dir = get_checkpoint_dir(output_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("dsRAG PARALLEL INGESTION PIPELINE (8-GPU) - INCREMENTAL CHECKPOINTS")
    print("=" * 70)
    print(f"Domain:          {domain}")
    print(f"Output:          {output_dir}")
    print(f"Checkpoints:     {checkpoint_dir}")
    print(f"Workers:         {num_workers}")
    print(f"Embedding:       {config.embedding_model}")
    print(f"AutoContext LLM: {config.autocontext_model}")
    print(f"Skip AutoContext:{config.skip_autocontext}")
    if limit:
        print(f"Limit:           {limit}")
    print("=" * 70)

    # Only finance domain supported for parallel processing currently
    if domain != "finance":
        raise ValueError(f"Parallel processing only supports 'finance' domain, got: {domain}")

    if not input_dir:
        input_dir = "data/test_files/finance-bench-pdfs"

    # Get all PDF files
    pdf_files = sorted(Path(input_dir).glob("*.pdf"))
    total_pdfs = len(pdf_files)

    # Filter out skipped files (for recovery runs)
    if skip_files:
        skip_set = set(skip_files)
        pdf_files = [p for p in pdf_files if p.name not in skip_set]
        skipped_count = total_pdfs - len(pdf_files)
        if skipped_count > 0:
            print(f"\nSkipping {skipped_count} problematic files: {', '.join(skip_files)}")

    # Resume logic: skip already-processed files (check checkpoint files)
    if resume:
        print(f"\nChecking for checkpoint files in {checkpoint_dir}...")
        processed_files = get_processed_files_from_checkpoints(checkpoint_dir)
        if processed_files:
            before_count = len(pdf_files)
            pdf_files = [p for p in pdf_files if p.name not in processed_files]
            resumed_count = before_count - len(pdf_files)
            print(f"Resume: Found {len(processed_files)} checkpoint files, skipping {resumed_count} PDFs")
            print(f"Remaining to process: {len(pdf_files)}")

    if limit:
        pdf_files = pdf_files[:limit]

    if not pdf_files:
        raise FileNotFoundError(f"No PDFs found in {input_dir}")

    print(f"\nFound {len(pdf_files)} PDFs to process")
    print(f"Distributing across {num_workers} workers (1 GPU each)")

    # Calculate per-worker distribution
    pdfs_per_worker = [0] * num_workers
    for i in range(len(pdf_files)):
        pdfs_per_worker[i % num_workers] += 1
    print(f"PDFs per worker: {pdfs_per_worker}")
    print()

    # Prepare tasks (PDF path, GPU ID, config)
    config_dict = {
        'embedding_model': config.embedding_model,
        'autocontext_model': config.autocontext_model,
        'autocontext_temperature': config.autocontext_temperature,
        'chunk_size': config.chunk_size,
        'chunk_overlap': config.chunk_overlap,
        'batch_size': config.batch_size,
        'chroma_batch_size': config.chroma_batch_size,
        'skip_autocontext': config.skip_autocontext,
    }

    # Track worker progress
    worker_counts = {i: 0 for i in range(num_workers)}
    tasks = []
    for i, pdf in enumerate(pdf_files):
        gpu_id = i % num_workers
        worker_counts[gpu_id] += 1
        tasks.append((
            str(pdf),
            gpu_id,
            config_dict,
            worker_counts[gpu_id],  # This PDF's index for this worker
            pdfs_per_worker[gpu_id],  # Total PDFs for this worker
            str(checkpoint_dir),  # Checkpoint directory for incremental saves
        ))

    # ========================================
    # PHASE 1: Parallel PDF Processing
    # ========================================
    print("=" * 70)
    print("PHASE 1: Parallel PDF Processing")
    print("=" * 70)
    start_time = time.time()

    # Use 'spawn' context for CUDA safety
    ctx = get_context('spawn')

    # Worker timeout (10 minutes per PDF - if it takes longer, something is wrong)
    WORKER_TIMEOUT_SECONDS = 600  # 10 minutes

    # Process in parallel with timeout handling
    if tasks:
        with ctx.Pool(processes=num_workers) as pool:
            # Submit all tasks asynchronously
            async_results = [pool.apply_async(process_single_pdf, (task,)) for task in tasks]

            # Collect results with timeout
            completed = 0
            failed = 0
            for i, (task, async_result) in enumerate(zip(tasks, async_results)):
                pdf_name = Path(task[0]).name
                try:
                    async_result.get(timeout=WORKER_TIMEOUT_SECONDS)
                    completed += 1
                except Exception as e:
                    failed += 1
                    print(f"  ⚠ TIMEOUT/ERROR on {pdf_name}: {e}")
                    # Continue - checkpoint may still have been saved

            print(f"\nPhase 1 results: {completed} completed, {failed} failed/timeout")
    else:
        print("No new PDFs to process - all already have checkpoints")

    phase1_time = time.time() - start_time
    print(f"\nPhase 1 complete in {phase1_time/60:.1f} minutes")

    # ========================================
    # PHASE 2: ChromaDB Write (from checkpoints)
    # ========================================
    print()
    print("=" * 70)
    print("PHASE 2: ChromaDB Write (from checkpoints)")
    print("=" * 70)

    from langchain_core.documents import Document
    from langchain_chroma import Chroma
    from tqdm import tqdm

    # Import here to avoid issues
    sys.path.insert(0, str(BASE_DIR))
    from src.config import get_embedding_model

    # Load ALL chunks from checkpoint files (includes previous runs)
    all_chunks = load_chunks_from_checkpoints(checkpoint_dir)
    print(f"Loaded {len(all_chunks)} chunks from checkpoint files")

    if not all_chunks:
        print("No chunks to write - exiting")
        return

    print(f"Loading embedding model: {config.embedding_model}...")
    embeddings = get_embedding_model(config.embedding_model)

    # Create output directory (clear existing to avoid duplicates)
    output_path = Path(output_dir)
    if output_path.exists():
        import shutil
        print(f"Clearing existing ChromaDB at {output_path}...")
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Convert chunk dicts to Documents
    documents = [
        Document(page_content=c['content'], metadata=c['metadata'])
        for c in all_chunks
    ]

    print(f"Writing {len(documents)} chunks to ChromaDB (fresh build from checkpoints)...")

    db = Chroma(
        persist_directory=str(output_path),
        embedding_function=embeddings,
        collection_name='langchain'
    )

    # Add in batches
    for i in tqdm(range(0, len(documents), config.chroma_batch_size), desc="Batches"):
        batch = documents[i:i + config.chroma_batch_size]
        db.add_documents(batch)
        gc.collect()

    phase2_time = time.time() - start_time - phase1_time
    total_time = time.time() - start_time

    # ========================================
    # Summary
    # ========================================
    final_count = db._collection.count()

    print()
    print("=" * 70)
    print("INGESTION COMPLETE")
    print("=" * 70)
    print(f"Domain:          {domain}")
    print(f"Total chunks:    {final_count}")
    print(f"Phase 1 (Parse): {phase1_time/60:.1f} minutes")
    print(f"Phase 2 (Embed): {phase2_time/60:.1f} minutes")
    print(f"Total time:      {total_time/60:.1f} minutes ({total_time/3600:.1f} hours)")
    print(f"ChromaDB path:   {output_dir}")
    print(f"Workers used:    {num_workers}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="dsRAG-style parallel ChromaDB builder (4-GPU support)"
    )
    parser.add_argument(
        "--domain",
        type=str,
        required=True,
        choices=["finance", "medical", "legal"],
        help="Domain to process (only 'finance' supports parallel)"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        help="Input directory for PDFs"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output ChromaDB directory"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of parallel workers (default: 8, one per GPU)"
    )
    parser.add_argument(
        "--skip-files",
        type=str,
        default="",
        help="Comma-separated list of filenames to skip (for recovery runs)"
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="cohere-v3",
        help="Embedding model (default: cohere-v3)"
    )
    parser.add_argument(
        "--autocontext-model",
        type=str,
        default="gpt-4o-mini",
        help="LLM for AutoContext generation"
    )
    parser.add_argument(
        "--skip-autocontext",
        action="store_true",
        help="Use rule-based headers instead of LLM (faster/cheaper)"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=2500,
        help="Max characters per prose chunk"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Chunks per ChromaDB batch"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of PDFs (for testing)"
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable resume (reprocess all PDFs, even if already in ChromaDB)"
    )

    args = parser.parse_args()

    config = DsRAGConfig(
        embedding_model=args.embedding_model,
        autocontext_model=args.autocontext_model,
        skip_autocontext=args.skip_autocontext,
        chunk_size=args.chunk_size,
        chroma_batch_size=args.batch_size,
    )

    # Parse skip files list
    skip_files = None
    if args.skip_files:
        skip_files = [f.strip() for f in args.skip_files.split(',') if f.strip()]

    run_parallel_ingestion(
        domain=args.domain,
        output_dir=args.output_dir,
        input_dir=args.input_dir,
        config=config,
        num_workers=args.num_workers,
        limit=args.limit,
        skip_files=skip_files,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()
