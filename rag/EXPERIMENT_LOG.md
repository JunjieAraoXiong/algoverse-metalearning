# Experiment Log: Financial RAG Project

This document tracks our engineering decisions, experiments, and technical hurdles.

## 1. Ingestion Optimizations (December 2024)

### Goal
Ingest 367 FinanceBench PDFs (10-Ks, 10-Qs) into a ChromaDB vector store with rich metadata and table structure preservation.

### Attempt 1: Sequential Processing (CPU)
*   **Method:** Loop through each PDF one-by-one.
*   **Result:** Estimated completion time ~4-5 hours.
*   **Bottleneck:** Tesseract OCR (single core) was the limiting factor. The H100 GPUs sat idle 99% of the time waiting for text.

### Attempt 2: Naive Parallelism (CPU)
*   **Method:** Used `ProcessPoolExecutor` with `max_workers = cpu_count() - 2` (126 workers).
*   **Result:** **CRASH (Out of Memory).**
*   **Root Cause:** Each worker spawns a heavy instance of `Unstructured` + `PyTorch` + `LayoutModel`.
    *   126 workers × ~3-5GB RAM ≈ >400GB RAM demand.
    *   Node limit: 200GB.
*   **Lesson:** CPU count is not the limit for heavy AI tasks; RAM is.

### Attempt 3: Optimized Parallelism (Current Strategy)
*   **Method:** Capped `max_workers` at **16**.
*   **Hardware Mapping:**
    *   **CPU:** 16 cores used for parsing/OCR.
    *   **RAM:** ~60-80GB usage (safe headroom).
    *   **GPU:** H100s used for Embedding generation (`bge-large`) in massive batches (5,000 chunks).
*   **Status:** Stable. Estimated runtime ~15-20 minutes.

### Technical Stack Decisions
*   **OCR Engine:** Tesseract (via `Unstructured`). Chosen for reliability over speed. Future work: `Surya` or `Nougat` for GPU acceleration.
*   **Chunking:** `chunk_by_title` to preserve semantic sections.
*   **Table Handling:** `hi_res` strategy converts HTML tables to **Markdown**. This preserves row/column headers for the LLM.

## 2. Infrastructure Setup (Together AI Cluster)

### Dependency Hell & Fixes
*   **Issue:** `ImportError: libGL.so.1` (OpenCV).
    *   **Fix:** `sudo apt-get install libgl1` or use `opencv-python-headless`.
*   **Issue:** `Unable to get page count` (Poppler).
    *   **Fix:** `sudo apt-get install poppler-utils`.
*   **Issue:** `Permission denied: /home/user`.
    *   **Fix:** Redirected cache paths (`NLTK_DATA`, `MPLCONFIGDIR`, `HF_HOME`) to the writable `/data/home/` volume.

### GPU Utilization
*   **Embeddings:** Configured `langchain-huggingface` to detect `cuda` device automatically.
*   **Reranking:** Updated `CrossEncoder` to initialize on `cuda`.
*   **LLM:** Future plan to host `vLLM` locally for zero-cost inference.
