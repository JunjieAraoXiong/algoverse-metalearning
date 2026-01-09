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

### Attempt 4: Local SSD Optimization (The /tmp Strategy)
*   **Discovery:** Reading 368 PDFs simultaneously from a Network Drive (`/data/`) caused massive congestion and "silent freezes."
*   **Solution:** Copying PDFs to the node's local NVMe SSD (`/tmp/junjie_pdfs/`) before processing.
*   **Result:** Significantly reduced startup latency and eliminated "thundering herd" bottlenecks on 60-worker runs.

### Strategy Comparison: "Hi-Res" vs "Fast"
*   **Hi-Res:** Essential for final research. Preserves tables. Slow (CPU bound OCR).
*   **Fast:** Used for pipeline verification. Text-only. 10x faster.
*   **Decision:** Use "Fast" for database structure testing, "Hi-Res" for final "Gold" database build.

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
