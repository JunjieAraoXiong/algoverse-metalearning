# AI Engineering Study Notes: High-Performance RAG Ingestion

These notes summarize the architectural and operational lessons learned while building a large-scale Financial RAG pipeline on an H100 GPU cluster.

---

## Module 1: Memory & Parallelism (RAM vs. CPU)
**The Problem:** Most clusters assign many CPUs (64+) but finite RAM (200GB). Launching one process per CPU causes an "Out of Memory" (OOM) crash if each process loads a heavy AI model.
*   **The Rule:** Your `max_workers` must be limited by **RAM**, not CPU count.
*   **The Math:** `(Memory per Worker) * (Number of Workers) < (Total Node RAM) - (Safety Buffer)`.
*   **Sweet Spot:** For PDF parsing (Unstructured/PyTorch), 16-28 workers is usually the stability limit for a 200GB node.

## Module 2: The "Thundering Herd" (Network I/O)
**The Problem:** `/data/` drives are usually Network-Attached Storage (NAS). They are slow at "Random Reads." If 60 workers all request 60 different 100MB PDFs at the same millisecond, the drive controller chokes, causing a "Silent Freeze."
*   **The Fix: Local SSD Buffering.**
    *   Always copy your dataset to the node's **local** SSD (`/tmp/` or `/scratch/`) before processing.
    *   Local NVMe drives have much higher "IOPS" (Input/Output Operations Per Second).
*   **Workflow:** `cp /data/source/*.pdf /tmp/workdir/` -> `process /tmp/workdir/`.

## Module 3: Persistent Sessions (Resilience)
**The Problem:** SSH connections are volatile. If your laptop sleeps, your Wi-Fi drops, or your terminal crashes, the server kills your running job.
*   **The Tool: `tmux` (Terminal Multiplexer).**
    *   `tmux` keeps the process alive on the server independently of your laptop.
    *   **Workflow:** `tmux new -s name` -> `run script` -> `Ctrl+B, D` (Detach).
    *   **Recovery:** `tmux attach -t name`.

## Module 4: Strategy Trade-offs (OCR vs. Fast)
**The Problem:** Some PDFs are "text-hidden" (images) or have complex tables.
*   **`hi_res` Strategy:** Uses Computer Vision to find tables and OCR to read text. (Slow, high precision).
*   **`fast` Strategy:** Extracts raw text strings from the PDF metadata. (10x faster, low precision for tables).
*   **Research Rule:** Use `fast` to debug the "plumbing" of your code. Use `hi_res` for the final "Gold" data generation.

## Module 5: Common Cluster Gotchas
*   **Permission Denied:** NLTK and HuggingFace try to download models to your `~/.cache` (which is often read-only on Slurm nodes).
    *   **Fix:** Export environment variables (`NLTK_DATA`, `HF_HOME`) to a writable `/data/` path.
*   **System Tools:** Tools like `poppler-utils` (for PDF reading) are often missing on clean GPU nodes.
    *   **Fix:** `sudo apt-get install` must be run on the GPU node, not the login node.
*   **ChromaDB Max Batch Size:** Chroma crashes if you try to `add` more than ~5461 documents at once.
    *   **Fix:** When using `--fast` mode (which generates 20k+ chunks per batch), you *must* slice the list into sub-batches of 5000 before adding to DB.
## Module 6: Industry Intelligence (Gestell.ai Benchmark)
**The Benchmark:** FinanceBench (50k+ pages, 10k questions).
*   **Traditional RAG:** ~30-35% accuracy.
*   **Fine-tuned Embeddings (Databricks):** ~65% accuracy.
*   **Gestell.ai:** ~88% accuracy.

**Key Differentiators (Actionable):**
1.  **Natural Language Structuring:** Instead of heuristics (`chunk_by_title`), they use LLMs to "structure" PDFs based on natural language instructions (e.g., "Extract fiscal year as metadata"). *We are doing this via regex in `src/ingest.py` (v2), but LLM-based is more robust.*
2.  **Specialized Knowledge Graphs:** "Naive Graphs + Naive Vectors = Worst Results." They use highly specialized graphs based on use-case (e.g., Company relationships, Financial statement links).
3.  **Chain of Thought (CoT):** Used at *both* ingestion (to structure data) and retrieval (to reason about where to look).
4.  **Re-ranking:** Essential for "squeezing the last few %" at scale. *We have this in `hybrid_filter_rerank`.*

**Takeaway:** Our `ingest.py` (Metadata extraction) and `hybrid_filter_rerank` (Re-ranking) are on the right track. The next leap is **LLM-driven Structuring** (asking models to extract data) and **Specialized KGs** (connecting concepts).
