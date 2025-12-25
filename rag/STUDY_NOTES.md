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
