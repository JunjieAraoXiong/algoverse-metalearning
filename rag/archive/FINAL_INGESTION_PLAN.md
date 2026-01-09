# Final Ingestion Plan (The "Sleep-Safe" Method)

This is the definitive order of operations to build the "Gold" Finance RAG database overnight.

---

### 1. Setup Session (CRITICAL)
**WARNING**: Your jobs are killed exactly at 8 hours (`08:00:00`).
Connection drops ("Broken pipe") kill your job if you don't use `tmux`.

**ALWAYS start a tmux session first:**
```bash
tmux new -s rag_ingest
```

## 2. Request GPU & Move Data
Inside tmux:
```bash
srun --gpus=8 --cpus-per-task=64 --mem=400G --time=08:00:00 --pty bash
# (Wait for node allocation...)

# Re-install dependencies (wiped on reset)
sudo apt-get update && sudo apt-get install -y poppler-utils libgl1 tesseract-ocr

# Move data to fast SSD (prevents I/O freeze)
mkdir -p /tmp/junjie_pdfs/
cp data/test_files/finance-bench-pdfs/*.pdf /tmp/junjie_pdfs/

# Setup paths
cd /data/home/junjiexiong/algoverse-metalearning/rag
source venv/bin/activate
export NLTK_DATA=/data/home/junjiexiong/nltk_data
export MPLCONFIGDIR=/data/home/junjiexiong/.config/matplotlib
export HF_HOME=/data/home/junjiexiong/.cache/huggingface
mkdir -p $NLTK_DATA $MPLCONFIGDIR $HF_HOME
```

### Step 3: Fast I/O Setup
*Move the PDFs to the local SSD to prevent network bottlenecks.*
```bash
mkdir -p /tmp/junjie_pdfs
cp data/test_files/finance-bench-pdfs/*.pdf /tmp/junjie_pdfs/
```

## 3. Run Ingestion (FAST MODE)
This will process all 367 PDFs in **~20 minutes** (instead of 4 hours).
It uses the new `--fast` flag (skips OCR) and smaller chunks for better retrieval.

```bash
nohup python src/ingest.py --fast --chunk-size 1000 --batch-size 20 --data-dir /tmp/junjie_pdfs/ > ingestion_fast.log 2>&1 &
```

**Why this mode?**
- **Speed**: Gets you a working database immediately.
- **Precision**: Smaller chunks (1000 chars) are often better for specific questions.
- **Safety**: Still resumable. If it stops, just run it again.

## 4. Launch Local vLLM (Free Inference)
Once ingestion is done, turn on the "Brain" (Llama-3 on 8x H100s).

```bash
# In a new tmux session:
bash scripts/launch_vllm.sh
```

## 5. Run Evaluation
Test the pipeline immediately to get a baseline score.

```bash
# In your main session:
python src/bulk_testing.py --model meta-llama/Meta-Llama-3.1-70B-Instruct --pipeline hybrid_filter_rerank --top-k 10
```

---

### Step 6: Verify & Sleep
1. Run `tail -f ingestion_fast.log` to see the first few lines of output.
2. If you see "Processing 367 PDF files...", press **Ctrl+C** (this stops the *view*, not the script).
3. Run `squeue -u junjiexiong` to confirm the job is still "R".
4. **Close your laptop. You are done.**
