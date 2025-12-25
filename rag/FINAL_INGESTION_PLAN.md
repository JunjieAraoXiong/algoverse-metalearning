# Final Ingestion Plan (The "Sleep-Safe" Method)

This is the definitive order of operations to build the "Gold" Finance RAG database overnight.

---

### Step 1: Secure the Resource
Run this on the login node to get your 8-hour window.
```bash
srun --gpus=8 --cpus-per-task=64 --mem=400G --time=08:00:00 --pty bash
```

### Step 2: Refresh the Environment
*These tools and paths are essential for the AI models to work.*
```bash
# 1. Update system tools
sudo apt-get update && sudo apt-get install -y poppler-utils libgl1 tesseract-ocr

# 2. Setup paths
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

### Step 4: Fire and Forget (Persistence)
*Use `nohup` to ensure the script runs even after you close your laptop.*
```bash
nohup python src/create_database_v2.py --data-dir /tmp/junjie_pdfs/ > ingestion_final.log 2>&1 &
```

---

### Step 5: Verify & Sleep
1. Run `tail -f ingestion_final.log` to see the first few lines of output.
2. If you see "Processing 368 PDF files...", press **Ctrl+C** (this stops the *view*, not the script).
3. Run `squeue -u junjiexiong` to confirm the job is still "R".
4. **Close your laptop. You are done.**
