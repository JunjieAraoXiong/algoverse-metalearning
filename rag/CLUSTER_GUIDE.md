# Together AI GPU Cluster - Session Checklist

Follow these steps every time you start a new session to ensure the environment is ready for Finance RAG experiments.

## 1. Request GPU Allocation
Run this on the login node to get your 8x H100s (Big Node):
```bash
srun --gpus=8 --cpus-per-task=64 --mem=400G --time=08:00:00 --pty bash
```
*If this fails (node unavailable), fall back to: `--cpus-per-task=32 --mem=200G`*

## 2. Re-install System Dependencies
*These are wiped from the system path when the node resets.*
```bash
sudo apt-get update && sudo apt-get install -y poppler-utils libgl1 tesseract-ocr
```

## 3. Activate Python Environment
```bash
cd ~/algoverse-metalearning/rag
source venv/bin/activate
```

## 4. Set Required Environment Variables
*Ensures writable paths for AI models and caches.*
```bash
export NLTK_DATA=/data/home/junjiexiong/nltk_data
export MPLCONFIGDIR=/data/home/junjiexiong/.config/matplotlib
export HF_HOME=/data/home/junjiexiong/.cache/huggingface
mkdir -p $NLTK_DATA $MPLCONFIGDIR $HF_HOME
```

## 5. Run Ingestion (If needed)
Build the vector database (Parallel & GPU accelerated):
```bash
python src/create_database_v2.py
```

## 6. Run Evaluation
Test the RAG pipeline on FinanceBench:
```bash
python src/bulk_testing.py --pipeline hybrid_filter_rerank --top-k 10
```

---

## Troubleshooting Tips
* **OOM Error:** If ingestion crashes with "abruptly terminated", ensure `max_workers` in `create_database_v2.py` is capped (e.g., `min(16, ...)`).
* **Missing API Key:** Check `.env` file exists in the `rag/` directory.
* **GPU Not Found:** Run `nvidia-smi` to verify 8 GPUs are visible.
