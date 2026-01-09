# Cluster Guide

Together AI SLURM cluster setup and commands.

## SSH Access

```bash
# Login
ssh -J junjiexiong@ssh.t-2fc5bf5b-fa73-47b4-96e1-acdca7819356.s1.us-central-8a.cloud.together.ai junjiexiong@slurm-login
```

## Initial Setup (One-Time)

### 1. Upload Code & ChromaDB (from local machine)

```bash
# Upload entire project
rsync -avz --progress -e "ssh -J junjiexiong@ssh.t-2fc5bf5b-fa73-47b4-96e1-acdca7819356.s1.us-central-8a.cloud.together.ai" \
    /Users/hansonxiong/Desktop/algoverse/rag \
    junjiexiong@slurm-login:/data/junjiexiong/algoverse/
```

### 2. Setup Environment (on cluster)

```bash
cd /data/junjiexiong/algoverse/rag

# Create venv
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Set API keys
echo 'export TOGETHER_API_KEY="your_key"' >> ~/.bashrc
source ~/.bashrc

# Set cache paths
cat >> ~/.bashrc << 'EOF'
export HF_HOME=/data/junjiexiong/.cache/huggingface
export NLTK_DATA=/data/junjiexiong/.cache/nltk_data
export MPLCONFIGDIR=/data/junjiexiong/.cache/matplotlib
EOF
source ~/.bashrc
mkdir -p $HF_HOME $NLTK_DATA $MPLCONFIGDIR
```

### 3. Verify ChromaDB

```bash
python -c "
from langchain_chroma import Chroma
db = Chroma(persist_directory='chroma')
print(f'ChromaDB: {db._collection.count()} chunks')
"
# Expected: 129949 chunks
```

## Running Evaluations

### Interactive Session

```bash
# 1. Start tmux (survives disconnection)
tmux new -s rag

# 2. Request GPU
srun --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=04:00:00 --pty bash

# 3. Activate environment
cd /data/junjiexiong/algoverse/rag
source .venv/bin/activate

# 4. Run evaluation
python src/bulk_testing.py \
    --model meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo \
    --pipeline hybrid_filter_rerank \
    --top-k 10

# 5. Detach: Ctrl+B, D
# 6. Reconnect later: tmux attach -t rag
```

### Batch Job

```bash
sbatch scripts/eval_job.sh
```

## SLURM Quick Reference

| Command | Description |
|---------|-------------|
| `squeue -u junjiexiong` | Your jobs |
| `scancel <job_id>` | Cancel job |
| `sinfo` | Cluster status |
| `tail -f logs/eval_*.out` | Watch output |

## GPU Options

```bash
# 1 GPU (evaluation, reranking)
srun --gres=gpu:1 --mem=32G --time=02:00:00 --pty bash

# 8 GPUs (vLLM server for Llama 70B)
srun --gres=gpu:8 --mem=256G --time=08:00:00 --pty bash
```

## Local vLLM Server (Free Inference)

Run Llama-3.1-70B locally instead of using paid APIs.

### Launch via SLURM Batch Job

```bash
sbatch scripts/launch_vllm.sh
```
*Requests 8 GPUs, 256GB RAM, 24 hour time limit.*

**Monitor:**
```bash
squeue -u $USER                    # Check job status
tail -f logs/vllm_<job_id>.out     # Watch server logs
scancel <job_id>                   # Stop the server
```

### Pointing RAG to Local vLLM

The system auto-routes `meta-llama/` models to local vLLM:
```bash
python src/bulk_testing.py \
    --model meta-llama/Meta-Llama-3.1-70B-Instruct \
    --pipeline hybrid_filter_rerank \
    --top-k 10
```
- Endpoint: `http://localhost:8000/v1`
- No API key required

### Cost Savings

| Method | Cost |
|--------|------|
| Together AI API | ~$0.90/M tokens |
| OpenAI GPT-4o | ~$5.00/M tokens |
| **Local vLLM** | **$0 (free)** |

*Trade-off: Requires GPU allocation. Use for long-running experiments.*

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Connection dropped | Use tmux |
| Job pending | `squeue -o "%R"` to see reason |
| Module not found | `source .venv/bin/activate` |
| Time limit | Use longer `--time` or tmux |

## Paths

| Item | Location |
|------|----------|
| Project | `/data/junjiexiong/algoverse/rag` |
| ChromaDB | `/data/junjiexiong/algoverse/rag/chroma` |
| Results | `/data/junjiexiong/algoverse/rag/bulk_runs` |
