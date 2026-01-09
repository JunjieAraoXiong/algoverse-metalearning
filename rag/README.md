# RAG System

RAG system for financial document QA, evaluated on FinanceBench.

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Set API keys
cp .env.example .env
# Edit .env with your keys

# Run evaluation
python src/bulk_testing.py \
    --model meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo \
    --pipeline hybrid_filter_rerank \
    --top-k 10
```

## Project Structure

```
rag/
├── src/
│   ├── config.py              # Central configuration
│   ├── bulk_testing.py        # Evaluation runner
│   ├── ingest.py              # PDF → ChromaDB
│   ├── providers/             # LLM adapters (Anthropic, OpenAI, Google, etc.)
│   ├── retrieval_tools/       # Retrieval pipelines
│   └── postprocessing/        # Answer post-processing
├── evaluation/                # Metrics & LLM judge
├── dataset_adapters/          # Dataset loaders (FinanceBench, PubMedQA)
├── data/                      # Question sets and test files
├── scripts/                   # SLURM job scripts
└── docs/                      # Documentation
```

## Retrieval Pipelines

| Pipeline | Latency | Use Case |
|----------|---------|----------|
| `semantic` | ~50ms | Prototyping |
| `hybrid` | ~100ms | General |
| `hybrid_filter` | ~120ms | Domain-specific |
| `hybrid_filter_rerank` | ~300ms | Production (default) |

## Supported Models

| Provider | Models |
|----------|--------|
| Together | Llama 3.1 70B |
| DeepSeek | DeepSeek Chat |
| Google | Gemini 3 Flash |
| Anthropic | Claude 4.5 |
| OpenAI | GPT-5.2 |

## Cluster Usage

```bash
# Setup environment
source scripts/setup_env.sh

# Run evaluation job
sbatch scripts/eval_job.sh

# Launch vLLM server
sbatch scripts/launch_vllm.sh
```

## Docs

- [Cluster Guide](docs/cluster.md) - SLURM setup & commands

## Commands

```bash
# Run evaluation
python src/bulk_testing.py --model deepseek-chat --pipeline hybrid_filter_rerank

# With LLM judge
python src/bulk_testing.py --model claude-sonnet-4-5-20250514 --use-llm-judge

# Verify ChromaDB
python -c "from langchain_chroma import Chroma; db = Chroma(persist_directory='chroma'); print(db._collection.count())"
```
