# Financial RAG System

RAG system for financial document QA, evaluated on FinanceBench. Research target: ICLR 2026 FinAI Workshop.

## Status

- **ChromaDB:** 129,949 chunks (ready)
- **Evaluation:** Working (150 questions)
- **Cluster:** Together AI SLURM

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Set API keys
export TOGETHER_API_KEY="your_key"

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
│   ├── ingest.py              # PDF → ChromaDB (not needed, DB provided)
│   ├── providers/             # LLM adapters (Anthropic, OpenAI, Google, etc.)
│   └── retrieval_tools/       # Retrieval pipelines
├── evaluation/                # Metrics & LLM judge
├── dataset_adapters/          # Dataset loaders (FinanceBench, PubMedQA)
├── chroma/                    # Vector database (129K chunks)
├── data/question_sets/        # Test questions (150)
├── scripts/                   # SLURM job scripts
└── docs/                      # Documentation
```

## Retrieval Pipelines

| Pipeline | Latency | Quality | Use Case |
|----------|---------|---------|----------|
| `semantic` | ~50ms | Low | Prototyping |
| `hybrid` | ~100ms | Medium | General |
| `hybrid_filter` | ~120ms | High | Domain-specific |
| `hybrid_filter_rerank` | ~300ms | Highest | Production (default) |

## Current Performance

| Question Type | Score | Target |
|---------------|-------|--------|
| metrics-generated | 0.35 | 0.55+ |
| domain-relevant | 0.60 | 0.70+ |
| novel-generated | 0.53 | 0.65+ |
| **Overall** | **0.495** | **0.65+** |

## Supported Models

| Provider | Models | Cost (150 Qs) |
|----------|--------|---------------|
| Together | Llama 3.1 70B | ~$0.50 |
| DeepSeek | DeepSeek Chat | ~$0.05 |
| Google | Gemini 3 Flash | ~$0.07 |
| Anthropic | Claude 4.5 | ~$2.00 |
| OpenAI | GPT-5.2 | ~$3.20 |

## Docs

- [Cluster Guide](docs/cluster.md) — SLURM setup & commands
- [Status Notes](docs/status.md) — Current project status

## Commands

```bash
# Run evaluation
python src/bulk_testing.py --model deepseek-chat --pipeline hybrid_filter_rerank

# With LLM judge
python src/bulk_testing.py --model claude-sonnet-4-5-20250514 --use-llm-judge

# Verify ChromaDB
python -c "from langchain_chroma import Chroma; db = Chroma(persist_directory='chroma'); print(db._collection.count())"
```
