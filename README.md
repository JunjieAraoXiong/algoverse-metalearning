# Algoverse - Meta-Learning for RAG Pipeline Selection

A research project exploring **meta-learning for RAG pipeline selection** across multiple domains (Finance, Healthcare, Legal). The core hypothesis: instead of optimizing a single retrieval pipeline, we can learn to *select* the optimal pipeline per question type.

## Project Vision

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        META-LEARNING FOR RAG                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Question ──▶ META-ROUTER ──▶ Select Pipeline ──▶ Execute ──▶ Answer       │
│                    │                                                        │
│                    │   Learns patterns:                                     │
│                    │   • numeric questions → hybrid_filter                  │
│                    │   • dense prose → semantic                             │
│                    │   • multi-hop → hybrid_filter_rerank                   │
│                    │                                                        │
│   ┌────────────────┴────────────────┐                                       │
│   │         TRAINING DOMAINS        │                                       │
│   ├─────────────┬─────────────┬─────┴─────┐                                 │
│   │  FINANCE    │  HEALTHCARE │   LEGAL   │                                 │
│   │ FinanceBench│  PubMedQA   │   CUAD    │                                 │
│   │  (tables)   │   (prose)   │ (clauses) │                                 │
│   └─────────────┴─────────────┴───────────┘                                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Key Insight**: Different question types need different retrieval strategies. Per-file RAG (51%) outperforms shared-store RAG (19%) by 2.7x on FinanceBench. We aim to learn this selection automatically.

## Repository Structure

```
algoverse/
├── rag/                          # Core RAG system
│   ├── src/
│   │   ├── config.py             # Central configuration
│   │   ├── bulk_testing.py       # Evaluation framework
│   │   ├── ingest.py             # PDF → ChromaDB ingestion
│   │   ├── providers/            # LLM adapters (Claude, GPT, Gemini, etc.)
│   │   ├── retrieval_tools/      # Pipeline components
│   │   │   ├── semantic.py       # Vector similarity
│   │   │   ├── hybrid.py         # BM25 + Semantic
│   │   │   ├── metadata_filter.py# Company/year filtering
│   │   │   ├── rerank.py         # Cross-encoder reranking
│   │   │   └── router.py         # Pipeline routing
│   │   └── postprocessing/       # Answer post-processing
│   ├── evaluation/               # Metrics (semantic sim, LLM judge)
│   ├── dataset_adapters/         # Dataset loaders (FinanceBench, PubMedQA)
│   ├── data/                     # Test data and question sets
│   ├── scripts/                  # SLURM job scripts
│   └── docs/                     # Documentation
│
├── meta-learning/                # Research materials
│   └── extracted/                # Extracted text from papers
│
└── .claude/                      # Claude Code agents and commands
```

## Quick Start

```bash
# 1. Clone and setup
git clone https://github.com/JunjieAraoXiong/algoverse-metalearning.git
cd algoverse-metalearning
git checkout features/metalearning

# 2. Setup environment
cd rag
pip install -r requirements.txt

# 3. Configure API keys
cp .env.example .env
# Edit .env with your API keys

# 4. Run evaluation
python src/bulk_testing.py \
    --model meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo \
    --pipeline hybrid_filter_rerank \
    --top-k 10
```

## Available Retrieval Pipelines

| Pipeline | Components | Latency | Best For |
|----------|------------|---------|----------|
| `semantic` | Vector search only | ~50ms | Simple queries |
| `hybrid` | BM25 + Semantic | ~100ms | General use |
| `hybrid_filter` | + Metadata filtering | ~120ms | Domain-specific |
| `hybrid_filter_rerank` | + Cross-encoder | ~300ms | Production |

## Supported LLM Providers

| Provider | Models | Notes |
|----------|--------|-------|
| Together | Llama 3.1 70B | Free cluster inference |
| DeepSeek | DeepSeek Chat | Low cost |
| Google | Gemini 3 Flash | Fast |
| Anthropic | Claude 4.5 | High quality |
| OpenAI | GPT-5.2 | High quality |

## Cluster Usage

For Together AI SLURM cluster:

```bash
# Setup environment
source scripts/setup_env.sh

# Run evaluation job
sbatch scripts/eval_job.sh

# Launch vLLM server (free inference)
sbatch scripts/launch_vllm.sh
```

See [Cluster Guide](rag/docs/cluster.md) for details.

## Roadmap

### Phase 1: RAG System (Complete)
- [x] Multi-provider LLM abstraction
- [x] 4 retrieval pipelines
- [x] Metadata filtering
- [x] Evaluation framework
- [x] Cluster setup scripts

### Phase 2: Per-File Retrieval
- [ ] Enhanced question metadata extraction
- [ ] ChromaDB native filtering
- [ ] Fallback strategies

### Phase 3: Question-Type Routing
- [ ] Question classifier
- [ ] Type-specific prompts
- [ ] Table-priority retrieval

### Phase 4: Meta-Learning Router
- [ ] Multi-domain datasets (PubMedQA, CUAD)
- [ ] Oracle labels via grid search
- [ ] Meta-router training
- [ ] Cross-domain evaluation

## Research References

- [FinanceBench](https://arxiv.org/abs/2311.11944) - Financial QA benchmark
- [Toolformer](https://arxiv.org/abs/2302.04761) - Tool use in LLMs
- [MAML](https://arxiv.org/abs/1703.03400) - Model-Agnostic Meta-Learning
