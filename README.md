# Multi-Domain Financial Agent

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
Multi-Domain-Financial-Agent/
│
├── rag/                          # Core RAG system
│   ├── src/                      # Implementation
│   │   ├── config.py             # Central configuration
│   │   ├── providers/            # LLM adapters (Claude, GPT, Gemini)
│   │   ├── retrieval_tools/      # Pipeline components
│   │   │   ├── semantic.py       # Vector similarity
│   │   │   ├── hybrid.py         # BM25 + Semantic
│   │   │   ├── metadata_filter.py# Company/year filtering
│   │   │   └── rerank.py         # Cross-encoder reranking
│   │   ├── bulk_testing.py       # Evaluation framework
│   │   └── meta_learning/        # Meta-router (WIP)
│   ├── evaluation/               # Metrics (semantic sim, LLM judge)
│   ├── dataset_adapters/         # Dataset loaders
│   └── ROADMAP.md                # Detailed RAG roadmap
│
├── llm_as_a_judge/               # LLM-as-a-Judge evaluation
│   ├── aum_runs/                 # Aum's evaluation results
│   ├── junjie_runs/              # Junjie's evaluation results
│   ├── visualizations/           # Generated charts
│   ├── analyze_results.py        # Analysis script
│   └── run_judge_evaluation.py   # Evaluation runner
│
├── meta-learning/                # Research materials
│   ├── files/                    # Reference papers (PDFs)
│   │   ├── Lec17_Meta_learning.pdf
│   │   └── 2302.04761.pdf        # Toolformer paper
│   └── extracted/                # Extracted text for reference
│
├── notes/                        # Research notes
│   └── ahkil_group.tex           # Group meeting notes
│
└── roadmap.md                    # High-level project roadmap
```

## Current Results

### LLM-as-a-Judge Evaluation (Best Configs)

| Rank | Configuration | Score | Key Factor |
|------|---------------|-------|------------|
| 1 | Element-Based + Forced Answer | **70.4%** | 100% correct doc retrieval |
| 2 | Element-Based + Hybrid Metadata | 59.8% | Metadata filtering |
| 3 | 2000-Char Chunking | 50.4% | Larger context |
| 4 | Character-Based Chunking | 46.9% | Baseline |

### RAG Semantic Similarity Scores

| Question Type | Current | Target |
|---------------|---------|--------|
| metrics-generated | 0.35 | 0.55+ |
| domain-relevant | 0.60 | 0.70+ |
| novel-generated | 0.53 | 0.65+ |
| **Overall** | **0.495** | **0.65+** |

## Available Retrieval Pipelines

| Pipeline | Components | Latency | Best For |
|----------|------------|---------|----------|
| `semantic` | Vector search only | ~50ms | Simple queries |
| `hybrid` | BM25 + Semantic | ~100ms | General use |
| `hybrid_filter` | + Metadata filtering | ~120ms | Domain-specific |
| `hybrid_filter_rerank` | + Cross-encoder | ~300ms | Production |

## Quick Start

```bash
# 1. Clone and setup
git clone https://github.com/JunjieAraoXiong/algoverse-metalearning.git
cd algoverse-metalearning
git checkout features/metalearning

# 2. Install RAG dependencies
cd rag
pip install -r requirements.txt

# 3. Configure API keys
echo "TOGETHER_API_KEY=your_key" >> rag/.env

# 4. Ingest Data (Fast Mode)
# Takes ~20 minutes on cluster (skips OCR)
cd rag
python src/ingest.py --fast --chunk-size 1000 --batch-size 20 --data-dir /tmp/junjie_pdfs/

# 5. Launch Inference Server (Free H100s)
# Starts Llama-3-70B on 8x H100s
bash scripts/launch_vllm.sh

# 6. Run evaluation
python src/bulk_testing.py --model meta-llama/Meta-Llama-3.1-70B-Instruct --pipeline hybrid_filter_rerank --top-k 10
```

## Roadmap

### Phase 1: RAG Improvements (90% Complete)
- [x] Project structure and config system
- [x] LLM provider abstraction (5 providers)
- [x] Retrieval pipelines (4 strategies)
- [x] Metadata extraction and filtering
- [x] Evaluation framework
- [ ] Full ingestion with v2 pipeline

### Phase 2: Per-File Retrieval
- [ ] Enhanced metadata extraction from questions
- [ ] ChromaDB native filtering
- [ ] Fallback strategies

### Phase 3: Question-Type Routing
- [ ] Question classifier (metrics/domain/novel)
- [ ] Type-specific prompts
- [ ] Table-priority retrieval

### Phase 4: Meta-Learning
- [ ] PubMedQA dataset setup
- [ ] CUAD dataset setup
- [ ] Oracle labels via grid search
- [ ] Meta-router training
- [ ] Cross-domain evaluation

See `rag/ROADMAP.md` for detailed implementation plan.

## Research References

### Meta-Learning
- [MAML](https://arxiv.org/abs/1703.03400) - Model-Agnostic Meta-Learning
- [Prototypical Networks](https://arxiv.org/abs/1703.05175) - Metric-based approach
- [MetaICL](https://arxiv.org/abs/2205.12755) - In-context learning

### RAG & Retrieval
- [DPR](https://arxiv.org/abs/2004.04906) - Dense Passage Retrieval
- [BGE](https://arxiv.org/abs/2309.07597) - Embedding models
- [Toolformer](https://arxiv.org/abs/2302.04761) - Tool use in LLMs

### Benchmarks
- [FinanceBench](https://arxiv.org/abs/2311.11944) - Financial QA
- [PubMedQA](https://pubmedqa.github.io/) - Biomedical QA
- [CUAD](https://www.atticusprojectai.org/cuad) - Contract Understanding

## Team

- **Contributors**: GarrickPinon, shxwheen, JunjieAraoXiong, aumhirpara2001-stack
- **License**: MIT

## Project Status

| Component | Status |
|-----------|--------|
| RAG System | Active Development |
| LLM Judge Evaluation | Complete |
| Meta-Learning Router | Not Started |
| Multi-Domain Support | Planned |
