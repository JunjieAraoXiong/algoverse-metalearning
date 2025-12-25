# Financial RAG System

A Retrieval-Augmented Generation (RAG) system for question answering on financial documents, evaluated on the FinanceBench benchmark. Part of a larger research project exploring meta-learning for cross-domain RAG optimization.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         FINANCIAL RAG SYSTEM                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│  │   INGEST    │───▶│   STORE     │───▶│  RETRIEVE   │───▶│  GENERATE   │  │
│  │  (PDFs)     │    │  (ChromaDB) │    │  (Pipeline) │    │   (LLM)     │  │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘  │
│        │                  │                  │                  │          │
│        ▼                  ▼                  ▼                  ▼          │
│  ┌───────────┐      ┌───────────┐      ┌───────────┐      ┌───────────┐   │
│  │Unstructured│     │ BGE-Large │      │  Hybrid   │      │  Claude   │   │
│  │  hi_res   │      │Embeddings │      │ +Filter   │      │   GPT     │   │
│  │  +OCR     │      │  (FREE)   │      │ +Rerank   │      │  Gemini   │   │
│  └───────────┘      └───────────┘      └───────────┘      └───────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Why RAG?**
- Retrieves relevant documents before generating answers (reduces hallucination)
- Enables answering questions about private/recent data not in the model's training set
- Grounds responses in actual source material

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Create .env file with API keys
echo "TOGETHER_API_KEY=your_key" >> .env
echo "OPENAI_API_KEY=your_key" >> .env

# 3. Add PDFs to data/test_files/finance-bench-pdfs/
# Download from: https://github.com/patronus-ai/financebench/tree/main/pdfs

# 4. Build vector database (with metadata)
python src/create_database_v2.py --sample 5  # Test on 5 PDFs first
python src/create_database_v2.py             # Full ingestion (2-4 hours)

# 5. Run evaluation
python src/bulk_testing.py --pipeline hybrid_filter_rerank --top-k 10
```

## Retrieval Pipeline

```
                              RETRIEVAL PIPELINE
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  Question: "What is 3M's FY2018 capital expenditure?"                       │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ STEP 1: INITIAL RETRIEVAL (k × 3 = 15 chunks)                       │   │
│  │                                                                      │   │
│  │  ┌─────────────┐         ┌─────────────┐                            │   │
│  │  │   BM25      │────┐    │  Semantic   │────┐                       │   │
│  │  │  (keyword)  │    │    │  (vector)   │    │                       │   │
│  │  └─────────────┘    │    └─────────────┘    │                       │   │
│  │                     ▼                       ▼                        │   │
│  │                  ┌──────────────────────────────┐                   │   │
│  │                  │     HYBRID MERGE (50/50)     │                   │   │
│  │                  └──────────────────────────────┘                   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ STEP 2: METADATA FILTER                                              │   │
│  │                                                                      │   │
│  │  Question → extract_metadata() → {company: "3M", year: 2018}        │   │
│  │  15 chunks → filter(company="3M", year=2018) → 8 chunks             │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ STEP 3: RERANK (Cross-Encoder)                                       │   │
│  │                                                                      │   │
│  │  8 chunks → BGE-Reranker(question, chunk) → score → top 5           │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ STEP 4: GENERATION                                                   │   │
│  │                                                                      │   │
│  │  Context = concat(5 chunks) + Question → LLM → Answer               │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Available Pipelines

| Pipeline | Description | Latency | Quality | Best For |
|----------|-------------|---------|---------|----------|
| `semantic` | Pure vector similarity | ~50ms | Low | Quick prototyping |
| `hybrid` | BM25 + Semantic (50/50) | ~100ms | Medium | General use |
| `hybrid_filter` | Hybrid + metadata filtering | ~120ms | High | Domain-specific queries |
| `hybrid_filter_rerank` | Hybrid + filter + cross-encoder | ~300ms | Highest | Production (default) |

**Why Hybrid Search?**
- **BM25**: Catches exact keyword matches ("FY2018", "CapEx")
- **Semantic**: Catches meaning similarity ("revenue" matches "total sales")
- Combined: Best of both worlds

**Why Metadata Filtering?**
- Per-file RAG (51% accuracy) vs shared-store RAG (19%) = **2.7x improvement**
- Prevents retrieving Apple's 2019 data when asking about 3M's 2018 data

## Project Structure

```
rag/
├── src/
│   ├── config.py                    # Central configuration
│   ├── providers/                   # LLM provider adapters
│   │   ├── base.py                  # Abstract LLMProvider class
│   │   ├── factory.py               # get_provider(model_name)
│   │   ├── openai_provider.py       # GPT models
│   │   ├── anthropic_provider.py    # Claude models
│   │   └── google_provider.py       # Gemini models
│   ├── retrieval_tools/             # Retrieval pipelines
│   │   ├── tool_registry.py         # Pipeline builder
│   │   ├── semantic.py              # Vector similarity
│   │   ├── hybrid.py                # BM25 + Semantic
│   │   ├── metadata_filter.py       # Company/year filtering
│   │   └── rerank.py                # Cross-encoder reranking
│   ├── metadata_utils.py            # Filename/question metadata extraction
│   ├── create_database_v2.py        # Improved ingestion with metadata
│   ├── bulk_testing.py              # Evaluation framework
│   └── meta_learning/               # Meta-learning components (WIP)
├── evaluation/                      # Metrics (semantic similarity, LLM judge)
├── dataset_adapters/                # Dataset loaders (FinanceBench, PubMedQA)
├── data/test_files/finance-bench-pdfs/  # 367 PDFs (636 MB)
├── chroma/                          # Vector database
└── bulk_runs/                       # Evaluation results
```

## Benchmark Results

**FinanceBench Context:**

| Approach | Accuracy | Notes |
|----------|----------|-------|
| Baseline RAG (shared-store) | 19% | GPT-4-Turbo, 81% wrong/refused |
| Improved RAG (Ragie) | 27% | Hybrid search, better ingestion |
| Per-file RAG | 51% | Retrieve from correct document only |
| Long-context (100k+ tokens) | ~70-80% | Expensive, high latency |

**Our Current Scores:**

| Question Type | Score | Target |
|---------------|-------|--------|
| metrics-generated | 0.35 | 0.55+ |
| domain-relevant | 0.60 | 0.70+ |
| novel-generated | 0.53 | 0.65+ |
| **Overall** | **0.495** | **0.65+** |

## Commands

```bash
# Database creation
python src/create_database_v2.py              # Full ingestion with metadata
python src/create_database_v2.py --sample 5   # Test on 5 PDFs

# Evaluation
python src/bulk_testing.py --model claude-sonnet-4-5 --top-k 10 --pipeline hybrid_filter_rerank
python src/bulk_testing.py --model gemini-3-flash --top-k 10   # Cheaper option
python src/bulk_testing.py --subset data/question_sets/financebench_subset_questions.csv

# Pipeline tests
python tests/test_pipelines.py  # Iterate over all pipeline configs
```

## Supported Models

| Provider | Models | Cost (150 questions) |
|----------|--------|---------------------|
| Anthropic | Claude 4.5 Sonnet/Opus | ~$2.00 |
| OpenAI | GPT-5.2 | ~$3.20 |
| Google | Gemini 3 Flash/Pro | ~$0.07 |
| DeepSeek | DeepSeek Chat | ~$0.05 |
| Together | Llama 3.1 70B | ~$0.50 |

## Configuration

All settings in `src/config.py`:

```python
# Embeddings (FREE local options)
EMBEDDINGS = {
    "bge-large": "BAAI/bge-large-en-v1.5",  # Recommended
    "bge-base": "BAAI/bge-base-en-v1.5",
    "openai-large": "text-embedding-3-large",  # Paid
}

# Rerankers
RERANKERS = {
    "bge-reranker": "BAAI/bge-reranker-large",  # Recommended, FREE
    "ms-marco": "cross-encoder/ms-marco-MiniLM-L-6-v2",
}
```

## Research Direction: Meta-Learning

This project is part of a larger research effort to build a **meta-learning system** that learns to select the optimal retrieval pipeline for any question type, generalizing across domains:

- **Finance**: FinanceBench (tables, numbers, 10-K/10-Q filings)
- **Healthcare**: PubMedQA (dense medical prose)
- **Legal**: CUAD (contract clauses, extractive QA)

**Key insight**: Different question types need different pipelines. Instead of optimizing ONE pipeline, we optimize the CHOICE of pipeline.

See `ROADMAP.md` for detailed implementation plan.

## Glossary

| Term | Definition |
|------|------------|
| **RAG** | Retrieval-Augmented Generation - retrieve docs, then generate answer |
| **Embedding** | Dense vector representation of text (~1024 dimensions) |
| **ChromaDB** | Open-source vector database for storing/searching embeddings |
| **BM25** | Classic keyword-based search algorithm |
| **Cross-Encoder** | Model that scores (query, doc) pairs for relevance |
| **Reranker** | Cross-encoder used to re-order retrieved documents |
| **Chunk** | A segment of text from a document (~1000-2000 chars) |
| **10-K** | Annual SEC filing (comprehensive financial report) |
| **10-Q** | Quarterly SEC filing |

## References

- [FinanceBench Paper](https://arxiv.org/abs/2311.11944)
- [FinanceBench GitHub](https://github.com/patronus-ai/financebench)
- [Ragie FinanceBench Results](https://www.ragie.ai/blog/ragie-outperformed-financebench)
- [Patronus AI Docs](https://docs.patronus.ai/docs/research_and_differentiators/financebench)

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Database not found | Run `python src/create_database_v2.py` first |
| Module not found | Run from project root: `cd /path/to/rag` |
| Rate limit errors | Partial results saved automatically; check API quota |
| Low similarity scores | Increase top-k, verify PDFs match dataset, check CSV sources |
