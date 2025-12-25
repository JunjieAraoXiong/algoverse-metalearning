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
## 🚀 Quick Start (Cluster)

**1. Process all 367 PDFs (Fast Mode):**
```bash
# Takes ~20 minutes on 1 CPU
python src/ingest.py --fast --chunk-size 1000 --batch-size 20 --data-dir /tmp/junjie_pdfs/
```

**2. Launch Inference Server (Free H100s):**
```bash
# Starts Llama-3-70B on 8x H100s
bash scripts/launch_vllm.sh
```

**3. Run Evaluation:**
```bash
python src/bulk_testing.py --model meta-llama/Meta-Llama-3.1-70B-Instruct
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
│   ├── ingest.py                    # Main ingestion script (PDF -> ChromaDB)
│   ├── metadata_utils.py            # Filename/question metadata extraction
│   ├── bulk_testing.py              # Evaluation framework
│   ├── providers/                   # LLM provider adapters
│   │   ├── base.py
│   │   ├── factory.py
│   │   ├── openai_provider.py
│   │   ├── anthropic_provider.py
│   │   ├── google_provider.py
│   │   ├── deepseek_provider.py
│   │   └── together_provider.py
│   └── retrieval_tools/             # Retrieval pipelines
│       ├── tool_registry.py         # Pipeline builder
│       ├── semantic.py              # Vector similarity
│       ├── hybrid.py                # BM25 + Semantic
│       ├── hybrid_filter.py         # + Metadata filtering
│       └── rerank.py                # Cross-encoder reranking
├── data/
│   └── test_files/finance-bench-pdfs/  # Source PDFs
├── requirements.txt
└── README.md
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
# Database creation (Resumable)
python src/ingest.py --sample 5               # Test on 5 PDFs first
python src/ingest.py --batch-size 10          # Full ingestion (saves every 10 files)

# Evaluation
python src/bulk_testing.py --model claude-sonnet-4-5 --top-k 10 --pipeline hybrid_filter_rerank
python src/bulk_testing.py --model gemini-3-flash --top-k 10   # Cheaper option
python src/bulk_testing.py --subset data/question_sets/financebench_subset_questions.csv
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
| Database not found | Run `python src/ingest.py` first |
| Module not found | Run from project root: `cd /path/to/rag` |
| Rate limit errors | Partial results saved automatically; check API quota |
| Low similarity scores | Increase top-k, verify PDFs match dataset, check CSV sources |
