# Financial RAG → Meta-Learning Roadmap

---

## Work Completed Summary

### Foundation Work (Done Before Phases)

| Work Item | Time Spent | What Was Done | Files Created/Modified |
|-----------|------------|---------------|------------------------|
| **Project Structure** | ~2 hrs | Set up directory structure, config patterns, coding guidelines | `.claude/CLAUDE.md`, `NOTES.md` |
| **Central Configuration** | ~1 hr | Centralized all defaults, model configs, embedding configs | `src/config.py` |
| **LLM Provider Abstraction** | ~1.5 hrs | Abstract base class + factory for multiple providers | `src/providers/*.py` (5 files) |
| **Retrieval Pipeline Framework** | ~2 hrs | Modular retrieval with semantic, hybrid, filter, rerank | `src/retrieval_tools/*.py` (6 files) |
| **Evaluation Framework** | ~1 hr | Bulk testing runner with metrics and reporting | `src/bulk_testing.py`, `evaluation/` |
| **Dataset Adapters** | ~1 hr | Pluggable dataset loaders for FinanceBench, PubMedQA | `dataset_adapters/*.py` |
| **PDF Download** | ~30 min | Downloaded 367/368 FinanceBench PDFs (636 MB) | `scripts/download_financebench_pdfs.py` |
| **Documentation** | ~1 hr | Project notes, guidelines, this roadmap | `NOTES.md`, `ROADMAP.md` |

**Total Foundation Work: ~10 hours**

### What's Built and Working

```
✅ COMPLETE                    ⚠️ NEEDS WORK                 🔲 NOT STARTED
─────────────────────────────────────────────────────────────────────────
✅ Project structure           ⚠️ ChromaDB (old, no metadata) 🔲 Meta-router
✅ Config system               ⚠️ Table detection             🔲 Oracle labels
✅ Provider abstraction        ⚠️ Question classifier         🔲 PubMedQA setup
✅ Retrieval pipelines                                        🔲 CUAD setup
✅ Reranker integration
✅ Bulk testing framework
✅ Metadata extraction (filename)
✅ V2 ingestion script (untested on full set)
✅ 367 PDFs downloaded
```

---

## System Architecture Overview

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
│  │Unstructured│     │ BGE-Large │      │  Hybrid   │      │Claude 4.5 │   │
│  │  hi_res   │      │Embeddings │      │ +Filter   │      │ GPT 5.2   │   │
│  │  +OCR     │      │  (FREE)   │      │ +Rerank   │      │ Gemini 3  │   │
│  └───────────┘      └───────────┘      └───────────┘      └───────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Technical Notes: RAG Architecture

**What is RAG?**
- **R**etrieval-**A**ugmented **G**eneration - instead of asking an LLM to answer from memory, we first retrieve relevant documents, then generate an answer grounded in those documents
- Reduces hallucination because the LLM has real source material
- Enables answering questions about private/recent data not in training set

**Why these 4 stages?**
1. **Ingest** - PDFs aren't searchable. We must extract text, split into chunks, and create searchable representations
2. **Store** - Vector databases (ChromaDB) enable fast similarity search over millions of chunks
3. **Retrieve** - Find the most relevant chunks for a question (this is where most errors happen!)
4. **Generate** - LLM synthesizes an answer from retrieved context

**Why local embeddings (BGE) instead of OpenAI?**
- OpenAI charges per token for BOTH ingestion AND every query
- 367 PDFs × ~500 chunks × 2 passes = expensive
- BGE-large is comparable quality and completely FREE (runs locally)
- Trade-off: slightly slower, uses local CPU/GPU

**Why multiple LLM providers?**
- Different models have different strengths (Claude for reasoning, Gemini for speed, DeepSeek for cost)
- Provider abstraction lets us swap models without changing code
- Enables A/B testing different models on same questions

---

## What We Built (Detailed Component Map)

```
rag/
├── src/
│   ├── config.py                    ✅ BUILT - Central configuration
│   │   ├── EMBEDDINGS               - 6 models (4 free local, 2 paid OpenAI)
│   │   ├── PROVIDERS                - 5 LLM providers (OpenAI, Anthropic, Google, Together, DeepSeek)
│   │   ├── RERANKERS                - 3 reranker options
│   │   └── DEFAULTS                 - All default settings
│   │
│   ├── providers/                   ✅ BUILT - LLM abstraction layer
│   │   ├── base.py                  - Abstract LLMProvider class
│   │   ├── factory.py               - get_provider(model_name) with caching
│   │   ├── openai_provider.py       - GPT-4o, GPT-5.2
│   │   ├── anthropic_provider.py    - Claude 4.5 Sonnet/Opus
│   │   └── google_provider.py       - Gemini 3 Flash/Pro
│   │
│   ├── retrieval_tools/             ✅ BUILT - Retrieval pipelines
│   │   ├── tool_registry.py         - Pipeline builder & registry
│   │   ├── semantic.py              - Pure vector similarity
│   │   ├── hybrid.py                - BM25 + Semantic (50/50)
│   │   ├── metadata_filter.py       - Filter by company/year
│   │   └── rerank.py                - Cross-encoder reranking (BGE)
│   │
│   ├── metadata_utils.py            ✅ BUILT - Metadata extraction
│   │   ├── parse_filename()         - Extract company/year/doc_type from PDF name
│   │   ├── extract_metadata_from_question()  - Extract from questions
│   │   └── filter_chunks_by_metadata()       - Filter retrieved chunks
│   │
│   ├── create_database.py           ✅ BUILT - Basic ingestion
│   ├── create_database_element_based.py     ✅ BUILT - Element-aware ingestion
│   ├── create_database_v2.py        ✅ BUILT - Improved ingestion with metadata
│   │
│   ├── bulk_testing.py              ✅ BUILT - Evaluation framework
│   │   ├── BulkTestConfig           - Configuration dataclass
│   │   ├── BulkTestRunner           - Main runner class
│   │   ├── process_single_question()- RAG pipeline execution
│   │   └── run_bulk_test()          - Batch evaluation
│   │
│   └── meta_learning/               🔲 STUB - Not yet implemented
│       ├── router.py
│       ├── oracle_labels.py
│       ├── episodes.py
│       ├── meta_trainer.py
│       └── evaluator.py
│
├── evaluation/                      ✅ BUILT - Metrics
│   ├── metrics.py                   - embedding_similarity, aggregate_metrics
│   └── llm_judge.py                 - LLM-as-a-Judge evaluation
│
├── dataset_adapters/                ✅ BUILT - Dataset loaders
│   ├── base.py                      - BaseDatasetAdapter
│   ├── financebench.py              - FinanceBench loader
│   └── pubmedqa.py                  - PubMedQA loader
│
├── data/test_files/
│   └── finance-bench-pdfs/          ✅ 367 PDFs downloaded (636 MB)
│
└── chroma/                          ⚠️  OLD - Needs rebuild with v2 ingestion
```

---

## Retrieval Pipeline Flow (What Happens on Each Question)

```
                              RETRIEVAL PIPELINE
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  Question: "What is 3M's FY2018 capital expenditure?"                       │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ STEP 1: INITIAL RETRIEVAL (k × factor = 5 × 3 = 15 chunks)          │   │
│  │                                                                      │   │
│  │  ┌─────────────┐         ┌─────────────┐                            │   │
│  │  │   BM25      │────┐    │  Semantic   │────┐                       │   │
│  │  │  (keyword)  │    │    │  (vector)   │    │                       │   │
│  │  └─────────────┘    │    └─────────────┘    │                       │   │
│  │                     ▼                       ▼                        │   │
│  │                  ┌──────────────────────────────┐                   │   │
│  │                  │     HYBRID MERGE (50/50)     │                   │   │
│  │                  │     → 15 candidate chunks    │                   │   │
│  │                  └──────────────────────────────┘                   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ STEP 2: METADATA FILTER                                              │   │
│  │                                                                      │   │
│  │  Question → extract_metadata() → {company: "3M", year: 2018}        │   │
│  │                                                                      │   │
│  │  15 chunks → filter(company="3M", year=2018) → 8 chunks             │   │
│  │                                                                      │   │
│  │  ⚠️  CURRENT ISSUE: Chunk metadata incomplete (no company/year)     │   │
│  │  ✅ FIX: create_database_v2.py adds this metadata                   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ STEP 3: RERANK (Cross-Encoder)                                       │   │
│  │                                                                      │   │
│  │  Model: BAAI/bge-reranker-large                                     │   │
│  │                                                                      │   │
│  │  8 chunks → CrossEncoder(question, chunk) → score → top 5           │   │
│  │                                                                      │   │
│  │  Output: 5 most relevant chunks                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ STEP 4: GENERATION                                                   │   │
│  │                                                                      │   │
│  │  Context = concat(5 chunks)                                         │   │
│  │  Prompt = system_prompt + context + question                        │   │
│  │  LLM = Claude 4.5 Sonnet (or GPT-5.2, Gemini 3)                    │   │
│  │                                                                      │   │
│  │  → Answer: "$1,577 million"                                         │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Technical Notes: Retrieval Pipeline

**Why retrieve more than we need (k × factor)?**
- If we want 5 final chunks, we retrieve 15 first (factor = 3)
- This gives filtering and reranking room to work
- Without over-retrieval, filtering might leave us with 0 relevant chunks

**What is BM25?**
- Classic keyword-based search algorithm (like Google circa 2000)
- Matches exact terms: "FY2018" matches "FY2018" but not "fiscal year 2018"
- Fast, interpretable, great for specific terms like company names and years
- Weakness: misses synonyms and semantic similarity

**What is Semantic (Vector) Search?**
- Embeds question and chunks into high-dimensional vectors
- Finds chunks with similar "meaning" even if words differ
- "What is the revenue?" matches "Total sales were $X million"
- Weakness: can miss exact keyword matches, sometimes retrieves vaguely similar but wrong content

**Why Hybrid (50/50)?**
- Combines strengths of both: BM25 catches exact matches, semantic catches meaning
- Research shows hybrid consistently outperforms either alone
- The 50/50 weight is a reasonable default; could be tuned per domain

**What is a Cross-Encoder Reranker?**
- Takes (question, chunk) pairs and scores relevance 0-1
- Much more accurate than embedding similarity but ~100x slower
- That's why we only rerank top 15, not all 50,000 chunks
- BGE-reranker-large is SOTA for English, runs locally (FREE)

**Why is metadata filtering so important?**
- FinanceBench questions are specific: "3M's FY2018 CapEx"
- Without filtering, we might retrieve Adobe's 2019 data instead
- Per-file RAG (51%) vs shared-store RAG (19%) shows 2.7x improvement just from filtering!

---

## Available Retrieval Pipelines

```
┌────────────────────────────────────────────────────────────────────────────┐
│                        PIPELINE OPTIONS                                     │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. semantic                                                                │
│     ┌──────────┐                                                           │
│     │ Semantic │──────────────────────────────────────▶ Results            │
│     └──────────┘                                                           │
│     Pure vector similarity. Fast, but misses keyword matches.              │
│                                                                             │
│  2. hybrid                                                                  │
│     ┌──────────┐   ┌──────────┐                                           │
│     │   BM25   │──▶│  Merge   │──────────────────────▶ Results            │
│     │ Semantic │──▶│  50/50   │                                           │
│     └──────────┘   └──────────┘                                           │
│     Combines keyword + semantic. Better recall.                            │
│                                                                             │
│  3. hybrid_filter                                                           │
│     ┌──────────┐   ┌──────────┐   ┌──────────┐                            │
│     │  Hybrid  │──▶│ Metadata │──▶│  Top-K   │────────▶ Results            │
│     └──────────┘   │  Filter  │   └──────────┘                            │
│                    └──────────┘                                            │
│     Filters by company/year before taking top-K.                           │
│                                                                             │
│  4. hybrid_filter_rerank  ← DEFAULT (RECOMMENDED)                          │
│     ┌──────────┐   ┌──────────┐   ┌──────────┐                            │
│     │  Hybrid  │──▶│ Metadata │──▶│ Rerank   │────────▶ Results            │
│     └──────────┘   │  Filter  │   │(CrossEnc)│                            │
│                    └──────────┘   └──────────┘                            │
│     Best quality. Reranker scores relevance precisely.                     │
│                                                                             │
└────────────────────────────────────────────────────────────────────────────┘
```

### Technical Notes: Pipeline Selection

**When to use each pipeline:**

| Pipeline | Best For | Latency | Quality |
|----------|----------|---------|---------|
| `semantic` | Quick prototyping, simple questions | ~50ms | Low |
| `hybrid` | General use, mixed question types | ~100ms | Medium |
| `hybrid_filter` | Domain-specific with clear metadata | ~120ms | High |
| `hybrid_filter_rerank` | Production, accuracy-critical | ~300ms | Highest |

**Why is `hybrid_filter_rerank` the default?**
- Financial questions have clear metadata (company, year) making filtering effective
- Reranking catches subtle relevance that embedding similarity misses
- The 200ms extra latency is acceptable for accuracy-critical applications
- For real-time chat, might drop to `hybrid_filter` to reduce latency

**Trade-offs:**
- More stages = higher accuracy but slower
- Reranking is the biggest latency hit (~200ms for 15 chunks)
- Filtering requires good metadata; if metadata is wrong, it hurts instead of helps

---

## Ingestion Pipeline (create_database_v2.py)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         INGESTION FLOW (V2)                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  INPUT: 3M_2018_10K.pdf                                                     │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ STEP 1: PARSE FILENAME                                               │   │
│  │                                                                      │   │
│  │  "3M_2018_10K.pdf" → parse_filename() →                             │   │
│  │  {                                                                   │   │
│  │    company: "3M",                                                    │   │
│  │    year: 2018,                                                       │   │
│  │    doc_type: "10K",                                                  │   │
│  │    fiscal_period: "FY2018"                                          │   │
│  │  }                                                                   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ STEP 2: PDF PARSING (Unstructured.io)                                │   │
│  │                                                                      │   │
│  │  partition_pdf(                                                      │   │
│  │    strategy="hi_res",         # High-quality OCR                    │   │
│  │    infer_table_structure=True # Detect tables                       │   │
│  │  )                                                                   │   │
│  │                                                                      │   │
│  │  → Elements: [Title, Text, Table, Text, Table, ...]                 │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ STEP 3: SEMANTIC CHUNKING                                            │   │
│  │                                                                      │   │
│  │  chunk_by_title(                                                     │   │
│  │    max_characters=2000,                                             │   │
│  │    combine_text_under_n_chars=1000                                  │   │
│  │  )                                                                   │   │
│  │                                                                      │   │
│  │  → Groups content by section headers                                │   │
│  │  → Keeps tables intact (doesn't split mid-table)                    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ STEP 4: ENRICH METADATA                                              │   │
│  │                                                                      │   │
│  │  Each chunk gets:                                                    │   │
│  │  {                                                                   │   │
│  │    company: "3M",              # From filename                      │   │
│  │    year: 2018,                 # From filename                      │   │
│  │    doc_type: "10K",            # From filename                      │   │
│  │    fiscal_period: "FY2018",    # Derived                            │   │
│  │    element_type: "table",      # From Unstructured                  │   │
│  │    page_number: 45,            # From Unstructured                  │   │
│  │    source_file: "3M_2018_10K.pdf"                                   │   │
│  │  }                                                                   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ STEP 5: EMBED & STORE                                                │   │
│  │                                                                      │   │
│  │  Model: BAAI/bge-large-en-v1.5 (FREE, local)                        │   │
│  │  Store: ChromaDB (persistent)                                       │   │
│  │                                                                      │   │
│  │  367 PDFs → ~50,000+ chunks → ChromaDB                              │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Technical Notes: Ingestion

**Why Unstructured.io with `hi_res` strategy?**
- Financial documents have complex layouts: tables, multi-column text, headers/footers
- `hi_res` uses OCR + layout detection to properly extract tables
- Alternative `fast` strategy just extracts text linearly (loses table structure)
- Trade-off: `hi_res` is ~10x slower but much better for financial docs

**Why `chunk_by_title` instead of fixed-size chunking?**
- Fixed 1000-char chunks can split mid-sentence or mid-table
- `chunk_by_title` respects document structure (sections, headers)
- Tables stay intact as single chunks (critical for financial data!)
- Produces semantically coherent chunks that embed better

**Why extract metadata from filenames?**
- FinanceBench PDFs follow pattern: `COMPANY_YEAR_DOCTYPE.pdf`
- Parsing this gives us structured metadata for filtering
- Without it, we can only match on text content (less precise)
- Example: `3M_2018_10K.pdf` → `{company: "3M", year: 2018, doc_type: "10K"}`

**What is ChromaDB?**
- Open-source vector database (like Pinecone but free/local)
- Stores embeddings + metadata + original text
- Supports filtering by metadata fields
- Persists to disk so we don't re-embed every time

**Embedding dimension matters:**
- BGE-large: 1024 dimensions
- Higher dimensions = more expressive but larger storage
- 50,000 chunks × 1024 dims × 4 bytes = ~200MB (manageable)

---

## Current State

| Metric | Our Score | Target |
|--------|-----------|--------|
| Semantic Similarity | 0.495 | 0.65+ |
| metrics-generated | 0.35 | 0.55+ |
| domain-relevant | 0.60 | 0.70+ |
| novel-generated | 0.53 | 0.65+ |

## Benchmark Context (FinanceBench)

| Approach | Accuracy | Notes |
|----------|----------|-------|
| Baseline RAG (shared-store) | 19% | GPT-4-Turbo, 81% wrong/refused |
| Improved RAG (Ragie) | 27% | Hybrid search, better ingestion |
| Per-file RAG (single-store) | 51% | Retrieve from correct document only |
| Long-context (100k+ tokens) | ~70-80% | Expensive, high latency |

**Key insight**: Per-file retrieval (51%) is 2.7x better than shared-store (19%). Our goal is to match or beat per-file RAG accuracy.

### Technical Notes: FinanceBench Benchmark

**What is FinanceBench?**
- 10,231 questions about publicly traded companies from 10-K/10-Q filings
- 150 questions with human-annotated answers (what we test on)
- Questions span: numerical extraction, analysis, multi-hop reasoning
- Created by Patronus AI to test RAG systems on real financial documents

**Why is it hard?**
- Financial documents are dense: 100+ page 10-Ks with complex tables
- Questions require precise numerical answers ("$1,577 million" not "$1.5 billion")
- Wrong company/year data is worse than no answer (hallucination risk)
- Many questions need calculation (ratios, YoY changes)

**What do the accuracy numbers mean?**
- **19% baseline**: Just embedding search + GPT-4 = 81% wrong/refused
- **51% per-file**: When you give the correct document only, accuracy jumps 2.7x
- **70-80% long-context**: Feed entire 100-page doc into 100k context window (expensive!)

**Our evaluation metric: Semantic Similarity**
- We measure embedding similarity between predicted and gold answers
- 0.495 avg score ≈ "sometimes correct, often partially correct"
- Not the same as accuracy (binary right/wrong) but correlates
- Scores by question type reveal where we fail (metrics-generated: 0.35)

---

---

## Required Reading: FinanceBench Repository

Before diving deeper, familiarize yourself with the benchmark:

**GitHub:** https://github.com/patronus-ai/financebench

| File/Folder | What It Contains | Priority |
|-------------|------------------|----------|
| `README.md` | Overview, data loading code, evaluation methodology | ⭐ **READ FIRST** |
| `data/financebench_open_source.jsonl` | 150 annotated QA pairs with answers + evidence | ⭐ **Essential** |
| `data/financebench_document_information.jsonl` | Document metadata (which PDFs map to which companies) | ⭐ **Essential** |
| `evaluation_playground.ipynb` | Interactive notebook to explore the data | Helpful |
| `results/` | Their model evaluation results (GPT-4, Claude, etc.) | Reference |
| `pdfs/` | Source PDFs (we already downloaded 367 of these) | Already have |
| [arXiv Paper](https://arxiv.org/abs/2311.11944) | Full methodology, all results, analysis | ⭐ **Read for context** |

**Key code from their README to load data:**
```python
import json
questions = [json.loads(line) for line in open("financebench_open_source.jsonl")]
doc_info = [json.loads(line) for line in open("financebench_document_information.jsonl")]

# Each question has:
# - question: the question text
# - answer: gold answer
# - evidence: text from source doc
# - doc_name: which PDF it comes from
# - question_type: metrics-generated, domain-relevant, or novel-generated
```

---

## Phase 1: Improved Ingestion (Current)

**Status**: 90% complete | **Estimated Time Remaining**: ~2-4 hours (mostly waiting for ingestion)

### What This Phase Accomplishes
Transform raw PDFs into a searchable vector database with rich metadata, enabling accurate filtering by company/year during retrieval.

### Completed Tasks ✅

| Task | Time Spent | What Was Done | Files Modified |
|------|------------|---------------|----------------|
| Design metadata schema | ~30 min | Defined `DocumentMetadata` dataclass with company, year, doc_type, quarter, fiscal_period | `src/metadata_utils.py` |
| Implement filename parser | ~20 min | `parse_filename()` extracts metadata from `COMPANY_YEAR_DOCTYPE.pdf` pattern | `src/metadata_utils.py` |
| Create v2 ingestion script | ~1 hr | Full pipeline: parse filename → Unstructured hi_res → chunk_by_title → enrich metadata → embed → store | `src/create_database_v2.py` |
| Add table→markdown conversion | ~20 min | `html_table_to_markdown()` converts Unstructured's HTML tables to markdown | `src/create_database_v2.py` |
| Add element type tagging | ~10 min | `get_element_type()` maps Unstructured elements to table/prose/title/etc | `src/create_database_v2.py` |
| Install OCR dependencies | ~10 min | `brew install poppler tesseract` for hi_res PDF parsing | System |
| Test on sample PDF | ~15 min | Verified 486 chunks from 3M_2016_10K.pdf with correct metadata | Tested |

### Remaining Tasks ⬜

| Task | Estimated Time | What To Do | Command/Notes |
|------|----------------|------------|---------------|
| Run full ingestion | 2-4 hrs (mostly waiting) | Process all 367 PDFs through v2 pipeline | `python src/create_database_v2.py` |
| Verify metadata coverage | ~15 min | Check all chunks have company/year metadata | Query ChromaDB, spot check |
| (Optional) Tune table detection | ~1 hr | Currently 0 tables detected; may need to adjust Unstructured params | Investigate `infer_table_structure` |

### Verified Output (from test run)
```
Metadata: {
  'company': '3M',
  'year': 2016,
  'doc_type': '10K',
  'fiscal_period': 'FY2016',
  'element_type': 'other',
  'source_file': '3M_2016_10K.pdf'
}
```

### Key Files
| File | Purpose |
|------|---------|
| `src/create_database_v2.py` | Main ingestion script (run this) |
| `src/metadata_utils.py` | Filename parsing + question metadata extraction |
| `src/create_database.py` | Old basic ingestion (deprecated) |
| `src/create_database_element_based.py` | Previous attempt (superseded by v2) |

### How To Run
```bash
# Test on small sample first
python src/create_database_v2.py --sample 5

# Full ingestion (will take 2-4 hours)
python src/create_database_v2.py

# Check output
ls -la chroma/  # Should see new database files
```

---

## Phase 2: Per-File Retrieval Strategy

**Status**: 0% complete | **Estimated Time**: 3-4 hours

### What This Phase Accomplishes
Implement strict metadata filtering so questions about "3M FY2018" only retrieve chunks from the 3M 2018 10-K, not from other companies or years.

### Why This Matters
- Per-file RAG (51%) vs shared-store RAG (19%) = **2.7x improvement**
- This is the single biggest lever for accuracy improvement
- Most RAG errors come from retrieving irrelevant context

### Strategy A: Metadata-Filtered Retrieval (Recommended)
```
Question → Extract {company, year} → Filter ChromaDB → Retrieve → Generate
```

### Strategy B: Two-Stage Retrieval (Alternative)
```
Question → Stage 1: Identify document → Stage 2: Retrieve from document → Generate
```

### All Tasks

| Task | Estimated Time | Status | What To Do |
|------|----------------|--------|------------|
| Enhance `extract_metadata_from_question()` | ~1 hr | ⬜ | Improve company/year extraction from question text (regex + NER) |
| Update `filter_chunks_by_metadata()` | ~30 min | ⬜ | Make filtering stricter, add exact match option |
| Add ChromaDB native filtering | ~1 hr | ⬜ | Use `db.similarity_search(filter={"company": "3M", "year": 2018})` |
| Implement fallback strategy | ~30 min | ⬜ | If filter returns 0 results, relax constraints progressively |
| Add logging for filter effectiveness | ~20 min | ⬜ | Track how many chunks filtered out, catch over-filtering |
| Benchmark per-file vs shared | ~1 hr | ⬜ | Run same questions with and without filtering, compare scores |

### Key Files to Modify
| File | Changes Needed |
|------|----------------|
| `src/metadata_utils.py` | Improve `extract_metadata_from_question()` |
| `src/retrieval_tools/metadata_filter.py` | Add strict filtering, ChromaDB native filter |
| `src/retrieval_tools/tool_registry.py` | Wire up new filtering logic |

### How To Test
```bash
# After changes, run on subset
python src/bulk_testing.py --subset data/question_sets/financebench_subset_questions.csv --pipeline hybrid_filter

# Compare scores with and without filtering
python src/bulk_testing.py --pipeline hybrid          # No filter
python src/bulk_testing.py --pipeline hybrid_filter   # With filter
```

### Technical Notes: Per-File Retrieval

**Why does per-file work so much better?**
- Shared-store problem: "What is 3M's revenue?" might retrieve Apple's revenue chunk (similar words!)
- Per-file solution: First identify the document, then search within it
- This mimics how humans search: find the right report, then Ctrl+F within it

**How metadata filtering achieves this:**
```python
# Without filtering (shared-store):
results = db.similarity_search("3M FY2018 revenue", k=5)
# Might return: [Adobe_2019, 3M_2017, Apple_2018, 3M_2018, Microsoft_2018]

# With filtering (per-file equivalent):
results = db.similarity_search(
    "3M FY2018 revenue",
    k=5,
    filter={"company": "3M", "year": 2018}  # ← This is the key!
)
# Returns: [3M_2018_chunk1, 3M_2018_chunk2, ...] (all from correct doc)
```

**Fallback strategy matters:**
- What if we can't extract company/year from question?
- What if the metadata is wrong?
- Need graceful degradation: try filtered → fall back to unfiltered

**Two-stage vs single-stage:**
- Single-stage: Filter ChromaDB directly (what we do now)
- Two-stage: First classify document, then query that document's index
- Two-stage is cleaner but requires separate indices per document (more complex)

---

## Phase 3: Question-Type Routing

**Status**: 0% complete | **Estimated Time**: 4-5 hours

### What This Phase Accomplishes
Route different question types to specialized retrieval/generation strategies. Our weakest area (metrics-generated: 0.35) needs table-aware handling.

### Why This Matters
| Type | Current Score | Target | Gap |
|------|---------------|--------|-----|
| metrics-generated | 0.35 | 0.55+ | **Biggest opportunity** |
| domain-relevant | 0.60 | 0.70+ | Medium |
| novel-generated | 0.53 | 0.65+ | Medium |

### All Tasks

| Task | Estimated Time | Status | What To Do |
|------|----------------|--------|------------|
| Build question classifier | ~1.5 hr | ⬜ | Classify question → {metrics, domain, novel} using keywords/regex or small model |
| Create metrics-specific prompt | ~45 min | ⬜ | Structured extraction: "Find the exact number in the context for X" |
| Create domain-specific prompt | ~30 min | ⬜ | Chain-of-thought: "Analyze the data and reason step by step" |
| Create novel-specific prompt | ~30 min | ⬜ | Multi-hop: "Consider multiple factors and synthesize" |
| Add table-priority retrieval | ~1 hr | ⬜ | For metrics questions, boost chunks with `element_type="table"` |
| Add calculation verification | ~45 min | ⬜ | Post-generation check: "Does this number appear in context?" |
| Wire up routing in pipeline | ~30 min | ⬜ | Modify `bulk_testing.py` to use classifier → select prompt |

### Key Files to Create/Modify
| File | Purpose |
|------|---------|
| `src/question_classifier.py` | **NEW** - Classify question type |
| `src/prompts/` | **NEW** - Directory for prompt templates |
| `src/prompts/metrics_prompt.py` | Numeric extraction prompt |
| `src/prompts/domain_prompt.py` | Analytical reasoning prompt |
| `src/prompts/novel_prompt.py` | Multi-hop synthesis prompt |
| `src/bulk_testing.py` | Add routing logic |

### Example Question Classification
```python
def classify_question(question: str) -> str:
    q_lower = question.lower()

    # Metrics indicators
    if any(w in q_lower for w in ['what is the', 'how much', 'what was the',
                                   'capex', 'revenue', 'ratio', 'margin', '$']):
        return 'metrics-generated'

    # Domain indicators
    if any(w in q_lower for w in ['is it', 'does', 'should', 'capital-intensive',
                                   'healthy', 'risk', 'outlook']):
        return 'domain-relevant'

    # Novel/complex
    if any(w in q_lower for w in ['if we', 'excluding', 'trend', 'compare',
                                   'which segment', 'what drove']):
        return 'novel-generated'

    return 'domain-relevant'  # Default
```

### How To Test
```bash
# Run with question-type breakdown
python src/bulk_testing.py --model claude-sonnet-4-5 --top-k 10

# Check results by question type
# Output CSV will have question_type column for analysis
```

### Technical Notes: Question Types

**Why does question type matter?**
Different questions need different retrieval AND generation strategies:

**1. Metrics-Generated (our weakest: 0.35)**
```
Example: "What is the FY2018 capital expenditure amount for 3M?"
Expected: "$1,577 million"

Why hard:
- Need to find exact table row/cell
- Must get units right (millions vs billions)
- LLM might hallucinate plausible-sounding numbers

Better strategy:
- Prioritize table chunks (filter by element_type="table")
- Use structured extraction prompting
- Verify: "Does this number appear verbatim in context?"
```

**2. Domain-Relevant (decent: 0.60)**
```
Example: "Is 3M a capital-intensive business based on FY2022 data?"
Expected: "Yes, based on high PP&E/Assets ratio of X%..."

Why medium:
- Needs interpretation, not just extraction
- Requires domain knowledge (what makes a business "capital-intensive"?)
- Answer is synthesized, not copied

Better strategy:
- Retrieve more context (longer chunks)
- Use chain-of-thought prompting
- Include domain definitions in prompt
```

**3. Novel-Generated (OK: 0.53)**
```
Example: "If we exclude M&A impact, which segment dragged down 3M's growth?"
Expected: "Safety & Industrial segment, excluding acquisitions..."

Why medium:
- Requires multi-hop reasoning
- May need data from multiple sections
- Counterfactual ("if we exclude...")

Better strategy:
- Multi-query retrieval (rephrase question multiple ways)
- Retrieve from multiple document sections
- Explicit reasoning steps in prompt
```

**Question classifier approach:**
- Train small classifier on question text → type
- Or use regex/keyword rules (simpler, often good enough)
- Route to different pipeline/prompt based on type

---

## Phase 4: Evaluation & Validation

**Status**: 0% complete | **Estimated Time**: 2-3 hours

### What This Phase Accomplishes
Rigorously evaluate our improved RAG system, measure accuracy (not just similarity), and document what works.

### Success Criteria
| Metric | Target | Notes |
|--------|--------|-------|
| Overall accuracy | ≥ 50% | Match per-file RAG baseline |
| metrics-generated | ≥ 40% | From current 0.35 similarity |
| domain-relevant | ≥ 65% | From current 0.60 similarity |
| novel-generated | ≥ 55% | From current 0.53 similarity |

### All Tasks

| Task | Estimated Time | Status | What To Do |
|------|----------------|--------|------------|
| Run full 150-question eval | ~30 min (run time) | ⬜ | `python src/bulk_testing.py --model claude-sonnet-4-5` |
| Add accuracy metric | ~30 min | ⬜ | Binary correct/incorrect based on gold answer match |
| Analyze by question type | ~30 min | ⬜ | Group results by metrics/domain/novel, identify patterns |
| Analyze by company/year | ~20 min | ⬜ | Check if certain companies/years perform worse |
| Compare vs baseline | ~20 min | ⬜ | Document improvement over initial 0.495 score |
| Document failure modes | ~30 min | ⬜ | Categorize errors: wrong doc, wrong number, hallucination, etc. |
| Write evaluation report | ~30 min | ⬜ | Summarize findings for future reference |

### Key Commands
```bash
# Full evaluation with best model
python src/bulk_testing.py --model claude-sonnet-4-5 --top-k 10 --pipeline hybrid_filter_rerank

# Quick evaluation with cheaper model (for iteration)
python src/bulk_testing.py --model gemini-3-flash --top-k 10

# Results will be in bulk_runs/ directory
ls bulk_runs/*.csv
ls bulk_runs/*.json  # Summary stats
```

### Evaluation Output Format
```
bulk_runs/
├── 2024-12-12_financebench_claude45-sonnet_k10_t0.csv   # Full results
└── 2024-12-12_financebench_claude45-sonnet_k10_t0.json  # Summary
```

### How To Analyze Results
```python
import pandas as pd
df = pd.read_csv('bulk_runs/LATEST_RESULTS.csv')

# Overall score
print(f"Mean similarity: {df['semantic_similarity'].mean():.3f}")

# By question type
print(df.groupby('question_type')['semantic_similarity'].mean())

# Worst performing questions
print(df.nsmallest(10, 'semantic_similarity')[['question', 'semantic_similarity']])
```

---

## Phase 5: Meta-Learning Pivot

**Status**: 0% complete | **Estimated Time**: 8-12 hours

### What This Phase Accomplishes
Build a meta-learning system that learns to select the optimal retrieval pipeline for any question, generalizing across Finance, Healthcare, and Legal domains.

### Why This Is Novel (Paper Contribution)
- Most RAG papers optimize ONE pipeline
- We optimize the CHOICE of pipeline
- Cross-domain generalization is underexplored
- Clean experimental setup for reproducibility

---

### Required Reading: Meta-Learning

| Resource | What You'll Learn | Priority |
|----------|-------------------|----------|
| [MAML Paper](https://arxiv.org/abs/1703.03400) | Model-Agnostic Meta-Learning fundamentals | ⭐ **Core concept** |
| [Prototypical Networks](https://arxiv.org/abs/1703.05175) | Metric-based meta-learning (simpler than MAML) | ⭐ **Recommended approach** |
| [Meta-Learning Survey](https://arxiv.org/abs/2004.05439) | Overview of all meta-learning approaches | Reference |
| [Learn2Learn Library](https://github.com/learnables/learn2learn) | PyTorch meta-learning implementations | Practical |
| Our notes: `src/meta_learning/README.md` | Project-specific meta-learning design | When ready |

**Key concepts to understand:**
- **Episode**: One training iteration (support set + query set)
- **Support set**: Few examples with labels (e.g., 5 questions with best pipeline)
- **Query set**: Questions to predict pipeline for
- **N-way K-shot**: N classes (pipelines), K examples per class

---

### Required Reading: New Domains

#### Healthcare: PubMedQA
| Resource | What It Contains | Priority |
|----------|------------------|----------|
| [PubMedQA Paper](https://arxiv.org/abs/1909.06146) | Dataset description, baselines, methodology | ⭐ **Read first** |
| [HuggingFace Dataset](https://huggingface.co/datasets/pubmed_qa) | Direct data access | ⭐ **Use this** |
| [PubMed](https://pubmed.ncbi.nlm.nih.gov/) | Source medical literature | Reference |

**PubMedQA characteristics:**
- ~1,000 questions about medical research
- Yes/No/Maybe answers with reasoning
- Dense prose (no tables) - different from Finance!
- Tests comprehension of scientific abstracts

#### Legal: CUAD (Contract Understanding)
| Resource | What It Contains | Priority |
|----------|------------------|----------|
| [CUAD Paper](https://arxiv.org/abs/2103.06268) | Contract Understanding Atticus Dataset | ⭐ **Read first** |
| [HuggingFace Dataset](https://huggingface.co/datasets/cuad) | Direct data access | ⭐ **Use this** |
| [GitHub Repo](https://github.com/TheAtticusProject/cuad) | Code, examples, evaluation | Reference |

**CUAD characteristics:**
- 510 contracts, 13,000+ annotations
- 41 clause types (termination, liability, IP rights, etc.)
- Extractive QA - find specific clauses
- Tests precise legal text extraction

---

### All Tasks

| Task | Estimated Time | Status | What To Do |
|------|----------------|--------|------------|
| **Setup Phase** |
| Set up PubMedQA dataset | ~1 hr | ⬜ | Create `dataset_adapters/pubmedqa.py`, download data |
| Set up CUAD dataset | ~1 hr | ⬜ | Create `dataset_adapters/cuad.py`, download data |
| Ingest PubMedQA docs | ~2 hrs | ⬜ | Create embeddings for medical abstracts |
| Ingest CUAD contracts | ~2 hrs | ⬜ | Create embeddings for legal documents |
| **Oracle Labels** |
| Run grid search on Finance | ~2 hrs | ⬜ | Run all 4 pipelines on all 150 questions |
| Run grid search on PubMedQA | ~2 hrs | ⬜ | Run all 4 pipelines on PubMedQA questions |
| Run grid search on CUAD | ~2 hrs | ⬜ | Run all 4 pipelines on CUAD questions |
| Create oracle label dataset | ~30 min | ⬜ | For each question, record best pipeline |
| **Meta-Router** |
| Design router architecture | ~1 hr | ⬜ | MLP classifier: question embedding → pipeline |
| Implement episodic training | ~2 hrs | ⬜ | Support set → router state → predict query set |
| Train on Finance + Healthcare | ~1 hr | ⬜ | Hold out Legal for testing |
| **Evaluation** |
| Evaluate on held-out Legal | ~1 hr | ⬜ | Test cross-domain generalization |
| Compare vs fixed pipeline | ~30 min | ⬜ | Meta-router vs always-hybrid_filter_rerank |
| Document results | ~1 hr | ⬜ | Write up findings for paper |

### Key Files to Create
```
src/meta_learning/
├── router.py              # Meta-router model
├── oracle_labels.py       # Grid search for best pipeline per question
├── episodes.py            # Episodic data sampling
├── meta_trainer.py        # Training loop
└── evaluator.py           # Cross-domain evaluation

dataset_adapters/
├── pubmedqa.py            # PubMedQA loader
└── cuad.py                # CUAD loader

chroma_pubmedqa/           # Vector store for medical domain
chroma_cuad/               # Vector store for legal domain
```

### Meta-Learning Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     META-LEARNING FOR RAG                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  TRAINING: Learn which pipeline works best for which question type         │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ EPISODE τ (one domain, e.g., Finance)                                │   │
│  │                                                                      │   │
│  │  Support Set Sτ:                                                     │   │
│  │  ┌─────────────────────────────────────────────────────────────┐    │   │
│  │  │ Q1: "What is 3M's FY2018 CapEx?" → best: hybrid_filter      │    │   │
│  │  │ Q2: "Is 3M capital-intensive?"   → best: semantic           │    │   │
│  │  │ Q3: "What drove margin change?"  → best: hybrid_filter_rerank│   │   │
│  │  └─────────────────────────────────────────────────────────────┘    │   │
│  │                           │                                          │   │
│  │                           ▼                                          │   │
│  │  ┌─────────────────────────────────────────────────────────────┐    │   │
│  │  │              META-ROUTER fφ                                  │    │   │
│  │  │                                                              │    │   │
│  │  │  Input: question embedding + support set                     │    │   │
│  │  │  Output: pipeline_id to use                                  │    │   │
│  │  │                                                              │    │   │
│  │  │  Architecture: Transformer or MLP classifier                 │    │   │
│  │  └─────────────────────────────────────────────────────────────┘    │   │
│  │                           │                                          │   │
│  │                           ▼                                          │   │
│  │  Query Set:                                                          │   │
│  │  ┌─────────────────────────────────────────────────────────────┐    │   │
│  │  │ Q4: "What is Adobe's FY2019 revenue?"                       │    │   │
│  │  │     → Router predicts: hybrid_filter                        │    │   │
│  │  │     → Execute pipeline → Evaluate → Update router           │    │   │
│  │  └─────────────────────────────────────────────────────────────┘    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  CROSS-DOMAIN GENERALIZATION:                                               │
│                                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                  │
│  │   FINANCE    │    │  HEALTHCARE  │    │    LEGAL     │                  │
│  │ FinanceBench │    │   PubMedQA   │    │     CUAD     │                  │
│  │              │    │              │    │              │                  │
│  │ 10-K, 10-Q   │    │ Medical lit  │    │  Contracts   │                  │
│  │ Tables, nums │    │ Dense prose  │    │  Clauses     │                  │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘                  │
│         │                   │                   │                          │
│         └───────────────────┼───────────────────┘                          │
│                             ▼                                              │
│                   ┌──────────────────┐                                     │
│                   │   SHARED ROUTER  │                                     │
│                   │                  │                                     │
│                   │ Learns patterns: │                                     │
│                   │ • numeric → filter│                                    │
│                   │ • dense → semantic│                                    │
│                   │ • multi-hop → rerank                                   │
│                   └──────────────────┘                                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Domains
1. **Finance** - FinanceBench (current)
2. **Healthcare** - PubMedQA
3. **Legal** - CUAD (Contract Understanding)

### Tasks:
- [ ] Set up PubMedQA dataset and evaluation
- [ ] Set up CUAD dataset and evaluation
- [ ] Implement oracle labels (grid-search per question type)
- [ ] Train meta-router
- [ ] Evaluate cross-domain few-shot adaptation

### Paper Contribution:
- Meta-learned tool selection beats any fixed pipeline
- Cross-domain generalization with few-shot adaptation
- Clean recipe: grid-search → oracle → train router → episodic eval

### Technical Notes: Meta-Learning

**What is meta-learning?**
- "Learning to learn" - instead of learning one task, learn how to quickly adapt to new tasks
- Classic example: recognize new animal species from 5 examples (few-shot learning)
- Our application: learn which RAG pipeline works best for which question type

**Why meta-learning for RAG?**
- Different domains have different optimal pipelines
- Finance: tables, numbers → needs filtering
- Medical: dense prose → semantic might be better
- Legal: specific clauses → keyword matching important
- Instead of manually tuning per domain, learn the pattern

**What are "oracle labels"?**
- Ground truth for "which pipeline is best for this question"
- Created by grid search: run all pipelines on each question, pick winner
- Example:
  ```
  Q: "What is 3M's CapEx?"
  - semantic: 0.3 score
  - hybrid: 0.4 score
  - hybrid_filter: 0.7 score  ← winner
  - hybrid_filter_rerank: 0.65 score
  Oracle label: "hybrid_filter"
  ```

**Episodic training:**
- Sample a "task" (e.g., a domain or question type)
- Show router a few examples with oracle labels (support set)
- Ask router to predict pipeline for new question (query set)
- Update router based on how well it predicts

**Cross-domain generalization:**
- Train on Finance + Medical → test on Legal (unseen domain)
- If router learns abstract patterns ("numeric questions need filtering")
- It should generalize without seeing Legal training data
- This is the key contribution for a paper!

**Why this is novel:**
- Most RAG papers optimize one pipeline
- We're saying: optimize the CHOICE of pipeline
- Meta-learning is underexplored in RAG literature
- Clean experimental setup: 3 domains, 4 pipelines, episodic evaluation

---

## Progress Tracker

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PROJECT PROGRESS                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  FOUNDATION (Pre-Phase Work)                           [██████████] 100%   │
│  ├─ ✅ Project structure & guidelines                                      │
│  ├─ ✅ Central configuration (config.py)                                   │
│  ├─ ✅ LLM provider abstraction (5 providers)                              │
│  ├─ ✅ Retrieval pipelines (4 strategies)                                  │
│  ├─ ✅ Evaluation framework                                                │
│  └─ ✅ 367 PDFs downloaded                                                 │
│  Time spent: ~10 hours                                                      │
│                                                                             │
│  PHASE 1: Improved Ingestion                           [████████░░] 90%    │
│  ├─ ✅ Metadata schema design                          (~30 min)           │
│  ├─ ✅ Filename parser                                 (~20 min)           │
│  ├─ ✅ V2 ingestion script                             (~1 hr)             │
│  ├─ ✅ Table→markdown conversion                       (~20 min)           │
│  ├─ ✅ Element type tagging                            (~10 min)           │
│  ├─ ✅ OCR dependencies (poppler, tesseract)           (~10 min)           │
│  ├─ ✅ Sample test (486 chunks verified)               (~15 min)           │
│  └─ ⬜ Full ingestion (367 PDFs)                       (~2-4 hrs waiting)  │
│  Time spent: ~2.5 hrs | Remaining: ~2-4 hrs                                │
│                                                                             │
│  PHASE 2: Per-File Retrieval                           [░░░░░░░░░░] 0%     │
│  ├─ ⬜ Enhance metadata extraction                     (~1 hr)             │
│  ├─ ⬜ Update filter logic                             (~30 min)           │
│  ├─ ⬜ ChromaDB native filtering                       (~1 hr)             │
│  ├─ ⬜ Fallback strategy                               (~30 min)           │
│  ├─ ⬜ Filter logging                                  (~20 min)           │
│  └─ ⬜ Benchmark comparison                            (~1 hr)             │
│  Estimated: ~4 hrs                                                          │
│                                                                             │
│  PHASE 3: Question-Type Routing                        [░░░░░░░░░░] 0%     │
│  ├─ ⬜ Question classifier                             (~1.5 hrs)          │
│  ├─ ⬜ Metrics-specific prompt                         (~45 min)           │
│  ├─ ⬜ Domain-specific prompt                          (~30 min)           │
│  ├─ ⬜ Novel-specific prompt                           (~30 min)           │
│  ├─ ⬜ Table-priority retrieval                        (~1 hr)             │
│  ├─ ⬜ Calculation verification                        (~45 min)           │
│  └─ ⬜ Wire up routing                                 (~30 min)           │
│  Estimated: ~5 hrs                                                          │
│                                                                             │
│  PHASE 4: Evaluation & Validation                      [░░░░░░░░░░] 0%     │
│  ├─ ⬜ Full 150-question eval                          (~30 min run)       │
│  ├─ ⬜ Add accuracy metric                             (~30 min)           │
│  ├─ ⬜ Analyze by question type                        (~30 min)           │
│  ├─ ⬜ Analyze by company/year                         (~20 min)           │
│  ├─ ⬜ Compare vs baseline                             (~20 min)           │
│  ├─ ⬜ Document failure modes                          (~30 min)           │
│  └─ ⬜ Write evaluation report                         (~30 min)           │
│  Estimated: ~3 hrs                                                          │
│                                                                             │
│  PHASE 5: Meta-Learning Pivot                          [░░░░░░░░░░] 0%     │
│  ├─ ⬜ PubMedQA dataset setup                          (~1 hr)             │
│  ├─ ⬜ CUAD dataset setup                              (~1 hr)             │
│  ├─ ⬜ Ingest PubMedQA                                 (~2 hrs)            │
│  ├─ ⬜ Ingest CUAD                                     (~2 hrs)            │
│  ├─ ⬜ Grid search (3 domains × 4 pipelines)           (~6 hrs)            │
│  ├─ ⬜ Create oracle labels                            (~30 min)           │
│  ├─ ⬜ Design router architecture                      (~1 hr)             │
│  ├─ ⬜ Implement episodic training                     (~2 hrs)            │
│  ├─ ⬜ Train on Finance + Healthcare                   (~1 hr)             │
│  ├─ ⬜ Evaluate on Legal (cross-domain)                (~1 hr)             │
│  ├─ ⬜ Compare vs fixed pipeline                       (~30 min)           │
│  └─ ⬜ Document results                                (~1 hr)             │
│  Estimated: ~12 hrs                                                         │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│  TIME SUMMARY                                                               │
│  ├─ Foundation (complete):     ~10 hrs                                     │
│  ├─ Phase 1 (90% complete):    ~2.5 hrs done, ~3 hrs remaining             │
│  ├─ Phase 2 (0%):              ~4 hrs estimated                            │
│  ├─ Phase 3 (0%):              ~5 hrs estimated                            │
│  ├─ Phase 4 (0%):              ~3 hrs estimated                            │
│  └─ Phase 5 (0%):              ~12 hrs estimated                           │
│                                                                             │
│  TOTAL: ~12.5 hrs done | ~27 hrs remaining                                 │
│  OVERALL PROGRESS: [████░░░░░░░░░░░░░░░░] ~32%                             │
│                                                                             │
│  ═══════════════════════════════════════════════════════════════════════   │
│  NEXT ACTION: Run full ingestion                                           │
│  COMMAND: python src/create_database_v2.py                                 │
│  ═══════════════════════════════════════════════════════════════════════   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Quick Reference

### Commands
```bash
# Test improved ingestion
python src/create_database_v2.py --sample 5

# Full ingestion
python src/create_database_v2.py

# Run evaluation
python src/bulk_testing.py --model claude-sonnet-4-5 --top-k 10

# Run on subset
python src/bulk_testing.py --subset data/question_sets/financebench_subset_questions.csv
```

### Cost Estimates (150 questions)
| Model | Cost/Run |
|-------|----------|
| Claude 4.5 Sonnet | ~$2.00 |
| GPT 5.2 | ~$3.20 |
| Gemini 3 Flash | ~$0.07 |
| DeepSeek Chat | ~$0.05 |

### Key Files
| File | Purpose |
|------|---------|
| `src/config.py` | Centralized configuration |
| `src/create_database_v2.py` | Improved ingestion |
| `src/bulk_testing.py` | Evaluation runner |
| `src/retrieval_tools/` | Retrieval pipelines |
| `src/meta_learning/` | Meta-learning (stubs) |

---

## Timeline (Effort-Based, No Dates)

| Phase | Effort | Dependencies |
|-------|--------|--------------|
| Phase 1 | 2-3 sessions | None |
| Phase 2 | 3-4 sessions | Phase 1 |
| Phase 3 | 2-3 sessions | Phase 2 |
| Phase 4 | 1-2 sessions | Phase 3 |
| Phase 5 | 5-7 sessions | Phase 4 |

---

*Last updated: December 12, 2024*

---

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
| **10-Q** | Quarterly SEC filing (less detailed than 10-K) |
| **Semantic Similarity** | Cosine similarity between embeddings (0-1) |
| **Meta-learning** | Learning to learn - adapting quickly to new tasks |
| **Episode** | One training iteration in meta-learning (support + query sets) |
| **Oracle Label** | Ground truth best pipeline for a question (from grid search) |

---

## Sources
- [FinanceBench Paper](https://arxiv.org/abs/2311.11944)
- [Ragie FinanceBench Results](https://www.ragie.ai/blog/ragie-outperformed-financebench)
- [Databricks Long Context RAG](https://www.databricks.com/blog/long-context-rag-performance-llms)
- [Patronus AI FinanceBench Docs](https://docs.patronus.ai/docs/research_and_differentiators/financebench)
