# Financial RAG Project Notes

## Project Overview

Building a Retrieval-Augmented Generation (RAG) system for financial document QA, evaluated on FinanceBench. Planning to pivot toward **meta-learning** for cross-domain tool selection.

---

## Current Performance

| Metric | Score |
|--------|-------|
| **Overall Semantic Similarity** | 0.495 |
| domain-relevant | 0.60 |
| novel-generated | 0.53 |
| **metrics-generated** | **0.35** ← Weak point |

**Score Distribution:**
- 30% of questions score < 0.3 (bad)
- 23% score > 0.7 (good)

---

## Question Types in FinanceBench

### 1. Metrics-Generated (50 questions) - HARDEST
**What:** Direct numerical extraction or calculation from financial statements.
**Examples:**
- "What is the FY2018 capital expenditure amount for 3M?"
- "What is the FY2019 fixed asset turnover ratio for Activision Blizzard?"
- "What is the year end FY2018 net PPNE for 3M?"

**Why Hard:**
- Requires finding exact table/row
- Often needs calculation (ratios, averages)
- Units matter (millions vs billions)

**Our Score:** 0.35 (worst)

### 2. Domain-Relevant (50 questions) - MEDIUM
**What:** Analytical questions requiring domain knowledge + data.
**Examples:**
- "Is 3M a capital-intensive business based on FY2022 data?"
- "What drove operating margin change as of FY2022 for 3M?"
- "Does 3M have a reasonably healthy liquidity profile?"

**Why Medium:**
- Needs interpretation, not just extraction
- Requires understanding financial concepts
- Longer, synthesized answers

**Our Score:** 0.60 (decent)

### 3. Novel-Generated (50 questions) - MEDIUM
**What:** Creative/inferential questions not directly in documents.
**Examples:**
- "If we exclude M&A impact, which segment dragged down 3M's growth?"
- "Does 3M maintain a stable trend of dividend distribution?"
- "Does Adobe have improving FCF conversion?"

**Why Medium:**
- Requires reasoning across multiple data points
- May need trend analysis
- Prose-style answers

**Our Score:** 0.53 (OK)

---

## Architecture

### Current Pipeline

```
INGESTION:
PDF → Unstructured (hi_res) → chunk_by_title (2000 chars) → ChromaDB + OpenAI Embeddings

RETRIEVAL:
Question → Pipeline Selection → Retrieve (k × factor) → Metadata Filter → Rerank → Top-K

GENERATION:
Context + Question → LLM → Answer

EVALUATION:
Predicted vs Gold → Semantic Similarity (0-1)
```

### Retrieval Pipelines Available
1. `semantic` - Pure embedding similarity
2. `hybrid` - BM25 + Semantic (50/50)
3. `hybrid_filter` - + Metadata filtering (company/year)
4. `hybrid_filter_rerank` - + Cross-encoder reranking (default)

### Key Components
| Component | Location | Notes |
|-----------|----------|-------|
| Config | `src/config.py` | All defaults centralized |
| Providers | `src/providers/` | Claude, GPT, Gemini, DeepSeek |
| Pipelines | `src/retrieval_tools/` | 4 retrieval strategies |
| Reranker | `src/retrieval_tools/rerank.py` | bge-reranker-large |
| Meta-learning | `src/meta_learning/` | Stubs only |

---

## Known Issues

### Ingestion Problems
1. **No filename metadata** - Can't reliably filter by company/year
2. **Tables → plain text** - Loses structure, hurts numeric QA
3. **No element tagging** - Can't distinguish tables from prose

### Retrieval Problems
1. **Single query** - Misses synonyms/rephrasings
2. **Flat structure** - No doc→section→chunk hierarchy
3. **BM25 built on-the-fly** - Slow for large corpus

### Generation Problems
1. **Same prompt for all** - Numeric questions need different handling
2. **No verification** - Hallucinated numbers not caught
3. **No chain-of-thought** - Complex calculations fail

---

## Improvement Plan

### Priority 1: Fix Ingestion (High Impact)
- [ ] Parse filename → `{company, year, doc_type}` metadata
- [ ] Preserve tables as markdown (not plain text)
- [ ] Tag `element_type` in metadata (table vs prose)

### Priority 2: Smarter Retrieval (Medium Impact)
- [ ] Multi-query generation (rephrase questions)
- [ ] Question-type routing (numeric → prefer tables)
- [ ] HyDE (Hypothetical Document Embeddings)

### Priority 3: Better Generation (Medium Impact)
- [ ] Numeric-specific prompting (extract → calculate → format)
- [ ] Verification pass for calculations
- [ ] Citation/grounding check

---

## Meta-Learning Pivot

### Concept
Instead of optimizing one RAG pipeline, learn to **select the right tool/pipeline** for each question type across domains.

### Architecture
```
Episode τ:
  Support set Sτ (few examples) → Router fφ → builds task-state zτ
  Query question x → fφ(x, zτ) → choose pipeline → retrieve → generate → answer
```

### Domains for Paper
1. **Finance** - FinanceBench (current)
2. **Healthcare** - PubMedQA
3. **Legal** - CUAD (Contract Understanding)

### Paper Contribution
- Meta-learned tool selection beats any fixed pipeline
- Cross-domain few-shot adaptation
- Clean recipe: grid-search → oracle labels → train router → episodic eval

---

## Cost Estimates

### Embeddings: NOW FREE
Using local `BAAI/bge-large-en-v1.5` instead of OpenAI.
- Ingestion: **FREE**
- Query retrieval: **FREE**
- Reranking: **FREE** (local bge-reranker-large)

### Per-Run Costs (150 questions) - LLM Only

| Model | Cost/Run |
|-------|----------|
| Claude 4.5 Sonnet | ~$2.00 |
| GPT 5.2 | ~$3.20 |
| GPT 5.2-mini | ~$0.32 |
| Gemini 3 Flash | ~$0.07 |
| DeepSeek Chat | ~$0.05 |
| Llama 3.1 70B | ~$0.12 |

### Recommended Strategy
```
Phase 1: Iterate cheap (Gemini Flash, $0.07/run) - Budget: ~$3
Phase 2: Validate (Claude 4.5 on full set) - Budget: ~$10
Phase 3: Final benchmarking (all models) - Budget: ~$25
Total: ~$40
```

### Re-ingestion Cost
- **FREE** (local embeddings)

---

## Commands Reference

```bash
# Run with default (Claude 4.5 Sonnet)
python src/bulk_testing.py --top-k 10 --initial-k-factor 5.0

# Run on subset (faster iteration)
python src/bulk_testing.py --subset data/question_sets/financebench_subset_questions.csv

# Try different models
python src/bulk_testing.py --model gpt-5.2
python src/bulk_testing.py --model gemini-3-flash
python src/bulk_testing.py --model deepseek-chat

# Rebuild database
python src/ingest.py
```

---

## Files Structure

```
rag/
├── src/
│   ├── config.py                 # Centralized configuration
│   ├── bulk_testing.py           # Main evaluation runner
│   ├── ingest.py                 # Improved ingestion with metadata
│   ├── metadata_utils.py         # Company/year extraction
│   ├── providers/                # LLM provider abstractions
│   │   ├── base.py
│   │   ├── factory.py
│   │   ├── openai_provider.py
│   │   ├── anthropic_provider.py
│   │   └── google_provider.py
│   └── retrieval_tools/          # Retrieval pipelines
│       ├── tool_registry.py
│       ├── semantic.py
│       ├── hybrid.py
│       ├── metadata_filter.py
│       └── rerank.py
├── evaluation/                   # Metrics
├── dataset_adapters/             # Dataset loaders
├── bulk_runs/                    # Results
├── chroma/                       # Vector database
├── .claude/
│   └── CLAUDE.md                 # Project guidelines
└── NOTES.md                      # This file
```

---

## Next Steps

1. **Improve ingestion** - Fix metadata, preserve tables
2. **Test with Claude 4.5** - See baseline improvement from better LLM
3. **Implement multi-query** - Improve retrieval recall
4. **Add numeric-specific prompting** - Target metrics-generated weakness
5. **Build second domain** - Start meta-learning experiments

---

*Last updated: December 2024*
