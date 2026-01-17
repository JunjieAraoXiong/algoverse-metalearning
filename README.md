# Self-Correcting RAG for Financial Document QA

A multi-agent RAG framework with judge-driven self-correction for high-stakes document QA.

**Paper**: Self-Correcting RAG: Judge-Driven Retrieval for Financial Document QA (FINAI@ICLR 2026)

## Key Features

- **Three Specialized Agents**: Retrieval, Reasoning, and Judge agents with escalation strategies
- **Self-Correction Loop**: Judge-driven retry when answers are below quality threshold
- **Rule-Based Routing**: Zero-cost pipeline selection matching LLM-based routers
- **Cross-Domain**: Works on Finance (FinanceBench), Medical (PubMedQA), and Legal (CUAD)

## Project Structure

```
rag/
├── src/
│   ├── agents/              # Multi-agent system (Algorithm 1)
│   │   ├── orchestrator.py  # Main retry loop
│   │   ├── retrieval_agent.py
│   │   ├── reasoning_agent.py
│   │   └── judge_agent.py
│   ├── retrieval_tools/     # Retrieval pipelines
│   │   ├── semantic.py      # Dense vector search
│   │   ├── hybrid.py        # BM25 + semantic ensemble
│   │   ├── rerank.py        # Cross-encoder reranking
│   │   ├── hyde.py          # Hypothetical Document Embeddings
│   │   └── router.py        # Question-type routing
│   ├── providers/           # LLM adapters (OpenAI, Anthropic, etc.)
│   ├── meta_learning/       # Router training
│   ├── config.py            # Central configuration
│   └── bulk_testing.py      # Evaluation entry point
├── evaluation/              # Metrics & LLM-as-Judge
├── dataset_adapters/        # FinanceBench, PubMedQA, CUAD loaders
└── scripts/                 # Experiment & training scripts
```

## Quick Start

### Installation
```bash
git clone https://github.com/JunjieAraoXiong/algoverse-metalearning.git
cd algoverse-metalearning/rag
pip install -r requirements.txt
cp .env.example .env  # Add your API keys (OPENAI_API_KEY, ANTHROPIC_API_KEY)
```

### Run Single-Pass Baseline
```bash
python src/bulk_testing.py \
    --dataset financebench \
    --pipeline hybrid_filter_rerank \
    --model gpt-4o-mini \
    --use-llm-judge
```

### Run Self-Correcting RAG (Agentic Mode)
```bash
python src/bulk_testing.py \
    --dataset financebench \
    --pipeline routed \
    --model gpt-4o-mini \
    --use-llm-judge \
    --use-agentic-retry \
    --max-retries 1
```

## Retrieval Pipelines

| Pipeline | Description | When to Use |
|----------|-------------|-------------|
| `semantic` | Dense vector search | Simple factual queries |
| `hybrid` | BM25 + semantic ensemble | Keyword-heavy queries |
| `hybrid_filter` | Hybrid + metadata filtering | Entity-specific queries |
| `hybrid_filter_rerank` | Full pipeline with reranking | Complex reasoning |
| `routed` | Dynamic selection + retry | **Production (recommended)** |

## Algorithm Overview

The orchestrator implements a judge-driven retry loop:

```
while attempt <= max_retries:
    docs = RetrievalAgent.retrieve(question, attempt)
    answer = ReasoningAgent.generate(question, docs)
    score = JudgeAgent.evaluate(question, answer)

    if score >= threshold:
        return answer

    escalate_strategies()
    attempt += 1
```

**Escalation strategies:**
- **Retrieval**: Increase k (10 -> 20 -> 25), enable HyDE
- **Reasoning**: Standard -> Conservative -> Detailed prompts
- **Judge**: Lower threshold (0.5 -> 0.4 -> 0.3)

## Reproducing Results

### Step 1: Prepare ChromaDB
```bash
# Download pre-built embeddings (recommended)
# [Link to be added after publication]

# Or build from scratch (requires SEC PDFs)
python src/ingest_docling.py --input-dir data/pdfs --output-dir chroma_docling
```

### Step 2: Run Experiments
```bash
# Table 1: Single-pass vs Self-Correcting RAG
python src/bulk_testing.py --dataset financebench --pipeline hybrid_filter_rerank --model gpt-4o-mini --use-llm-judge
python src/bulk_testing.py --dataset financebench --pipeline routed --model gpt-4o-mini --use-llm-judge --use-agentic-retry

# Table 4: Cross-domain validation
python src/bulk_testing.py --dataset pubmedqa --pipeline routed --model gpt-4o-mini --use-agentic-retry --domain medical
python src/bulk_testing.py --dataset cuad --pipeline routed --model gpt-4o-mini --use-agentic-retry --domain legal
```

## Supported Models

| Provider | Models | Notes |
|----------|--------|-------|
| OpenAI | gpt-4o-mini, gpt-4o | Recommended for evaluation |
| Anthropic | claude-sonnet-4-5-20250514 | High quality |
| Together | Llama 3.1 70B | For cluster deployment |

## Citation

```bibtex
@inproceedings{anonymous2026selfcorrecting,
  title={Self-Correcting RAG: Judge-Driven Retrieval for Financial Document QA},
  author={Anonymous},
  booktitle={FINAI Workshop at ICLR 2026},
  year={2026}
}
```

## License

MIT License
