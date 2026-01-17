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

## Provider Architecture

The system uses a provider abstraction layer for multi-LLM support:

```
+------------------+
|   User Code      |
|  (bulk_testing)  |
+--------+---------+
         |
         v
+------------------+
|  get_provider()  |  <-- Entry point
+--------+---------+
         |
         v
+------------------+
|    config.py     |  <-- Routes model name to provider type
+--------+---------+
         |
         v
+------------------+
|   factory.py     |  <-- Creates provider instances
+--------+---------+
         |
         v
+------------------+
|  LLMProvider ABC |  <-- Abstract interface: generate(), embed()
+--------+---------+
         |
    +----+----+
    |    |    |
    v    v    v
+------+------+------+
|OpenAI|Anthro|Togeth|  <-- Concrete implementations
+------+------+------+
```

### What Works Well

1. **Factory pattern** - Single `get_provider()` call handles all provider logic
2. **Lazy loading** - Providers instantiated only when needed
3. **Instance caching** - Same provider reused across calls
4. **Standardized response** - All providers return `LLMResponse` dataclass
5. **Config-driven routing** - Model names mapped to providers in `config.py`

### Research-Grade Standards

| Feature | Status | Notes |
|---------|--------|-------|
| Provider abstraction | Complete | ABC with OpenAI/Anthropic/Together |
| Environment-based API keys | Complete | Via `python-dotenv` |
| Usage tracking | Partial | Token counts returned but not aggregated |
| Cost tracking | Missing | No per-request cost calculation |
| Retry logic | Missing | No exponential backoff on failures |
| Rate limiting | Missing | No request throttling |
| Request logging | Missing | No structured logging for debugging |
| Model versioning | Complete | Full model IDs in config |
| Reproducibility | Partial | Seeds set but not logged with results |

### Quick Win: Cost Tracking

Add to `LLMResponse` dataclass:

```python
@dataclass
class LLMResponse:
    content: str
    model: str
    usage: dict
    cost_usd: float = 0.0  # Add this field

# In provider implementations:
COST_PER_1K = {
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "gpt-4o": {"input": 0.005, "output": 0.015},
    "claude-sonnet-4-5-20250514": {"input": 0.003, "output": 0.015},
}

def calculate_cost(model: str, usage: dict) -> float:
    rates = COST_PER_1K.get(model, {"input": 0, "output": 0})
    input_cost = (usage.get("prompt_tokens", 0) / 1000) * rates["input"]
    output_cost = (usage.get("completion_tokens", 0) / 1000) * rates["output"]
    return input_cost + output_cost
```

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
