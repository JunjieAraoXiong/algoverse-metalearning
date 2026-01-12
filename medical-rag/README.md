# Medical RAG - PubMedQA Evaluation

Biomedical question answering evaluation using the PubMedQA dataset.

## Quick Start

```bash
# Quick test (10 questions)
python src/evaluation.py --model gpt-4o-mini --limit 10

# Full evaluation
python src/evaluation.py --model gpt-4o-mini

# With different models
python src/evaluation.py --model claude-sonnet-4-5-20250514
python src/evaluation.py --model gpt-5.2
```

## Dataset

PubMedQA contains ~1,000 biomedical yes/no/maybe questions derived from PubMed research abstracts.

**Key characteristic**: Context (abstract) is provided with each question, so no retrieval is needed.

## Expected Results

| Model | Expected Accuracy |
|-------|-------------------|
| GPT-4o | ~75% |
| Claude 4.5 | ~75% |
| GPT-4o-mini | ~70% |
| Qwen3 235B | ~70% |

## Output

Results are saved to `results/` as JSON:

```json
{
  "model": "gpt-4o-mini",
  "accuracy": 0.742,
  "correct": 742,
  "total": 1000,
  "by_answer_type": {
    "yes": {"accuracy": 0.78, "correct": 312, "total": 400},
    "no": {"accuracy": 0.71, "correct": 284, "total": 400},
    "maybe": {"accuracy": 0.73, "correct": 146, "total": 200}
  }
}
```

## Project Structure

```
medical-rag/
├── src/
│   └── evaluation.py    # Main evaluation script
├── data/
│   └── pubmedqa/
│       └── pubmedqa_labeled.jsonl
├── results/             # Evaluation outputs
└── README.md
```

## Dependencies

Uses providers from the main `rag/` project. Make sure `rag/.env` has your API keys.
