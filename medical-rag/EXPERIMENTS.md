# PubMedQA Experiments Log

## Dataset
- **PubMedQA Labeled**: 1,000 biomedical yes/no/maybe questions
- **Task**: Reading comprehension (context provided)
- **SOTA**: ~77.9% (PMC-LLaMA fine-tuned), ~75% (GPT-4 zero-shot)

---

## Experiment 1: Baseline Model Comparison
**Date**: 2026-01-11
**Goal**: Establish baseline accuracy across models

| Model | Accuracy | Status | Notes |
|-------|----------|--------|-------|
| **gpt-5.2** | **69.1%** | Done | Best performer |
| Qwen2.5-72B | 66.0% | Done | Best value (Together serverless) |
| Kimi-K2 | 65.0% | Done | Great cost-efficiency |
| gpt-4o-mini | 55.5% | Done | Baseline |
| claude-sonnet-4 | - | Running | In progress (~10%) |

### Issues Found
- `claude-sonnet-4-5-20250514` → Model not found, using `claude-sonnet-4-20250514`
- `Qwen/Qwen3-235B-A22B-Instruct-2507` → Not serverless, used Qwen2.5-72B instead

### Commands
```bash
python src/evaluation.py --model gpt-4o-mini
python src/evaluation.py --model gpt-5.2
python src/evaluation.py --model moonshotai/Kimi-K2-Instruct
python src/evaluation.py --model claude-sonnet-4-20250514
python src/evaluation.py --model Qwen/Qwen2.5-72B-Instruct-Turbo
```

---

## Results Archive

### gpt-5.2 (2026-01-11)
```
Accuracy: 69.1% (691/1000)
```

### Qwen2.5-72B (2026-01-11)
```
Accuracy: 66.0% (660/1000)
```

### Kimi-K2-Instruct (2026-01-11)
```
Accuracy: 65.0% (650/1000)
```

### gpt-4o-mini (2026-01-11)
```
Accuracy: 55.5% (555/1000)
```

---

## Analysis

### Key Findings
1. **GPT-5.2 leads** at 69.1% - approaching GPT-4 zero-shot SOTA (~75%)
2. **Qwen2.5-72B** at 66.0% - excellent serverless alternative via Together
3. **Kimi-K2** at 65.0% - best value model via Together API
4. **gpt-4o-mini underperforms** at 55.5% - significant gap to larger models

### Model Comparison
| Model | Accuracy | Gap to SOTA | Provider | Cost |
|-------|----------|-------------|----------|------|
| gpt-5.2 | 69.1% | -5.9% | OpenAI | $$$ |
| Qwen2.5-72B | 66.0% | -9.0% | Together | $ |
| Kimi-K2 | 65.0% | -10.0% | Together | $ |
| gpt-4o-mini | 55.5% | -19.5% | OpenAI | $ |

## Notes
- PubMedQA has 3 answer types: yes, no, maybe
- "Maybe" is hardest - models tend to be overconfident
- Context is provided, so this tests reading comprehension, not retrieval
- Kimi-K2 outperforming gpt-4o-mini significantly!

## Next Steps
- [x] Run baseline comparison
- [x] Run Qwen2.5-72B as serverless alternative
- [ ] Complete Claude Sonnet 4 evaluation (in progress)
- [ ] Analyze error patterns by answer type
- [ ] Test with chain-of-thought prompting
