# LLM-as-a-Judge Evaluation Results

## Folder Structure

```
LLM_as_a_Judge/
├── junjie_runs/                    # Junjie's evaluation runs
│   ├── element_based_chunks_2000chars_forced_answer/
│   │   └── 2025-11-21_*.csv        # Best: 70.4%
│   └── simple-pipeline-forced-answer/
│       └── 2025-11-22_*.csv        # Score: 27.1%
│
├── aum_runs/                       # Aum's evaluation runs
│   ├── element_based_chunks_forced_answer.csv      # 35.9%
│   ├── element_based_chunking_hybrid_metadata.csv  # 59.8%
│   ├── 2000charchunks_judged.csv                   # 50.4%
│   ├── character_based_chunking.csv                # 46.9%
│   ├── simple-pipeline-forced-answer.csv           # 34.4%
│   ├── 2000charchunks_OpenAIEmbedding.csv          # 28.5%
│   └── 1000charchunks.csv                          # 21.3%
│
├── visualizations/                 # Generated charts
│   ├── overall_scores.png
│   └── score_distribution.png
│
├── analyze_results.py              # Comprehensive analysis script
├── detailed_analysis.csv           # Exported metrics
└── run_judge_evaluation.py         # Evaluation script
```

## Performance Rankings

| Rank | Configuration | Score | Median | Perfect | Zero | Source |
|------|--------------|-------|--------|---------|------|--------|
| 1 | Element-Based Chunking (Forced Answer) | **70.4%** | 90% | 7 | 2 | Junjie |
| 2 | Element-Based + Hybrid Metadata | 59.8% | 65% | 9 | 5 | Aum |
| 3 | 2000-Char Chunking (OpenAI Prompt) | 50.4% | 30% | 5 | 5 | Aum |
| 4 | Character-Based Chunking | 46.9% | 30% | 4 | 6 | Aum |
| 5 | Element-Based (Aum's version) | 35.9% | 30% | 2 | 6 | Aum |
| 6 | Simple Pipeline (Forced Answer) | 34.4% | 30% | 2 | 7 | Aum |
| 7 | 2000-Char + OpenAI Embeddings | 28.5% | 25% | 3 | 10 | Aum |
| 8 | Simple Pipeline (Junjie) | 27.1% | 20% | 2 | 7 | Junjie |
| 9 | 1000-Char Chunking (Baseline) | 21.3% | 10% | 1 | 12 | Aum |

## Breakdown Analysis

### By Question Type (Junjie's best config - 70.4%)
- **novel-generated**: Highest scores
- **domain-relevant**: Medium scores
- **metrics-generated**: Lowest scores (numeric calculations)

### Key Insights
- Junjie's element-based run (70.4%) significantly outperforms Aum's version (35.9%)
- Difference is due to better retrieval quality (100% correct docs vs 83%)
- Metrics-generated questions (calculations) are hardest across all configs
- Block and Costco companies have lowest scores

## Running Analysis

```bash
python3 analyze_results.py
```

This generates:
- Overall rankings with statistics
- Breakdowns by question type, company, doc type
- Error analysis of worst performing questions
- Visualizations in `visualizations/` folder
- Detailed CSV export

## Recommended Additional Experiments

1. **Chunk Sizes**: 500, 1000, 1500, 2000, 3000 chars
2. **Embeddings**: text-embedding-3-large, BGE, E5
3. **Retrieval**: k=5/10/15/20, hybrid search, reranking
4. **Prompting**: Chain-of-thought, few-shot examples
5. **Filtering**: Metadata filtering by company/date
