# Publication-Quality RAG Experiment Suite

## Overview

This experiment suite contains **25 experiments** organized into **6 categories**, designed to produce publication-quality results that far exceed Aum's 7 experiments.

## Experiment Categories

### 1. Ablation Study (5 experiments)
**Purpose:** Quantify contribution of each retrieval enhancement

| Experiment | Description | Expected Insight |
|------------|-------------|------------------|
| `baseline_semantic_only` | Pure semantic search | Lower bound |
| `ablation_hybrid` | + BM25 hybrid search | Impact of lexical matching |
| `ablation_metadata` | + Metadata filtering | Impact of company/year filters |
| `ablation_reranking` | + Cross-encoder reranking | Impact of reranking |
| `ablation_all_features` | All enhancements | Upper bound |

### 2. Chunk Size Analysis (9 experiments)
**Purpose:** Find optimal document segmentation for financial docs

- **Standard chunking:** 500, 1000, 1500, 2000, 3000 chars
- **Element-based:** 1000, 1500, 2000, 3000 chars

### 3. K-Value Sensitivity (6 experiments)
**Purpose:** Understand retrieval depth trade-offs

- k = 3, 5, 10, 15, 20, 30

### 4. Embedding Model Comparison (2 experiments)
**Purpose:** Quality vs cost trade-off

- `text-embedding-3-small` (cheaper, faster)
- `text-embedding-3-large` (higher quality)

### 5. Prompting Strategies (1 experiment)
**Purpose:** Validate forced-answer approach

### 6. Optimal Configurations (2 experiments)
**Purpose:** Test hypothesized best combinations

---

## Quick Start (Priority Order)

### Step 1: Run Core Ablation Study First
```bash
cd /Users/hansonxiong/Desktop/algoverse/LLM_as_a_Judge/experiments/results

# These 5 experiments are most important for publication
bash scripts/baseline_semantic_only.sh
bash scripts/ablation_hybrid.sh
bash scripts/ablation_metadata.sh
bash scripts/ablation_reranking.sh
bash scripts/ablation_all_features.sh
```
**Time:** ~30 minutes

### Step 2: Run K-Value Analysis
```bash
bash scripts/k_value_3.sh
bash scripts/k_value_5.sh
bash scripts/k_value_10.sh
bash scripts/k_value_15.sh
bash scripts/k_value_20.sh
bash scripts/k_value_30.sh
```
**Time:** ~30 minutes

### Step 3: Run Chunk Size Analysis
```bash
bash scripts/chunk_element_1000.sh
bash scripts/chunk_element_1500.sh
bash scripts/chunk_element_2000.sh
bash scripts/chunk_element_3000.sh
bash scripts/chunk_standard_1000.sh
bash scripts/chunk_standard_2000.sh
```
**Time:** ~30 minutes

### Step 4: Run All Remaining
```bash
bash run_all.sh
```
**Time:** ~2-4 hours total

---

## Output Files

Results saved to `experiments/results/`:
- `*.csv` - Raw results per experiment
- `*.log` - Execution logs
- `analysis.ipynb` - Publication figures notebook
- `experiment_manifest.json` - Complete configuration record

---

## Publication Figures Generated

The `analysis.ipynb` notebook generates:

1. **Figure 1:** Overall Performance Comparison (bar chart)
2. **Figure 2:** Ablation Study Results
3. **Figure 3:** Chunk Size vs Performance
4. **Figure 4:** K-Value Sensitivity Curve
5. **Table 1:** Summary Statistics (LaTeX format)

---

## Comparison: Your Suite vs Aum's

| Aspect | Aum's Experiments | Your Suite |
|--------|-------------------|------------|
| Total experiments | 7 | **25** |
| Ablation study | ✗ | ✓ (5 experiments) |
| Chunk size analysis | 2 sizes | **5 sizes × 2 strategies** |
| K-value analysis | 1 value | **6 values** |
| Embedding comparison | 1 model | **2 models** |
| Statistical analysis | ✗ | ✓ (p-values, CI) |
| Publication figures | ✗ | ✓ (4 figures + LaTeX table) |

---

## Prerequisites

Before running:

1. **API Keys** in `/Users/hansonxiong/Desktop/algoverse/shawheen rag/.env`:
   ```
   TOGETHER_API_KEY=your_key
   OPENAI_API_KEY=your_key
   ```

2. **Database created:**
   ```bash
   cd "/Users/hansonxiong/Desktop/algoverse/shawheen rag"
   python src/create_database_element_based.py
   ```

3. **Dependencies installed:**
   ```bash
   pip install matplotlib seaborn scipy jupyter
   ```

---

## After Running

1. Open `analysis.ipynb` in Jupyter
2. Run all cells to generate figures
3. Figures saved as PDF (vector) and PNG
4. LaTeX table saved for direct paper inclusion

---

## Estimated Total Time

- **Quick ablation (5 experiments):** 30 min
- **Full suite (25 experiments):** 2-4 hours
- **Analysis notebook:** 5 min

---

## Support

If experiments fail:
- Check API keys in `.env`
- Check rate limits (add sleep between runs if needed)
- Check logs in `experiments/results/*.log`
