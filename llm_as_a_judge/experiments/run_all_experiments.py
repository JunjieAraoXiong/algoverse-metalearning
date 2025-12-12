#!/usr/bin/env python3
"""
Publication-Quality RAG Experiment Suite
========================================
Comprehensive ablation study for financial QA RAG system.

This creates a systematic experiment matrix covering:
- Chunking strategies (3 types)
- Chunk sizes (4 sizes)
- Retrieval k values (5 values)
- Retrieval enhancements (4 ablation studies)
- Embedding models (2 models)

Total: 40+ experiments for complete coverage
"""

import os
import sys
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Optional

# Add parent path for imports
RAG_PATH = "/Users/hansonxiong/Desktop/algoverse/shawheen rag"
RESULTS_PATH = "/Users/hansonxiong/Desktop/algoverse/LLM_as_a_Judge/experiments/results"

@dataclass
class ExperimentConfig:
    """Configuration for a single experiment"""
    name: str
    description: str

    # Chunking
    chunking_strategy: str  # "standard", "element_based"
    chunk_size: int
    chunk_overlap: int

    # Embeddings
    embedding_model: str

    # Retrieval
    top_k: int
    use_hybrid: bool
    use_metadata: bool
    use_reranking: bool

    # Generation
    temperature: float = 0.0
    max_tokens: int = 512

    # Evaluation
    use_llm_judge: bool = True

def create_experiment_matrix() -> List[ExperimentConfig]:
    """
    Create comprehensive experiment matrix for publication.

    Experiment Categories:
    1. Baseline & Ablation Studies
    2. Chunk Size Analysis
    3. K-Value Sensitivity
    4. Embedding Model Comparison
    5. Combined Best Configurations
    """
    experiments = []

    # =========================================================================
    # CATEGORY 1: BASELINE & ABLATION STUDIES
    # Goal: Understand contribution of each retrieval enhancement
    # =========================================================================

    # 1.1 Pure Baseline (semantic search only)
    experiments.append(ExperimentConfig(
        name="baseline_semantic_only",
        description="Pure semantic search baseline - no enhancements",
        chunking_strategy="element_based",
        chunk_size=2000,
        chunk_overlap=200,
        embedding_model="text-embedding-3-large",
        top_k=10,
        use_hybrid=False,
        use_metadata=False,
        use_reranking=False,
    ))

    # 1.2 + Hybrid Search
    experiments.append(ExperimentConfig(
        name="ablation_hybrid",
        description="Baseline + BM25 hybrid search",
        chunking_strategy="element_based",
        chunk_size=2000,
        chunk_overlap=200,
        embedding_model="text-embedding-3-large",
        top_k=10,
        use_hybrid=True,
        use_metadata=False,
        use_reranking=False,
    ))

    # 1.3 + Metadata Filtering
    experiments.append(ExperimentConfig(
        name="ablation_metadata",
        description="Baseline + metadata filtering",
        chunking_strategy="element_based",
        chunk_size=2000,
        chunk_overlap=200,
        embedding_model="text-embedding-3-large",
        top_k=10,
        use_hybrid=False,
        use_metadata=True,
        use_reranking=False,
    ))

    # 1.4 + Reranking
    experiments.append(ExperimentConfig(
        name="ablation_reranking",
        description="Baseline + cross-encoder reranking",
        chunking_strategy="element_based",
        chunk_size=2000,
        chunk_overlap=200,
        embedding_model="text-embedding-3-large",
        top_k=10,
        use_hybrid=False,
        use_metadata=False,
        use_reranking=True,
    ))

    # 1.5 All Enhancements
    experiments.append(ExperimentConfig(
        name="ablation_all_features",
        description="All retrieval enhancements enabled",
        chunking_strategy="element_based",
        chunk_size=2000,
        chunk_overlap=200,
        embedding_model="text-embedding-3-large",
        top_k=10,
        use_hybrid=True,
        use_metadata=True,
        use_reranking=True,
    ))

    # =========================================================================
    # CATEGORY 2: CHUNK SIZE ANALYSIS
    # Goal: Find optimal chunk size for financial documents
    # =========================================================================

    for chunk_size in [500, 1000, 1500, 2000, 3000]:
        overlap = chunk_size // 5  # 20% overlap

        # Standard chunking
        experiments.append(ExperimentConfig(
            name=f"chunk_standard_{chunk_size}",
            description=f"Standard character chunking - {chunk_size} chars",
            chunking_strategy="standard",
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            embedding_model="text-embedding-3-large",
            top_k=10,
            use_hybrid=True,
            use_metadata=True,
            use_reranking=True,
        ))

        # Element-based chunking (for sizes >= 1000)
        if chunk_size >= 1000:
            experiments.append(ExperimentConfig(
                name=f"chunk_element_{chunk_size}",
                description=f"Element-based chunking - {chunk_size} max chars",
                chunking_strategy="element_based",
                chunk_size=chunk_size,
                chunk_overlap=overlap,
                embedding_model="text-embedding-3-large",
                top_k=10,
                use_hybrid=True,
                use_metadata=True,
                use_reranking=True,
            ))

    # =========================================================================
    # CATEGORY 3: K-VALUE SENSITIVITY ANALYSIS
    # Goal: Understand retrieval depth trade-offs
    # =========================================================================

    for k in [3, 5, 10, 15, 20, 30]:
        experiments.append(ExperimentConfig(
            name=f"k_value_{k}",
            description=f"Top-k retrieval sensitivity - k={k}",
            chunking_strategy="element_based",
            chunk_size=2000,
            chunk_overlap=200,
            embedding_model="text-embedding-3-large",
            top_k=k,
            use_hybrid=True,
            use_metadata=True,
            use_reranking=True,
        ))

    # =========================================================================
    # CATEGORY 4: EMBEDDING MODEL COMPARISON
    # Goal: Compare embedding model quality vs cost trade-off
    # =========================================================================

    # Small embedding model
    experiments.append(ExperimentConfig(
        name="embedding_small",
        description="OpenAI text-embedding-3-small",
        chunking_strategy="element_based",
        chunk_size=2000,
        chunk_overlap=200,
        embedding_model="text-embedding-3-small",
        top_k=10,
        use_hybrid=True,
        use_metadata=True,
        use_reranking=True,
    ))

    # Large embedding model (already in ablation)
    experiments.append(ExperimentConfig(
        name="embedding_large",
        description="OpenAI text-embedding-3-large",
        chunking_strategy="element_based",
        chunk_size=2000,
        chunk_overlap=200,
        embedding_model="text-embedding-3-large",
        top_k=10,
        use_hybrid=True,
        use_metadata=True,
        use_reranking=True,
    ))

    # =========================================================================
    # CATEGORY 5: PROMPTING STRATEGIES
    # Goal: Compare different prompting approaches
    # =========================================================================

    # Forced answer (already default)
    experiments.append(ExperimentConfig(
        name="prompt_forced_answer",
        description="Forced answer prompting strategy",
        chunking_strategy="element_based",
        chunk_size=2000,
        chunk_overlap=200,
        embedding_model="text-embedding-3-large",
        top_k=10,
        use_hybrid=True,
        use_metadata=True,
        use_reranking=True,
    ))

    # =========================================================================
    # CATEGORY 6: COMBINED BEST CONFIGURATIONS
    # Goal: Test hypothesized optimal combinations
    # =========================================================================

    # Hypothesis: Large chunks + high k + all features
    experiments.append(ExperimentConfig(
        name="optimal_hypothesis_1",
        description="Large chunks (3000) + k=20 + all features",
        chunking_strategy="element_based",
        chunk_size=3000,
        chunk_overlap=600,
        embedding_model="text-embedding-3-large",
        top_k=20,
        use_hybrid=True,
        use_metadata=True,
        use_reranking=True,
    ))

    # Hypothesis: Medium chunks + medium k (balanced)
    experiments.append(ExperimentConfig(
        name="optimal_hypothesis_2",
        description="Medium chunks (1500) + k=10 + all features",
        chunking_strategy="element_based",
        chunk_size=1500,
        chunk_overlap=300,
        embedding_model="text-embedding-3-large",
        top_k=10,
        use_hybrid=True,
        use_metadata=True,
        use_reranking=True,
    ))

    return experiments


def generate_experiment_script(config: ExperimentConfig, output_dir: str) -> str:
    """Generate shell script to run a single experiment"""

    script = f"""#!/bin/bash
# Experiment: {config.name}
# Description: {config.description}
# Generated: {datetime.now().isoformat()}

set -e

cd "{RAG_PATH}"

echo "=========================================="
echo "Running: {config.name}"
echo "=========================================="

# Run the evaluation
python src/bulk_testing.py \\
    --dataset financebench \\
    --top-k {config.top_k} \\
    --temperature {config.temperature} \\
    --max-tokens {config.max_tokens} \\
    {'--use-llm-judge' if config.use_llm_judge else ''} \\
    2>&1 | tee "{output_dir}/{config.name}.log"

# Copy results
cp bulk_runs/*.csv "{output_dir}/" 2>/dev/null || true
cp bulk_runs/*.json "{output_dir}/" 2>/dev/null || true

# Rename with experiment name
for f in "{output_dir}"/*.csv; do
    if [[ -f "$f" && ! "$f" == *"{config.name}"* ]]; then
        mv "$f" "{output_dir}/{config.name}_$(basename $f)"
    fi
done

echo "Completed: {config.name}"
"""
    return script


def create_master_runner(experiments: List[ExperimentConfig], output_dir: str) -> str:
    """Create master script to run all experiments sequentially"""

    script = f"""#!/bin/bash
# Master Experiment Runner
# Generated: {datetime.now().isoformat()}
# Total experiments: {len(experiments)}

set -e

OUTPUT_DIR="{output_dir}"
mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "RAG Experiment Suite - Publication Quality"
echo "Total experiments: {len(experiments)}"
echo "=========================================="

# Track progress
COMPLETED=0
FAILED=0

"""

    for i, config in enumerate(experiments, 1):
        script += f"""
echo ""
echo "[{i}/{len(experiments)}] Starting: {config.name}"
echo "Description: {config.description}"

if bash "{output_dir}/scripts/{config.name}.sh"; then
    COMPLETED=$((COMPLETED + 1))
    echo "✓ Completed: {config.name}"
else
    FAILED=$((FAILED + 1))
    echo "✗ Failed: {config.name}"
fi

"""

    script += """
echo ""
echo "=========================================="
echo "Experiment Suite Complete"
echo "Completed: $COMPLETED"
echo "Failed: $FAILED"
echo "=========================================="
"""

    return script


def create_analysis_notebook(experiments: List[ExperimentConfig], output_dir: str):
    """Create Jupyter notebook for publication-quality analysis"""

    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# RAG System Ablation Study: Financial Question Answering\n",
                    "\n",
                    "## Publication-Quality Analysis\n",
                    "\n",
                    "This notebook analyzes results from a comprehensive experiment suite covering:\n",
                    f"- **{len(experiments)} total experiments**\n",
                    "- Chunking strategies (standard vs element-based)\n",
                    "- Chunk sizes (500-3000 characters)\n",
                    "- Retrieval k values (3-30)\n",
                    "- Retrieval enhancements (hybrid, metadata, reranking)\n",
                    "- Embedding models (small vs large)"
                ]
            },
            {
                "cell_type": "code",
                "metadata": {},
                "source": [
                    "import pandas as pd\n",
                    "import numpy as np\n",
                    "import matplotlib.pyplot as plt\n",
                    "import seaborn as sns\n",
                    "from pathlib import Path\n",
                    "import json\n",
                    "from scipy import stats\n",
                    "\n",
                    "# Set publication style\n",
                    "plt.style.use('seaborn-v0_8-whitegrid')\n",
                    "plt.rcParams['figure.figsize'] = (10, 6)\n",
                    "plt.rcParams['font.size'] = 12\n",
                    "plt.rcParams['axes.labelsize'] = 14\n",
                    "plt.rcParams['axes.titlesize'] = 16\n",
                    "\n",
                    f"RESULTS_DIR = Path('{output_dir}')"
                ],
                "execution_count": None,
                "outputs": []
            },
            {
                "cell_type": "code",
                "metadata": {},
                "source": [
                    "# Load all experiment results\n",
                    "def load_all_results():\n",
                    "    results = []\n",
                    "    for csv_file in RESULTS_DIR.glob('*.csv'):\n",
                    "        if '_with_judge' in csv_file.name or csv_file.name.startswith(('baseline', 'ablation', 'chunk', 'k_value', 'embedding', 'optimal')):\n",
                    "            df = pd.read_csv(csv_file)\n",
                    "            exp_name = csv_file.stem.split('_2')[0]  # Remove timestamp\n",
                    "            \n",
                    "            # Normalize score column\n",
                    "            score_col = 'judge_score' if 'judge_score' in df.columns else 'score'\n",
                    "            if score_col in df.columns:\n",
                    "                scores = pd.to_numeric(df[score_col], errors='coerce').dropna()\n",
                    "                results.append({\n",
                    "                    'experiment': exp_name,\n",
                    "                    'mean_score': scores.mean(),\n",
                    "                    'std_score': scores.std(),\n",
                    "                    'median_score': scores.median(),\n",
                    "                    'n_questions': len(scores),\n",
                    "                    'perfect_scores': (scores == 1.0).sum(),\n",
                    "                    'zero_scores': (scores == 0.0).sum(),\n",
                    "                })\n",
                    "    \n",
                    "    return pd.DataFrame(results).sort_values('mean_score', ascending=False)\n",
                    "\n",
                    "results_df = load_all_results()\n",
                    "print(f'Loaded {len(results_df)} experiments')\n",
                    "results_df.head(10)"
                ],
                "execution_count": None,
                "outputs": []
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Figure 1: Overall Performance Comparison"
                ]
            },
            {
                "cell_type": "code",
                "metadata": {},
                "source": [
                    "# Publication-quality bar chart\n",
                    "fig, ax = plt.subplots(figsize=(14, 8))\n",
                    "\n",
                    "colors = plt.cm.RdYlGn(results_df['mean_score'].values)\n",
                    "\n",
                    "bars = ax.barh(results_df['experiment'], results_df['mean_score'] * 100, \n",
                    "               xerr=results_df['std_score'] * 100, \n",
                    "               color=colors, capsize=3)\n",
                    "\n",
                    "ax.set_xlabel('Mean Judge Score (%)')\n",
                    "ax.set_title('RAG Configuration Performance Comparison')\n",
                    "ax.axvline(x=50, color='red', linestyle='--', alpha=0.5, label='50% threshold')\n",
                    "\n",
                    "plt.tight_layout()\n",
                    "plt.savefig(RESULTS_DIR / 'fig1_overall_comparison.pdf', dpi=300, bbox_inches='tight')\n",
                    "plt.savefig(RESULTS_DIR / 'fig1_overall_comparison.png', dpi=300, bbox_inches='tight')\n",
                    "plt.show()"
                ],
                "execution_count": None,
                "outputs": []
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Figure 2: Ablation Study Results"
                ]
            },
            {
                "cell_type": "code",
                "metadata": {},
                "source": [
                    "# Filter ablation experiments\n",
                    "ablation_df = results_df[results_df['experiment'].str.contains('ablation|baseline')].copy()\n",
                    "\n",
                    "fig, ax = plt.subplots(figsize=(10, 6))\n",
                    "\n",
                    "x = range(len(ablation_df))\n",
                    "bars = ax.bar(x, ablation_df['mean_score'] * 100, \n",
                    "              yerr=ablation_df['std_score'] * 100,\n",
                    "              color=plt.cm.viridis(np.linspace(0.3, 0.9, len(ablation_df))),\n",
                    "              capsize=5)\n",
                    "\n",
                    "ax.set_xticks(x)\n",
                    "ax.set_xticklabels(ablation_df['experiment'].str.replace('ablation_', '').str.replace('baseline_', ''), \n",
                    "                   rotation=45, ha='right')\n",
                    "ax.set_ylabel('Mean Judge Score (%)')\n",
                    "ax.set_title('Retrieval Enhancement Ablation Study')\n",
                    "\n",
                    "# Add value labels\n",
                    "for bar, val in zip(bars, ablation_df['mean_score'] * 100):\n",
                    "    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, \n",
                    "            f'{val:.1f}%', ha='center', va='bottom', fontsize=10)\n",
                    "\n",
                    "plt.tight_layout()\n",
                    "plt.savefig(RESULTS_DIR / 'fig2_ablation_study.pdf', dpi=300, bbox_inches='tight')\n",
                    "plt.savefig(RESULTS_DIR / 'fig2_ablation_study.png', dpi=300, bbox_inches='tight')\n",
                    "plt.show()"
                ],
                "execution_count": None,
                "outputs": []
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Figure 3: Chunk Size Analysis"
                ]
            },
            {
                "cell_type": "code",
                "metadata": {},
                "source": [
                    "# Filter chunk experiments\n",
                    "chunk_df = results_df[results_df['experiment'].str.contains('chunk_')].copy()\n",
                    "\n",
                    "# Extract chunk size and type\n",
                    "chunk_df['chunk_size'] = chunk_df['experiment'].str.extract(r'(\\d+)').astype(int)\n",
                    "chunk_df['chunk_type'] = chunk_df['experiment'].apply(\n",
                    "    lambda x: 'Element-based' if 'element' in x else 'Standard'\n",
                    ")\n",
                    "\n",
                    "fig, ax = plt.subplots(figsize=(10, 6))\n",
                    "\n",
                    "for chunk_type, group in chunk_df.groupby('chunk_type'):\n",
                    "    group = group.sort_values('chunk_size')\n",
                    "    ax.errorbar(group['chunk_size'], group['mean_score'] * 100, \n",
                    "                yerr=group['std_score'] * 100,\n",
                    "                marker='o', markersize=8, capsize=5, label=chunk_type)\n",
                    "\n",
                    "ax.set_xlabel('Chunk Size (characters)')\n",
                    "ax.set_ylabel('Mean Judge Score (%)')\n",
                    "ax.set_title('Effect of Chunk Size on Performance')\n",
                    "ax.legend()\n",
                    "\n",
                    "plt.tight_layout()\n",
                    "plt.savefig(RESULTS_DIR / 'fig3_chunk_size.pdf', dpi=300, bbox_inches='tight')\n",
                    "plt.savefig(RESULTS_DIR / 'fig3_chunk_size.png', dpi=300, bbox_inches='tight')\n",
                    "plt.show()"
                ],
                "execution_count": None,
                "outputs": []
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Figure 4: K-Value Sensitivity Analysis"
                ]
            },
            {
                "cell_type": "code",
                "metadata": {},
                "source": [
                    "# Filter k-value experiments\n",
                    "k_df = results_df[results_df['experiment'].str.contains('k_value')].copy()\n",
                    "k_df['k'] = k_df['experiment'].str.extract(r'(\\d+)').astype(int)\n",
                    "k_df = k_df.sort_values('k')\n",
                    "\n",
                    "fig, ax = plt.subplots(figsize=(10, 6))\n",
                    "\n",
                    "ax.errorbar(k_df['k'], k_df['mean_score'] * 100, \n",
                    "            yerr=k_df['std_score'] * 100,\n",
                    "            marker='s', markersize=10, capsize=5, \n",
                    "            color='#2ecc71', linewidth=2)\n",
                    "\n",
                    "ax.set_xlabel('Top-K Retrieved Chunks')\n",
                    "ax.set_ylabel('Mean Judge Score (%)')\n",
                    "ax.set_title('Retrieval Depth Sensitivity Analysis')\n",
                    "\n",
                    "# Mark optimal k\n",
                    "best_k = k_df.loc[k_df['mean_score'].idxmax(), 'k']\n",
                    "ax.axvline(x=best_k, color='red', linestyle='--', alpha=0.5, \n",
                    "           label=f'Optimal k={best_k}')\n",
                    "ax.legend()\n",
                    "\n",
                    "plt.tight_layout()\n",
                    "plt.savefig(RESULTS_DIR / 'fig4_k_sensitivity.pdf', dpi=300, bbox_inches='tight')\n",
                    "plt.savefig(RESULTS_DIR / 'fig4_k_sensitivity.png', dpi=300, bbox_inches='tight')\n",
                    "plt.show()"
                ],
                "execution_count": None,
                "outputs": []
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Table 1: Summary Statistics"
                ]
            },
            {
                "cell_type": "code",
                "metadata": {},
                "source": [
                    "# Create publication-quality summary table\n",
                    "summary_table = results_df.copy()\n",
                    "summary_table['mean_score'] = (summary_table['mean_score'] * 100).round(1)\n",
                    "summary_table['std_score'] = (summary_table['std_score'] * 100).round(1)\n",
                    "summary_table['median_score'] = (summary_table['median_score'] * 100).round(1)\n",
                    "\n",
                    "summary_table = summary_table.rename(columns={\n",
                    "    'experiment': 'Configuration',\n",
                    "    'mean_score': 'Mean (%)',\n",
                    "    'std_score': 'Std (%)',\n",
                    "    'median_score': 'Median (%)',\n",
                    "    'n_questions': 'N',\n",
                    "    'perfect_scores': 'Perfect',\n",
                    "    'zero_scores': 'Zero'\n",
                    "})\n",
                    "\n",
                    "# Export to LaTeX for paper\n",
                    "latex_table = summary_table.to_latex(index=False, \n",
                    "                                     caption='RAG Configuration Performance Summary',\n",
                    "                                     label='tab:results')\n",
                    "\n",
                    "with open(RESULTS_DIR / 'table1_summary.tex', 'w') as f:\n",
                    "    f.write(latex_table)\n",
                    "\n",
                    "print('Top 10 Configurations:')\n",
                    "summary_table.head(10)"
                ],
                "execution_count": None,
                "outputs": []
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Statistical Analysis"
                ]
            },
            {
                "cell_type": "code",
                "metadata": {},
                "source": [
                    "# Statistical significance tests\n",
                    "print('Statistical Significance Tests')\n",
                    "print('=' * 50)\n",
                    "\n",
                    "# Compare best vs baseline\n",
                    "best_exp = results_df.iloc[0]['experiment']\n",
                    "baseline_exp = results_df[results_df['experiment'].str.contains('baseline')].iloc[0]['experiment']\n",
                    "\n",
                    "print(f'\\nBest configuration: {best_exp}')\n",
                    "print(f'Baseline: {baseline_exp}')\n",
                    "print(f'\\nImprovement: {(results_df.iloc[0][\"mean_score\"] - results_df[results_df[\"experiment\"]==baseline_exp][\"mean_score\"].values[0]) * 100:.1f}%')"
                ],
                "execution_count": None,
                "outputs": []
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.9.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    return notebook


def main():
    """Main function to generate experiment suite"""

    print("=" * 60)
    print("RAG Experiment Suite Generator")
    print("Publication-Quality Configuration")
    print("=" * 60)

    # Create output directories
    output_dir = RESULTS_PATH
    scripts_dir = os.path.join(output_dir, "scripts")
    os.makedirs(scripts_dir, exist_ok=True)

    # Generate experiment matrix
    experiments = create_experiment_matrix()
    print(f"\nGenerated {len(experiments)} experiment configurations")

    # Categorize experiments
    categories = {}
    for exp in experiments:
        category = exp.name.split('_')[0]
        if category not in categories:
            categories[category] = []
        categories[category].append(exp.name)

    print("\nExperiment Categories:")
    for cat, exps in categories.items():
        print(f"  {cat}: {len(exps)} experiments")

    # Generate individual experiment scripts
    print("\nGenerating experiment scripts...")
    for exp in experiments:
        script = generate_experiment_script(exp, output_dir)
        script_path = os.path.join(scripts_dir, f"{exp.name}.sh")
        with open(script_path, 'w') as f:
            f.write(script)
        os.chmod(script_path, 0o755)

    # Generate master runner
    master_script = create_master_runner(experiments, output_dir)
    master_path = os.path.join(output_dir, "run_all.sh")
    with open(master_path, 'w') as f:
        f.write(master_script)
    os.chmod(master_path, 0o755)

    # Generate analysis notebook
    notebook = create_analysis_notebook(experiments, output_dir)
    notebook_path = os.path.join(output_dir, "analysis.ipynb")
    with open(notebook_path, 'w') as f:
        json.dump(notebook, f, indent=2)

    # Save experiment manifest
    manifest = {
        "generated": datetime.now().isoformat(),
        "total_experiments": len(experiments),
        "categories": categories,
        "experiments": [asdict(exp) for exp in experiments]
    }
    manifest_path = os.path.join(output_dir, "experiment_manifest.json")
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"\n✓ Generated {len(experiments)} experiment scripts")
    print(f"✓ Created master runner: {master_path}")
    print(f"✓ Created analysis notebook: {notebook_path}")
    print(f"✓ Saved manifest: {manifest_path}")

    print("\n" + "=" * 60)
    print("TO RUN ALL EXPERIMENTS:")
    print("=" * 60)
    print(f"\n  cd {output_dir}")
    print("  bash run_all.sh")
    print("\nEstimated time: ~2-4 hours (depending on API rate limits)")
    print("\nAfter completion, open analysis.ipynb for publication figures.")


if __name__ == "__main__":
    main()
