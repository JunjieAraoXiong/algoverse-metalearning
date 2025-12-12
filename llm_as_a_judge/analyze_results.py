#!/usr/bin/env python3
"""
LLM-as-a-Judge Results Analysis Dashboard
Generates comprehensive metrics breakdowns and visualizations
"""

import pandas as pd
import os
import json
from pathlib import Path

# Try to import visualization libraries
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Note: Install matplotlib for visualizations: pip install matplotlib")

def load_all_results(base_path):
    """Load all CSV results from both sources"""
    results = {}

    # Junjie's runs (your runs)
    junjie_path = os.path.join(base_path, 'junjie_runs')
    if os.path.exists(junjie_path):
        for folder in os.listdir(junjie_path):
            folder_path = os.path.join(junjie_path, folder)
            if os.path.isdir(folder_path):
                for f in os.listdir(folder_path):
                    if f.endswith('_with_judge.csv'):
                        df = pd.read_csv(os.path.join(folder_path, f))
                        # Normalize column names
                        if 'judge_score' in df.columns:
                            df['score'] = df['judge_score']
                        name = f"[Junjie] {folder}"
                        results[name] = df

    # Aum's runs (teammate runs)
    aum_path = os.path.join(base_path, 'aum_runs')
    if os.path.exists(aum_path):
        for f in os.listdir(aum_path):
            if f.endswith('.csv'):
                df = pd.read_csv(os.path.join(aum_path, f))
                name = f"[Aum] {f.replace('.csv', '')}"
                results[name] = df

    return results

def calculate_overall_metrics(results):
    """Calculate overall scores for each configuration"""
    metrics = []
    for name, df in results.items():
        scores = pd.to_numeric(df['score'], errors='coerce').dropna()
        metrics.append({
            'Configuration': name,
            'Avg Score': round(scores.mean() * 100, 1),
            'Median': round(scores.median() * 100, 1),
            'Std Dev': round(scores.std() * 100, 1),
            'Min': round(scores.min() * 100, 1),
            'Max': round(scores.max() * 100, 1),
            'N Questions': len(scores),
            'Perfect (1.0)': (scores == 1.0).sum(),
            'Zero (0.0)': (scores == 0.0).sum()
        })

    return pd.DataFrame(metrics).sort_values('Avg Score', ascending=False)

def breakdown_by_question_type(results):
    """Analyze scores by question type"""
    all_breakdowns = []

    for name, df in results.items():
        if 'question_type' in df.columns:
            for qtype in df['question_type'].unique():
                subset = df[df['question_type'] == qtype]
                scores = pd.to_numeric(subset['score'], errors='coerce').dropna()
                if len(scores) > 0:
                    all_breakdowns.append({
                        'Configuration': name,
                        'Question Type': qtype,
                        'Avg Score': round(scores.mean() * 100, 1),
                        'Count': len(scores)
                    })

    return pd.DataFrame(all_breakdowns)

def breakdown_by_company(results):
    """Analyze scores by company"""
    all_breakdowns = []

    for name, df in results.items():
        if 'company' in df.columns:
            for company in df['company'].unique():
                subset = df[df['company'] == company]
                scores = pd.to_numeric(subset['score'], errors='coerce').dropna()
                if len(scores) > 0:
                    all_breakdowns.append({
                        'Configuration': name,
                        'Company': company,
                        'Avg Score': round(scores.mean() * 100, 1),
                        'Count': len(scores)
                    })

    return pd.DataFrame(all_breakdowns)

def breakdown_by_doc_type(results):
    """Analyze scores by document type"""
    all_breakdowns = []

    for name, df in results.items():
        if 'doc_type' in df.columns:
            for dtype in df['doc_type'].unique():
                subset = df[df['doc_type'] == dtype]
                scores = pd.to_numeric(subset['score'], errors='coerce').dropna()
                if len(scores) > 0:
                    all_breakdowns.append({
                        'Configuration': name,
                        'Doc Type': dtype,
                        'Avg Score': round(scores.mean() * 100, 1),
                        'Count': len(scores)
                    })

    return pd.DataFrame(all_breakdowns)

def error_analysis(results):
    """Analyze common failure patterns"""
    errors = []

    for name, df in results.items():
        scores = pd.to_numeric(df['score'], errors='coerce')

        # Find worst performing questions
        worst = df.nsmallest(3, 'score') if 'score' in df.columns else pd.DataFrame()

        for _, row in worst.iterrows():
            errors.append({
                'Configuration': name,
                'Question': row.get('question', '')[:100] + '...',
                'Score': row.get('score', 0),
                'Predicted': str(row.get('predicted_answer', ''))[:100] + '...',
                'Gold': str(row.get('gold_answer', ''))[:100] + '...'
            })

    return pd.DataFrame(errors)

def create_visualizations(overall_metrics, base_path):
    """Create visualization charts"""
    if not HAS_MATPLOTLIB:
        print("Skipping visualizations (matplotlib not installed)")
        return

    output_dir = os.path.join(base_path, 'visualizations')
    os.makedirs(output_dir, exist_ok=True)

    # 1. Overall scores bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ['#2ecc71' if '[Junjie]' in name else '#3498db'
              for name in overall_metrics['Configuration']]

    bars = ax.barh(overall_metrics['Configuration'], overall_metrics['Avg Score'], color=colors)
    ax.set_xlabel('Average Score (%)')
    ax.set_title('LLM-as-a-Judge Scores by Configuration')
    ax.axvline(x=50, color='red', linestyle='--', alpha=0.5, label='50% threshold')

    # Add value labels
    for bar, val in zip(bars, overall_metrics['Avg Score']):
        ax.text(val + 1, bar.get_y() + bar.get_height()/2, f'{val}%',
                va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'overall_scores.png'), dpi=150)
    plt.close()

    # 2. Score distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(overall_metrics['Configuration'], overall_metrics['Perfect (1.0)'],
           label='Perfect Score', color='#2ecc71')
    ax.bar(overall_metrics['Configuration'], overall_metrics['Zero (0.0)'],
           bottom=overall_metrics['Perfect (1.0)'], label='Zero Score', color='#e74c3c')
    ax.set_ylabel('Number of Questions')
    ax.set_title('Perfect vs Zero Scores by Configuration')
    ax.legend()
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'score_distribution.png'), dpi=150)
    plt.close()

    print(f"Visualizations saved to: {output_dir}/")

def generate_report(base_path):
    """Generate comprehensive analysis report"""
    print("=" * 70)
    print("LLM-as-a-Judge Comprehensive Analysis Report")
    print("=" * 70)

    results = load_all_results(base_path)

    if not results:
        print("No results found!")
        return

    # Overall metrics
    print("\n📊 OVERALL PERFORMANCE RANKINGS")
    print("-" * 70)
    overall = calculate_overall_metrics(results)
    print(overall.to_string(index=False))

    # Question type breakdown
    print("\n\n📋 BREAKDOWN BY QUESTION TYPE")
    print("-" * 70)
    qtype_df = breakdown_by_question_type(results)
    if not qtype_df.empty:
        # Show pivot table for your best config
        your_configs = [c for c in results.keys() if '[Junjie]' in c]
        if your_configs:
            best_config = your_configs[0]
            pivot = qtype_df[qtype_df['Configuration'] == best_config]
            print(f"\nYour best config ({best_config}):")
            print(pivot[['Question Type', 'Avg Score', 'Count']].to_string(index=False))

    # Company breakdown
    print("\n\n🏢 BREAKDOWN BY COMPANY")
    print("-" * 70)
    company_df = breakdown_by_company(results)
    if not company_df.empty:
        your_configs = [c for c in results.keys() if '[Junjie]' in c]
        if your_configs:
            best_config = your_configs[0]
            pivot = company_df[company_df['Configuration'] == best_config]
            print(f"\nYour best config ({best_config}):")
            print(pivot[['Company', 'Avg Score', 'Count']].to_string(index=False))

    # Error analysis
    print("\n\n❌ ERROR ANALYSIS (Worst Performing Questions)")
    print("-" * 70)
    errors_df = error_analysis(results)
    if not errors_df.empty:
        your_errors = errors_df[errors_df['Configuration'].str.contains(r'\[Junjie\]')]
        if not your_errors.empty:
            print("\nYour worst performing questions:")
            for _, row in your_errors.iterrows():
                print(f"\n  Config: {row['Configuration']}")
                print(f"  Score: {row['Score']}")
                print(f"  Question: {row['Question']}")

    # Create visualizations
    create_visualizations(overall, base_path)

    # Save detailed report
    report_path = os.path.join(base_path, 'detailed_analysis.csv')
    overall.to_csv(report_path, index=False)
    print(f"\n\n📄 Detailed metrics saved to: {report_path}")

    # Recommendations
    print("\n\n💡 RECOMMENDATIONS FOR ADDITIONAL EXPERIMENTS")
    print("-" * 70)
    print("""
1. CHUNK SIZE VARIATIONS
   - Test 500, 1000, 1500, 2000, 3000 character chunks
   - Compare element-based vs character-based at each size

2. EMBEDDING MODELS
   - OpenAI text-embedding-3-small vs text-embedding-3-large
   - Open source: BGE, E5, Instructor embeddings

3. RETRIEVAL STRATEGIES
   - k=5, k=10, k=15, k=20 retrieved chunks
   - Hybrid search (dense + sparse/BM25)
   - Reranking with cross-encoders

4. PROMPTING STRATEGIES
   - Forced answer vs natural response
   - Chain-of-thought prompting
   - Few-shot examples in prompt

5. METADATA FILTERING
   - Filter by company/document type
   - Temporal filtering for date-specific questions
    """)

    return overall

if __name__ == "__main__":
    base_path = os.path.dirname(os.path.abspath(__file__))
    generate_report(base_path)
