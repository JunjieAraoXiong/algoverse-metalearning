#!/usr/bin/env python3
"""Analyze bulk testing results and generate reports."""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


def load_results(csv_path: str) -> pd.DataFrame:
    """Load results CSV."""
    return pd.read_csv(csv_path)


def analyze_by_question_type(df: pd.DataFrame) -> dict:
    """Break down metrics by question type."""
    results = {}

    if 'question_type' not in df.columns:
        return results

    for qtype in df['question_type'].unique():
        subset = df[df['question_type'] == qtype]
        valid = subset[subset['error'].isna()]

        results[qtype] = {
            'count': len(subset),
            'success_count': len(valid),
            'error_rate': 1 - (len(valid) / len(subset)) if len(subset) > 0 else 1.0,
            'semantic_sim_mean': valid['semantic_similarity'].mean() if len(valid) > 0 else 0.0,
            'semantic_sim_std': valid['semantic_similarity'].std() if len(valid) > 0 else 0.0,
        }

    return results


def analyze_errors(df: pd.DataFrame) -> dict:
    """Analyze error patterns."""
    errors = df[df['error'].notna()]

    error_types = {}
    for err in errors['error']:
        # Extract error type
        if 'quota' in str(err).lower():
            key = 'quota_exceeded'
        elif 'dimension' in str(err).lower():
            key = 'dimension_mismatch'
        elif 'timeout' in str(err).lower():
            key = 'timeout'
        elif 'rate' in str(err).lower():
            key = 'rate_limit'
        else:
            key = 'other'

        error_types[key] = error_types.get(key, 0) + 1

    return {
        'total_errors': len(errors),
        'error_rate': len(errors) / len(df) if len(df) > 0 else 0,
        'error_types': error_types,
    }


def analyze_latency(df: pd.DataFrame) -> dict:
    """Analyze latency metrics."""
    valid = df[df['error'].isna()]

    if len(valid) == 0:
        return {'retrieval_ms': {}, 'generation_ms': {}}

    return {
        'retrieval_ms': {
            'mean': valid['retrieval_time_ms'].mean(),
            'p50': valid['retrieval_time_ms'].median(),
            'p95': valid['retrieval_time_ms'].quantile(0.95),
        },
        'generation_ms': {
            'mean': valid['generation_time_ms'].mean(),
            'p50': valid['generation_time_ms'].median(),
            'p95': valid['generation_time_ms'].quantile(0.95),
        },
    }


def find_worst_questions(df: pd.DataFrame, n: int = 10) -> list:
    """Find questions with lowest semantic similarity."""
    valid = df[df['error'].isna()].copy()

    if len(valid) == 0:
        return []

    worst = valid.nsmallest(n, 'semantic_similarity')

    return [
        {
            'question_id': row['question_id'],
            'question': row['question'][:100] + '...' if len(row['question']) > 100 else row['question'],
            'semantic_sim': row['semantic_similarity'],
            'question_type': row.get('question_type', 'unknown'),
        }
        for _, row in worst.iterrows()
    ]


def generate_report(csv_path: str) -> dict:
    """Generate full analysis report."""
    df = load_results(csv_path)

    valid = df[df['error'].isna()]

    report = {
        'summary': {
            'total_questions': len(df),
            'successful': len(valid),
            'error_rate': 1 - (len(valid) / len(df)) if len(df) > 0 else 1.0,
            'semantic_similarity': {
                'mean': valid['semantic_similarity'].mean() if len(valid) > 0 else 0.0,
                'std': valid['semantic_similarity'].std() if len(valid) > 0 else 0.0,
                'min': valid['semantic_similarity'].min() if len(valid) > 0 else 0.0,
                'max': valid['semantic_similarity'].max() if len(valid) > 0 else 0.0,
            },
        },
        'by_question_type': analyze_by_question_type(df),
        'errors': analyze_errors(df),
        'latency': analyze_latency(df),
        'worst_questions': find_worst_questions(df),
    }

    return report


def print_report(report: dict):
    """Pretty print the report."""
    print("\n" + "=" * 60)
    print("RESULTS ANALYSIS")
    print("=" * 60)

    s = report['summary']
    print(f"\nOverall: {s['successful']}/{s['total_questions']} successful ({(1-s['error_rate'])*100:.1f}%)")
    print(f"Semantic Similarity: {s['semantic_similarity']['mean']:.4f} +/- {s['semantic_similarity']['std']:.4f}")

    print("\n--- By Question Type ---")
    for qtype, metrics in report['by_question_type'].items():
        print(f"  {qtype}:")
        print(f"    Semantic Sim: {metrics['semantic_sim_mean']:.4f} ({metrics['success_count']}/{metrics['count']} questions)")

    if report['errors']['total_errors'] > 0:
        print("\n--- Errors ---")
        for err_type, count in report['errors']['error_types'].items():
            print(f"  {err_type}: {count}")

    if report['worst_questions']:
        print("\n--- Worst Performing Questions ---")
        for i, q in enumerate(report['worst_questions'][:5], 1):
            print(f"  {i}. [{q['question_type']}] sim={q['semantic_sim']:.3f}")
            print(f"     {q['question']}")

    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Analyze bulk testing results")
    parser.add_argument('csv_path', help='Path to results CSV')
    parser.add_argument('--json', action='store_true', help='Output as JSON')
    parser.add_argument('--output', '-o', help='Save report to file')

    args = parser.parse_args()

    if not Path(args.csv_path).exists():
        print(f"Error: File not found: {args.csv_path}")
        sys.exit(1)

    report = generate_report(args.csv_path)

    if args.json:
        output = json.dumps(report, indent=2)
        if args.output:
            with open(args.output, 'w') as f:
                f.write(output)
            print(f"Report saved to: {args.output}")
        else:
            print(output)
    else:
        print_report(report)
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"\nJSON report saved to: {args.output}")


if __name__ == "__main__":
    main()
