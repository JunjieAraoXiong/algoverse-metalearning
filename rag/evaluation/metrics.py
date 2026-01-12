"""Evaluation metrics for RAG system."""

from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd


def categorize_failure(row: Dict[str, Any]) -> str:
    """Categorize why a question failed to get a good answer.

    Categories:
    - 'ok': Answer is acceptable (semantic_similarity >= 0.5)
    - 'error': Processing error occurred
    - 'retrieval_empty': No documents retrieved
    - 'numeric_hallucination': Answer contains hallucinated numbers
    - 'generation_poor': Retrieved docs but generated poor answer

    Args:
        row: Dictionary or Series with result fields

    Returns:
        Category string
    """
    # Check for errors first
    if row.get('error'):
        return 'error'

    # Check for empty retrieval
    if not row.get('sources'):
        return 'retrieval_empty'

    # Check for numeric hallucination
    numeric_score = row.get('numeric_score')
    if numeric_score is not None and numeric_score < 0.5:
        return 'numeric_hallucination'

    # Check semantic similarity
    sem_sim = row.get('semantic_similarity', 0)
    if sem_sim < 0.5:
        return 'generation_poor'

    return 'ok'


def calculate_failure_breakdown(results_df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate breakdown of failure categories.

    Args:
        results_df: DataFrame with evaluation results

    Returns:
        Dictionary with failure category counts and percentages
    """
    categories = results_df.apply(
        lambda row: categorize_failure(row.to_dict()),
        axis=1
    )

    counts = categories.value_counts().to_dict()
    total = len(results_df)

    breakdown = {
        'counts': counts,
        'percentages': {k: v / total for k, v in counts.items()},
        'total': total,
    }

    return breakdown


def embedding_similarity(
    predicted: str,
    gold: str,
    embeddings,
) -> float:
    """Calculate cosine similarity between predicted and gold answers using embeddings.

    Args:
        predicted: The predicted answer text
        gold: The gold/reference answer text
        embeddings: Embedding model instance (HuggingFaceEmbeddings or similar)

    Returns:
        Cosine similarity score between 0 and 1
    """
    if not predicted or not gold:
        return 0.0

    try:
        # Get embeddings for both texts
        pred_embedding = embeddings.embed_query(predicted)
        gold_embedding = embeddings.embed_query(gold)

        # Convert to numpy arrays
        pred_vec = np.array(pred_embedding)
        gold_vec = np.array(gold_embedding)

        # Calculate cosine similarity
        dot_product = np.dot(pred_vec, gold_vec)
        pred_norm = np.linalg.norm(pred_vec)
        gold_norm = np.linalg.norm(gold_vec)

        if pred_norm == 0 or gold_norm == 0:
            return 0.0

        similarity = dot_product / (pred_norm * gold_norm)

        # Clamp to [0, 1] range (cosine similarity can be negative)
        return float(max(0.0, min(1.0, similarity)))

    except Exception as e:
        print(f"Error calculating embedding similarity: {e}")
        return 0.0


def calculate_aggregate_metrics(results_df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate aggregate metrics from evaluation results.

    Args:
        results_df: DataFrame with evaluation results including 'semantic_similarity',
                   optionally 'judge_score' and 'question_type' columns

    Returns:
        Dictionary with aggregate metrics
    """
    metrics = {}

    # Overall semantic similarity
    if 'semantic_similarity' in results_df.columns:
        sim_values = results_df['semantic_similarity'].dropna()
        metrics['semantic_similarity'] = {
            'mean': float(sim_values.mean()) if len(sim_values) > 0 else 0.0,
            'std': float(sim_values.std()) if len(sim_values) > 0 else 0.0,
            'min': float(sim_values.min()) if len(sim_values) > 0 else 0.0,
            'max': float(sim_values.max()) if len(sim_values) > 0 else 0.0,
            'count': int(len(sim_values)),
        }

    # LLM judge scores if available
    if 'judge_score' in results_df.columns:
        judge_values = results_df['judge_score'].dropna()
        metrics['judge_score'] = {
            'mean': float(judge_values.mean()) if len(judge_values) > 0 else 0.0,
            'std': float(judge_values.std()) if len(judge_values) > 0 else 0.0,
            'accuracy': float((judge_values >= 0.5).mean()) if len(judge_values) > 0 else 0.0,
            'count': int(len(judge_values)),
        }

    # Per question type breakdown
    if 'question_type' in results_df.columns:
        metrics['by_question_type'] = {}
        for q_type in results_df['question_type'].unique():
            if pd.isna(q_type):
                continue
            type_df = results_df[results_df['question_type'] == q_type]
            type_metrics = {}

            if 'semantic_similarity' in type_df.columns:
                sim_vals = type_df['semantic_similarity'].dropna()
                type_metrics['semantic_similarity_mean'] = float(sim_vals.mean()) if len(sim_vals) > 0 else 0.0
                type_metrics['count'] = int(len(sim_vals))

            if 'judge_score' in type_df.columns:
                judge_vals = type_df['judge_score'].dropna()
                type_metrics['judge_score_mean'] = float(judge_vals.mean()) if len(judge_vals) > 0 else 0.0
                type_metrics['judge_accuracy'] = float((judge_vals >= 0.5).mean()) if len(judge_vals) > 0 else 0.0

            metrics['by_question_type'][str(q_type)] = type_metrics

    # Timing metrics
    if 'retrieval_time_ms' in results_df.columns:
        retrieval_times = results_df['retrieval_time_ms'].dropna()
        metrics['retrieval_time_ms'] = {
            'mean': float(retrieval_times.mean()) if len(retrieval_times) > 0 else 0.0,
            'p50': float(retrieval_times.median()) if len(retrieval_times) > 0 else 0.0,
            'p95': float(retrieval_times.quantile(0.95)) if len(retrieval_times) > 0 else 0.0,
        }

    if 'generation_time_ms' in results_df.columns:
        gen_times = results_df['generation_time_ms'].dropna()
        metrics['generation_time_ms'] = {
            'mean': float(gen_times.mean()) if len(gen_times) > 0 else 0.0,
            'p50': float(gen_times.median()) if len(gen_times) > 0 else 0.0,
            'p95': float(gen_times.quantile(0.95)) if len(gen_times) > 0 else 0.0,
        }

    # Error rate
    if 'error' in results_df.columns:
        error_count = results_df['error'].notna().sum()
        metrics['error_rate'] = float(error_count / len(results_df)) if len(results_df) > 0 else 0.0

    # Numeric verification metrics
    if 'numeric_score' in results_df.columns:
        numeric_values = results_df['numeric_score'].dropna()
        if len(numeric_values) > 0:
            metrics['numeric_verification'] = {
                'mean': float(numeric_values.mean()),
                'hallucination_rate': float((numeric_values < 1.0).mean()),
                'perfect_rate': float((numeric_values == 1.0).mean()),
                'count': int(len(numeric_values)),
            }

    # Failure breakdown - categorize WHY questions failed
    metrics['failure_breakdown'] = calculate_failure_breakdown(results_df)

    return metrics


def format_metrics_summary(metrics: Dict[str, Any]) -> str:
    """Format metrics dictionary into a human-readable summary.

    Args:
        metrics: Dictionary of metrics from calculate_aggregate_metrics

    Returns:
        Formatted string summary
    """
    lines = []
    lines.append("=" * 60)
    lines.append("EVALUATION METRICS SUMMARY")
    lines.append("=" * 60)

    # Overall semantic similarity
    if 'semantic_similarity' in metrics:
        sim = metrics['semantic_similarity']
        lines.append(f"\nSemantic Similarity:")
        lines.append(f"  Mean:  {sim['mean']:.4f}")
        lines.append(f"  Std:   {sim['std']:.4f}")
        lines.append(f"  Range: [{sim['min']:.4f}, {sim['max']:.4f}]")
        lines.append(f"  Count: {sim['count']}")

    # LLM Judge scores
    if 'judge_score' in metrics:
        judge = metrics['judge_score']
        lines.append(f"\nLLM Judge:")
        lines.append(f"  Mean Score: {judge['mean']:.4f}")
        lines.append(f"  Accuracy:   {judge['accuracy']:.2%}")
        lines.append(f"  Count:      {judge['count']}")

    # Per question type
    if 'by_question_type' in metrics and metrics['by_question_type']:
        lines.append(f"\nBy Question Type:")
        for q_type, type_metrics in metrics['by_question_type'].items():
            lines.append(f"  {q_type}:")
            if 'semantic_similarity_mean' in type_metrics:
                lines.append(f"    Semantic Sim: {type_metrics['semantic_similarity_mean']:.4f}")
            if 'judge_score_mean' in type_metrics:
                lines.append(f"    Judge Score:  {type_metrics['judge_score_mean']:.4f}")
            if 'count' in type_metrics:
                lines.append(f"    Count:        {type_metrics['count']}")

    # Timing
    if 'retrieval_time_ms' in metrics:
        ret = metrics['retrieval_time_ms']
        lines.append(f"\nRetrieval Latency (ms):")
        lines.append(f"  Mean: {ret['mean']:.1f}  P50: {ret['p50']:.1f}  P95: {ret['p95']:.1f}")

    if 'generation_time_ms' in metrics:
        gen = metrics['generation_time_ms']
        lines.append(f"\nGeneration Latency (ms):")
        lines.append(f"  Mean: {gen['mean']:.1f}  P50: {gen['p50']:.1f}  P95: {gen['p95']:.1f}")

    # Error rate
    if 'error_rate' in metrics:
        lines.append(f"\nError Rate: {metrics['error_rate']:.2%}")

    # Numeric verification
    if 'numeric_verification' in metrics:
        num = metrics['numeric_verification']
        lines.append(f"\nNumeric Verification:")
        lines.append(f"  Mean Score:        {num['mean']:.4f}")
        lines.append(f"  Hallucination Rate: {num['hallucination_rate']:.2%}")
        lines.append(f"  Perfect Rate:      {num['perfect_rate']:.2%}")
        lines.append(f"  Count:             {num['count']}")

    # Failure breakdown
    if 'failure_breakdown' in metrics:
        fb = metrics['failure_breakdown']
        lines.append(f"\nFailure Breakdown:")
        for category, pct in sorted(fb['percentages'].items(), key=lambda x: -x[1]):
            count = fb['counts'].get(category, 0)
            lines.append(f"  {category:25} {pct:6.1%} ({count})")

    lines.append("\n" + "=" * 60)

    return "\n".join(lines)
