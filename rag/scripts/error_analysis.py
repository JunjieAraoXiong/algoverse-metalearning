#!/usr/bin/env python3
"""Error analysis for RAG system failures.

Analyzes questions where the system performed poorly to categorize
error types and identify improvement opportunities.

Usage:
    python scripts/error_analysis.py --results PATH [--threshold 0.5]

Output:
    docs/error_analysis.md - Detailed error analysis report
"""

import sys
import re
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
from collections import defaultdict

import pandas as pd
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.meta_learning.features import extract_features


# =============================================================================
# Error Categories
# =============================================================================

ERROR_CATEGORIES = {
    "missing_table": {
        "name": "Missing Table Data",
        "description": "Required table/numerical data not in retrieved context",
        "indicators": ["table", "revenue", "income", "margin", "ratio", "percent"],
    },
    "wrong_period": {
        "name": "Wrong Time Period",
        "description": "Retrieved data from wrong fiscal year/quarter",
        "indicators": ["FY", "fiscal", "year", "quarter", "Q1", "Q2", "Q3", "Q4"],
    },
    "wrong_company": {
        "name": "Wrong Company",
        "description": "Retrieved data from different company",
        "indicators": ["company", "corp", "inc", "llc"],
    },
    "calculation_error": {
        "name": "Calculation Error",
        "description": "LLM computed incorrect result from correct data",
        "indicators": ["calculate", "compute", "ratio", "percent", "growth"],
    },
    "format_mismatch": {
        "name": "Format Mismatch",
        "description": "Answer format differs from expected (units, precision)",
        "indicators": ["$", "million", "billion", "%", "thousands"],
    },
    "hallucination": {
        "name": "Hallucination",
        "description": "Answer contains numbers not present in context",
        "indicators": [],  # Detected programmatically
    },
    "incomplete_answer": {
        "name": "Incomplete Answer",
        "description": "Answer missing required components",
        "indicators": ["and", "both", "all", "each"],
    },
    "retrieval_failure": {
        "name": "Retrieval Failure",
        "description": "No relevant documents retrieved",
        "indicators": [],  # Detected from empty context
    },
}


# =============================================================================
# Error Analyzer
# =============================================================================

class ErrorAnalyzer:
    """Analyzes RAG system errors."""

    def __init__(self, results_path: str, threshold: float = 0.5):
        self.results_path = results_path
        self.threshold = threshold
        self.df = None
        self.errors = []

    def load_results(self):
        """Load evaluation results from CSV."""
        print(f"\nLoading results from: {self.results_path}")
        self.df = pd.read_csv(self.results_path)
        print(f"  Loaded {len(self.df)} results")

        # Get error cases (below threshold)
        self.errors = self.df[
            self.df["semantic_similarity"] < self.threshold
        ].copy()
        print(f"  Errors (similarity < {self.threshold}): {len(self.errors)}")

    def categorize_error(self, row: pd.Series) -> List[str]:
        """Categorize an error based on question and answer content.

        Returns:
            List of applicable error categories
        """
        categories = []
        question = str(row.get("question", "")).lower()
        gold = str(row.get("gold_answer", ""))
        predicted = str(row.get("predicted_answer", ""))
        error_msg = str(row.get("error", ""))

        # Check for retrieval failure
        if "No relevant documents" in error_msg or not predicted:
            categories.append("retrieval_failure")
            return categories  # If no retrieval, other categories don't apply

        # Check for hallucination (numbers in predicted but not in gold)
        pred_numbers = set(re.findall(r"[\d,]+\.?\d*", predicted))
        gold_numbers = set(re.findall(r"[\d,]+\.?\d*", gold))
        # Normalize numbers for comparison (keep all numbers, including single digits)
        pred_norm = {n for n in (self._normalize_number(x) for x in pred_numbers) if n}
        gold_norm = {n for n in (self._normalize_number(x) for x in gold_numbers) if n}
        if pred_norm and not pred_norm.intersection(gold_norm):
            categories.append("hallucination")

        # Check for format mismatch
        if self._has_format_mismatch(gold, predicted):
            categories.append("format_mismatch")

        # Check question-based categories
        features = extract_features(question)

        # Time period issues (question mentions time but answer seems wrong)
        if features.get("has_year") or features.get("has_quarter"):
            if self._likely_period_error(question, gold, predicted):
                categories.append("wrong_period")

        # Table data issues
        if features.get("has_metric_name") or features.get("expects_number"):
            if len(pred_numbers) == 0 and len(gold_numbers) > 0:
                categories.append("missing_table")

        # Calculation issues
        if any(kw in question for kw in ["ratio", "percentage", "growth", "change"]):
            if self._likely_calculation_error(gold, predicted):
                categories.append("calculation_error")

        # Default to missing_table if numeric question with low score
        if not categories and features.get("expects_number"):
            categories.append("missing_table")

        # Default category
        if not categories:
            categories.append("incomplete_answer")

        return categories

    def _normalize_number(self, num_str: str) -> Optional[str]:
        """Normalize a number string for comparison.

        Returns None if the string cannot be parsed as a number.
        """
        # Remove commas and convert to float
        try:
            return str(float(num_str.replace(",", "")))
        except ValueError:
            return None

    def _has_format_mismatch(self, gold: str, predicted: str) -> bool:
        """Check if answers have format differences."""
        # Check for unit differences
        gold_has_dollar = "$" in gold
        pred_has_dollar = "$" in predicted
        if gold_has_dollar != pred_has_dollar:
            return True

        # Check for magnitude differences (million vs billion)
        gold_millions = "million" in gold.lower()
        gold_billions = "billion" in gold.lower()
        pred_millions = "million" in predicted.lower()
        pred_billions = "billion" in predicted.lower()
        if gold_millions != pred_millions or gold_billions != pred_billions:
            return True

        return False

    def _likely_period_error(
        self, question: str, gold: str, predicted: str
    ) -> bool:
        """Check if error is likely due to wrong time period."""
        # Extract years from question
        q_years = re.findall(r"20\d{2}", question)
        if not q_years:
            return False

        # Check if predicted mentions a different year
        pred_years = re.findall(r"20\d{2}", predicted)
        for py in pred_years:
            if py not in q_years:
                return True

        return False

    def _likely_calculation_error(self, gold: str, predicted: str) -> bool:
        """Check if error is likely a calculation mistake."""
        # Extract numbers
        gold_nums = re.findall(r"[\d,]+\.?\d*", gold)
        pred_nums = re.findall(r"[\d,]+\.?\d*", predicted)

        if not gold_nums or not pred_nums:
            return False

        # Check if numbers are close but not exact (calculation error)
        try:
            g_val = float(gold_nums[0].replace(",", ""))
            p_val = float(pred_nums[0].replace(",", ""))

            # Within 20% but not exact
            ratio = p_val / g_val if g_val != 0 else 0
            if 0.8 < ratio < 1.2 and ratio != 1.0:
                return True
        except (ValueError, IndexError):
            pass

        return False

    def analyze(self) -> Dict[str, Any]:
        """Run full error analysis.

        Returns:
            Dictionary with analysis results
        """
        print("\n" + "=" * 60)
        print("ERROR ANALYSIS")
        print("=" * 60)

        results = {
            "timestamp": datetime.now().isoformat(),
            "total_questions": len(self.df),
            "error_count": len(self.errors),
            "error_rate": len(self.errors) / len(self.df),
            "threshold": self.threshold,
            "categories": defaultdict(list),
            "by_question_type": defaultdict(lambda: defaultdict(int)),
        }

        # Categorize each error
        category_counts = defaultdict(int)
        category_examples = defaultdict(list)

        for idx, row in self.errors.iterrows():
            categories = self.categorize_error(row)

            for cat in categories:
                category_counts[cat] += 1

                # Store example (max 3 per category)
                if len(category_examples[cat]) < 3:
                    category_examples[cat].append({
                        "question": row["question"],
                        "gold_answer": row["gold_answer"],
                        "predicted_answer": row.get("predicted_answer", ""),
                        "similarity": row["semantic_similarity"],
                        "question_type": row.get("question_type", "unknown"),
                    })

            # Track by question type
            qtype = row.get("question_type", "unknown")
            for cat in categories:
                results["by_question_type"][qtype][cat] += 1

        results["category_counts"] = dict(category_counts)
        results["category_examples"] = dict(category_examples)
        results["by_question_type"] = {
            k: dict(v) for k, v in results["by_question_type"].items()
        }

        # Print summary
        print(f"\nTotal errors: {len(self.errors)} / {len(self.df)} "
              f"({len(self.errors)/len(self.df)*100:.1f}%)")

        print("\nError distribution:")
        for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
            pct = count / len(self.errors) * 100
            name = ERROR_CATEGORIES.get(cat, {}).get("name", cat)
            print(f"  {name}: {count} ({pct:.1f}%)")

        print("\nBy question type:")
        for qtype, cats in results["by_question_type"].items():
            print(f"\n  {qtype}:")
            for cat, count in sorted(cats.items(), key=lambda x: -x[1]):
                name = ERROR_CATEGORIES.get(cat, {}).get("name", cat)
                print(f"    {name}: {count}")

        return results

    def generate_report(self, output_path: str, results: Dict[str, Any]):
        """Generate markdown error analysis report."""
        print(f"\nGenerating report: {output_path}")

        lines = [
            "# Error Analysis Report",
            "",
            f"Generated: {results['timestamp']}",
            f"Results file: `{self.results_path}`",
            "",
            "## Summary",
            "",
            f"- **Total questions:** {results['total_questions']}",
            f"- **Errors (similarity < {results['threshold']}):** {results['error_count']}",
            f"- **Error rate:** {results['error_rate']*100:.1f}%",
            "",
            "## Error Categories",
            "",
            "| Category | Count | Percentage |",
            "|----------|-------|------------|",
        ]

        # Category table
        for cat, count in sorted(
            results["category_counts"].items(), key=lambda x: -x[1]
        ):
            name = ERROR_CATEGORIES.get(cat, {}).get("name", cat)
            pct = count / results["error_count"] * 100
            lines.append(f"| {name} | {count} | {pct:.1f}% |")

        lines.extend([
            "",
            "## Category Descriptions",
            "",
        ])

        # Category descriptions
        for cat_id, cat_info in ERROR_CATEGORIES.items():
            if cat_id in results["category_counts"]:
                lines.extend([
                    f"### {cat_info['name']}",
                    "",
                    f"**Description:** {cat_info['description']}",
                    "",
                ])

        lines.extend([
            "",
            "## Examples by Category",
            "",
        ])

        # Examples
        for cat, examples in results["category_examples"].items():
            name = ERROR_CATEGORIES.get(cat, {}).get("name", cat)
            lines.extend([
                f"### {name}",
                "",
            ])

            for i, ex in enumerate(examples, 1):
                lines.extend([
                    f"**Example {i}** (similarity: {ex['similarity']:.3f}, "
                    f"type: {ex['question_type']})",
                    "",
                    f"- **Question:** {ex['question']}",
                    f"- **Gold:** {ex['gold_answer'][:200]}...",
                    f"- **Predicted:** {ex['predicted_answer'][:200]}...",
                    "",
                ])

        lines.extend([
            "",
            "## Breakdown by Question Type",
            "",
        ])

        # By question type
        for qtype, cats in results["by_question_type"].items():
            lines.extend([
                f"### {qtype}",
                "",
                "| Error Category | Count |",
                "|----------------|-------|",
            ])
            for cat, count in sorted(cats.items(), key=lambda x: -x[1]):
                name = ERROR_CATEGORIES.get(cat, {}).get("name", cat)
                lines.append(f"| {name} | {count} |")
            lines.append("")

        lines.extend([
            "",
            "## Recommendations",
            "",
            "Based on this analysis, consider the following improvements:",
            "",
        ])

        # Generate recommendations based on top errors
        top_errors = sorted(
            results["category_counts"].items(), key=lambda x: -x[1]
        )[:3]

        recommendations = {
            "missing_table": "- **Improve table extraction:** Ensure tables are preserved as atomic units during chunking. Consider using TableFormer or similar for better OCR.",
            "wrong_period": "- **Add temporal filtering:** Implement metadata filtering by fiscal year/quarter during retrieval.",
            "wrong_company": "- **Add company filtering:** Use company name extraction and metadata filtering.",
            "calculation_error": "- **Add verification step:** Implement numeric verification to check calculations against source data.",
            "format_mismatch": "- **Standardize output format:** Add post-processing to normalize units and number formats.",
            "hallucination": "- **Strengthen grounding:** Add explicit context verification step before generating answer.",
            "incomplete_answer": "- **Improve prompting:** Add explicit instructions to address all parts of multi-part questions.",
            "retrieval_failure": "- **Expand retrieval:** Increase top-k or use query expansion for low-recall queries.",
        }

        for cat, _ in top_errors:
            if cat in recommendations:
                lines.append(recommendations[cat])

        # Write report
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write("\n".join(lines))

        print(f"  Report saved to: {output_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Analyze RAG system errors"
    )
    parser.add_argument(
        "--results",
        type=str,
        required=True,
        help="Path to evaluation results CSV",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Similarity threshold for error classification (default: 0.5)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="docs/error_analysis.md",
        help="Output report path (default: docs/error_analysis.md)",
    )
    args = parser.parse_args()

    # Resolve paths
    project_root = Path(__file__).parent.parent
    results_path = args.results
    if not Path(results_path).is_absolute():
        results_path = str(project_root / results_path)
    output_path = str(project_root / args.output)

    print("=" * 60)
    print("ERROR ANALYSIS")
    print("=" * 60)
    print(f"Results: {results_path}")
    print(f"Threshold: {args.threshold}")
    print(f"Output: {output_path}")
    print("=" * 60)

    # Check if results file exists
    if not Path(results_path).exists():
        print(f"\nERROR: Results file not found: {results_path}")
        sys.exit(1)

    # Run analysis
    analyzer = ErrorAnalyzer(results_path, threshold=args.threshold)
    analyzer.load_results()
    results = analyzer.analyze()

    # Generate report
    analyzer.generate_report(output_path, results)

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
