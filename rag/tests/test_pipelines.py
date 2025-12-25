"""Parametrized smoke test for retrieval pipelines."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.bulk_testing import BulkTestRunner, BulkTestConfig  # noqa: E402
from dataset_adapters import FinanceBenchAdapter  # noqa: E402


PIPELINES = ["semantic", "hybrid", "hybrid_filter", "hybrid_filter_rerank"]
SUBSET_CSV = "data/question_sets/financebench_subset_questions.csv"


def run_pipeline(pipeline_id: str) -> None:
    print("=" * 80)
    print(f"TEST: PIPELINE = {pipeline_id}")
    print("=" * 80)

    config = BulkTestConfig(
        dataset_name="financebench",
        pipeline_id=pipeline_id,
        top_k_retrieval=5,
        temperature=0.0,
        max_tokens=256,
        initial_k_factor=3.0,
    )

    adapter = FinanceBenchAdapter(subset_csv=SUBSET_CSV)
    runner = BulkTestRunner(config)
    results_df = runner.run_bulk_test(adapter)
    runner.save_results(results_df, adapter)


if __name__ == "__main__":
    for pid in PIPELINES:
        run_pipeline(pid)
