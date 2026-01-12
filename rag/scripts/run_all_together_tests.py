#!/usr/bin/env python3
"""
Run comprehensive RAG evaluation tests using Together AI API.
Tests all combinations of pipelines and k values.
"""

import subprocess
import json
import os
from datetime import datetime
from pathlib import Path

# Together AI model
MODEL = "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"

# Test configurations
PIPELINES = ["semantic", "hybrid", "hybrid_filter", "hybrid_filter_rerank"]
K_VALUES = [5, 10, 20]

# Results storage
RESULTS_FILE = Path(__file__).parent.parent / "evaluation" / "together_api_results.json"


def run_experiment(pipeline: str, top_k: int) -> dict:
    """Run a single experiment and return results."""
    print(f"\n{'='*60}")
    print(f"Running: {pipeline} k={top_k}")
    print(f"{'='*60}\n")

    cmd = [
        "python", "src/bulk_testing.py",
        "--dataset", "financebench",
        "--model", MODEL,
        "--pipeline", pipeline,
        "--top-k", str(top_k),
    ]

    # Add reranker flag if needed
    if "rerank" in pipeline:
        cmd.extend(["--use-numeric-verify"])

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour timeout
            cwd=Path(__file__).parent.parent,
        )

        # Parse output for results
        output = result.stdout + result.stderr
        print(output)

        # Extract metrics from output
        metrics = parse_output(output)

        return {
            "pipeline": pipeline,
            "top_k": top_k,
            "model": MODEL,
            "success": result.returncode == 0,
            "metrics": metrics,
            "timestamp": datetime.now().isoformat(),
        }
    except subprocess.TimeoutExpired:
        return {
            "pipeline": pipeline,
            "top_k": top_k,
            "model": MODEL,
            "success": False,
            "error": "Timeout",
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        return {
            "pipeline": pipeline,
            "top_k": top_k,
            "model": MODEL,
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }


def parse_output(output: str) -> dict:
    """Parse bulk_testing.py output to extract metrics."""
    metrics = {}

    # Look for result lines like "Overall mean: 0.604"
    for line in output.split('\n'):
        line = line.strip()
        if 'overall' in line.lower() and ':' in line:
            try:
                metrics['overall'] = float(line.split(':')[-1].strip())
            except:
                pass
        elif 'metrics-generated' in line.lower() and ':' in line:
            try:
                metrics['metrics_generated'] = float(line.split(':')[-1].strip())
            except:
                pass
        elif 'domain-relevant' in line.lower() and ':' in line:
            try:
                metrics['domain_relevant'] = float(line.split(':')[-1].strip())
            except:
                pass
        elif 'novel-generated' in line.lower() and ':' in line:
            try:
                metrics['novel_generated'] = float(line.split(':')[-1].strip())
            except:
                pass

    return metrics


def load_existing_results() -> list:
    """Load existing results if any."""
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE) as f:
            return json.load(f)
    return []


def save_results(results: list):
    """Save results to JSON file."""
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {RESULTS_FILE}")


def print_results_table(results: list):
    """Print a nice results table."""
    print("\n" + "="*80)
    print("RESULTS SUMMARY - Together AI (Llama 3.1 70B Instruct Turbo)")
    print("="*80)

    # Header
    print(f"{'Pipeline':<25} {'k':<5} {'Overall':<10} {'Metrics':<10} {'Domain':<10} {'Novel':<10}")
    print("-"*80)

    for r in sorted(results, key=lambda x: (x['pipeline'], x['top_k'])):
        if r.get('success') and r.get('metrics'):
            m = r['metrics']
            print(f"{r['pipeline']:<25} {r['top_k']:<5} "
                  f"{m.get('overall', 'N/A'):<10.3f} "
                  f"{m.get('metrics_generated', 'N/A'):<10.3f} "
                  f"{m.get('domain_relevant', 'N/A'):<10.3f} "
                  f"{m.get('novel_generated', 'N/A'):<10.3f}")
        else:
            print(f"{r['pipeline']:<25} {r['top_k']:<5} FAILED: {r.get('error', 'Unknown')}")

    print("="*80)

    # Comparison with baseline
    print("\nBaseline Comparison:")
    print("-"*40)
    print("PyPDF + Llama 70B (semantic k=10): 0.485 overall, 0.267 metrics-gen")
    print("Docling + GPT-4o-mini (k=5):       0.604 overall, 0.519 metrics-gen")


def main():
    """Run all experiments."""
    print("="*60)
    print("Together AI RAG Evaluation Suite")
    print(f"Model: {MODEL}")
    print(f"Pipelines: {PIPELINES}")
    print(f"K values: {K_VALUES}")
    print(f"Total experiments: {len(PIPELINES) * len(K_VALUES)}")
    print("="*60)

    # Check for TOGETHER_API_KEY
    if not os.environ.get("TOGETHER_API_KEY"):
        print("\nERROR: TOGETHER_API_KEY environment variable not set!")
        print("Please set it: export TOGETHER_API_KEY='your-api-key'")
        return

    results = load_existing_results()

    # Track which experiments are already done
    done = {(r['pipeline'], r['top_k']) for r in results if r.get('success')}

    for pipeline in PIPELINES:
        for k in K_VALUES:
            if (pipeline, k) in done:
                print(f"\nSkipping {pipeline} k={k} (already completed)")
                continue

            result = run_experiment(pipeline, k)
            results.append(result)
            save_results(results)

    print_results_table(results)


if __name__ == "__main__":
    main()
