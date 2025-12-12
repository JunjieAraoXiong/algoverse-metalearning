"""
Run LLM-as-a-Judge evaluation on existing RAG results.

This script:
1. Loads CSV results from RAG pipeline runs
2. Evaluates each prediction using LLM-as-a-Judge
3. Saves results with judge scores and justifications
4. Generates comparison summary
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv
import openai

load_dotenv()


def llm_as_judge(
    question: str,
    gold_answer: str,
    predicted_answer: str,
    client,
    judge_model: str = "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"
):
    """
    Use an LLM to evaluate the quality of a predicted answer.

    Returns:
        Tuple of (score, justification)
    """
    if not predicted_answer or not gold_answer or pd.isna(predicted_answer) or pd.isna(gold_answer):
        return 0.0, "Empty answer"

    prompt = f"""You are a financial QA evaluation assistant. Your task is to evaluate how well a predicted answer matches the ground truth answer.

Question: {question}

Ground Truth Answer: {gold_answer}

Predicted Answer: {predicted_answer}

Evaluate the predicted answer on a scale from 0.0 to 1.0 based on:
1. **Correctness** - Does it contain the same factual information as the ground truth?
2. **Completeness** - Does it include all key details from the ground truth?
3. **Relevance** - Does it directly answer the question asked?

Scoring guidelines:
- 1.0: Perfect match - contains all correct information
- 0.8-0.9: Very good - minor differences in wording but same meaning
- 0.6-0.7: Good - correct main point but missing some details
- 0.4-0.5: Partial - some correct information but significant gaps
- 0.2-0.3: Poor - mostly incorrect or incomplete
- 0.0-0.1: Wrong - completely incorrect or irrelevant

Return ONLY valid JSON in this exact format (no other text):
{{"score": 0.XX, "justification": "Brief explanation of the score"}}"""

    try:
        response = client.chat.completions.create(
            model=judge_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=200
        )

        content = response.choices[0].message.content.strip()

        # Parse JSON response
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()

        result = json.loads(content)
        score = float(result.get("score", 0.0))
        justification = result.get("justification", "")

        # Ensure score is in valid range
        score = max(0.0, min(1.0, score))

        return score, justification

    except json.JSONDecodeError as e:
        return None, f"Failed to parse: {str(e)}"
    except Exception as e:
        return None, f"Error: {str(e)}"


def evaluate_results(csv_path: str, output_path: str, judge_model: str = "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"):
    """
    Run LLM-as-a-Judge on a CSV results file.
    """
    print(f"\n{'='*60}")
    print(f"Evaluating: {os.path.basename(csv_path)}")
    print(f"{'='*60}")

    # Load results
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} results")

    # Initialize Together API client
    client = openai.OpenAI(
        api_key=os.environ.get("TOGETHER_API_KEY"),
        base_url="https://api.together.xyz/v1"
    )

    # Run LLM-as-a-Judge on each row
    judge_scores = []
    judge_justifications = []

    print(f"\nRunning LLM-as-a-Judge with model: {judge_model}")

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Judging"):
        score, justification = llm_as_judge(
            question=row['question'],
            gold_answer=row['gold_answer'],
            predicted_answer=row['predicted_answer'],
            client=client,
            judge_model=judge_model
        )
        judge_scores.append(score)
        judge_justifications.append(justification)

    # Add judge results to dataframe
    df['judge_score'] = judge_scores
    df['judge_justification'] = judge_justifications

    # Save results
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")

    # Calculate and print summary
    valid_scores = [s for s in judge_scores if s is not None]

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total questions: {len(df)}")
    print(f"Successfully judged: {len(valid_scores)}")
    print(f"\nSemantic Similarity (existing):")
    print(f"  Mean: {df['semantic_similarity'].mean():.4f}")
    print(f"\nLLM Judge Score:")
    print(f"  Mean: {np.mean(valid_scores):.4f}")
    print(f"  Min:  {np.min(valid_scores):.4f}")
    print(f"  Max:  {np.max(valid_scores):.4f}")
    print(f"{'='*60}\n")

    return df


def main():
    # Set up paths
    base_dir = "/Users/hansonxiong/Desktop/algoverse/LLM_as_a_Judge"

    # Results to evaluate
    results_files = [
        {
            "name": "Optimized (element-based chunks)",
            "input": f"{base_dir}/element_based_chunks_2000chars_forced_answer/2025-11-21_22-03-20_fb_llama31-70b_k10_t00.csv",
            "output": f"{base_dir}/element_based_chunks_2000chars_forced_answer/2025-11-21_22-03-20_fb_llama31-70b_k10_t00_with_judge.csv"
        },
        {
            "name": "Simple pipeline",
            "input": f"{base_dir}/simple-pipeline-forced-answer/2025-11-22_00-18-24_fb_llama31-70b_k10_t00.csv",
            "output": f"{base_dir}/simple-pipeline-forced-answer/2025-11-22_00-18-24_fb_llama31-70b_k10_t00_with_judge.csv"
        }
    ]

    # Check for API key
    if not os.environ.get("TOGETHER_API_KEY"):
        print("ERROR: TOGETHER_API_KEY not found in environment")
        print("Please set it in your .env file or export it:")
        print("  export TOGETHER_API_KEY=your_key_here")
        sys.exit(1)

    # Evaluate each results file
    all_results = {}

    for result_info in results_files:
        print(f"\n{'#'*60}")
        print(f"# {result_info['name']}")
        print(f"{'#'*60}")

        if not os.path.exists(result_info['input']):
            print(f"ERROR: File not found: {result_info['input']}")
            continue

        df = evaluate_results(
            csv_path=result_info['input'],
            output_path=result_info['output']
        )

        all_results[result_info['name']] = df

    # Print comparison
    if len(all_results) == 2:
        print(f"\n{'='*60}")
        print("COMPARISON: Optimized vs Simple Pipeline")
        print(f"{'='*60}")

        for name, df in all_results.items():
            valid_judge = df['judge_score'].dropna()
            print(f"\n{name}:")
            print(f"  Semantic Similarity: {df['semantic_similarity'].mean():.4f}")
            print(f"  LLM Judge Score:     {valid_judge.mean():.4f}")

        print(f"\n{'='*60}")


if __name__ == "__main__":
    main()
