"""PubMedQA evaluation - no retrieval needed, context is provided.

This is a reading comprehension task where the model must answer
yes/no/maybe based on a provided biomedical research abstract.

SOTA techniques implemented:
- Few-shot prompting (+4-5% accuracy)
- Self-consistency voting (+2-3% accuracy)
- Optimized prompts based on SOTA analysis
"""

import json
import argparse
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from collections import Counter

from tqdm import tqdm
from dotenv import load_dotenv

# Reuse providers from main rag project
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "rag"))
from src.providers import get_provider
from src.config import DEFAULTS

load_dotenv(Path(__file__).parent.parent.parent / "rag" / ".env")


# =============================================================================
# Few-Shot Examples (loaded from real PubMedQA examples + synthetic maybe)
# =============================================================================

FEW_SHOT_EXAMPLES_PATH = Path(__file__).parent.parent / "data" / "pubmedqa" / "few_shot_examples.json"


def load_few_shot_examples(path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Load few-shot examples from JSON file.

    Uses real examples from PubMedQA pqa_artificial split for yes/no,
    and synthetic examples for maybe (since artificial split has no maybe).

    Args:
        path: Path to few_shot_examples.json (default: data/pubmedqa/few_shot_examples.json)

    Returns:
        List of example dictionaries with context, question, answer
    """
    path = path or FEW_SHOT_EXAMPLES_PATH

    if not path.exists():
        print(f"Warning: Few-shot examples not found at {path}")
        print("Run `python src/download_few_shot.py` to download real examples.")
        # Fall back to minimal synthetic examples
        return [
            {
                "context": "The study showed treatment group had 25% better outcomes (p<0.001).",
                "question": "Does the treatment improve outcomes?",
                "answer": "yes"
            },
            {
                "context": "No significant difference was found between groups (p=0.42).",
                "question": "Does the intervention help?",
                "answer": "no"
            },
            {
                "context": "Results trended positive but did not reach significance (p=0.08).",
                "question": "Is the treatment effective?",
                "answer": "maybe"
            },
        ]

    with open(path) as f:
        data = json.load(f)

    # Flatten: get examples for each answer type
    examples = []
    for answer_type in ['yes', 'no', 'maybe']:
        for ex in data.get(answer_type, []):
            examples.append({
                'context': ex['context'],
                'question': ex['question'],
                'answer': ex['answer']
            })

    return examples


# Cache the loaded examples
_CACHED_EXAMPLES: Optional[List[Dict[str, Any]]] = None


def get_few_shot_examples() -> List[Dict[str, Any]]:
    """Get cached few-shot examples (loads on first call)."""
    global _CACHED_EXAMPLES
    if _CACHED_EXAMPLES is None:
        _CACHED_EXAMPLES = load_few_shot_examples()
    return _CACHED_EXAMPLES


# =============================================================================
# Optimized Prompts (based on SOTA analysis)
# =============================================================================

OPTIMIZED_SYSTEM_PROMPT = """You are a biomedical researcher analyzing scientific abstracts.

Your task: Determine if the research findings SUPPORT, CONTRADICT, or are INCONCLUSIVE about the question.

Decision criteria:
- YES: The abstract provides clear, statistically significant evidence supporting the question's claim
- NO: The abstract provides clear evidence contradicting or refuting the question's claim
- MAYBE: The evidence is mixed, not statistically significant, or the abstract doesn't directly address the question

Important: Base your answer ONLY on what the evidence in the abstract shows, not general medical knowledge.

Respond with exactly one word: yes, no, or maybe"""

OPTIMIZED_SYSTEM_PROMPT_COT = """You are a biomedical researcher analyzing scientific abstracts.

Your task: Determine if the research findings SUPPORT, CONTRADICT, or are INCONCLUSIVE about the question.

First, identify the key findings and their statistical significance.
Then, determine if these findings directly answer the question.
Finally, provide your answer.

Decision criteria:
- YES: Clear, statistically significant evidence supporting the claim
- NO: Clear evidence contradicting or refuting the claim
- MAYBE: Mixed results, insufficient significance, or indirect evidence

End your response with "Answer: " followed by exactly one word: yes, no, or maybe"""


def load_pubmedqa(data_path: str) -> List[Dict[str, Any]]:
    """Load PubMedQA JSONL file.

    Args:
        data_path: Path to the pubmedqa_labeled.jsonl file

    Returns:
        List of question dictionaries
    """
    with open(data_path) as f:
        return [json.loads(line) for line in f]


def build_few_shot_prompt(
    item: Dict[str, Any],
    examples: Optional[List[Dict[str, Any]]] = None
) -> str:
    """Build a few-shot prompt with examples.

    Args:
        item: The question to answer
        examples: List of few-shot examples (default: load from JSON)

    Returns:
        Formatted user prompt with examples
    """
    if examples is None:
        examples = get_few_shot_examples()

    prompt_parts = ["Here are some examples:\n"]

    for i, ex in enumerate(examples, 1):
        prompt_parts.append(f"Example {i}:")
        prompt_parts.append(f"Abstract: {ex['context']}")
        prompt_parts.append(f"Question: {ex['question']}")
        prompt_parts.append(f"Answer: {ex['answer']}\n")

    prompt_parts.append("Now answer this question:")
    prompt_parts.append(f"Abstract: {item['context']}")
    prompt_parts.append(f"Question: {item['question']}")
    prompt_parts.append("Answer:")

    return "\n".join(prompt_parts)


def evaluate_question(
    item: Dict[str, Any],
    provider,
    temperature: float = 0.0,
    use_cot: bool = False,
    use_few_shot: bool = False,
    use_optimized_prompt: bool = False,
) -> Dict[str, Any]:
    """Evaluate a single PubMedQA question.

    Args:
        item: Question dictionary with context, question, answer
        provider: LLM provider instance
        temperature: Generation temperature
        use_cot: Whether to use chain-of-thought prompting
        use_few_shot: Whether to use few-shot examples
        use_optimized_prompt: Whether to use SOTA-optimized prompts

    Returns:
        Result dictionary with prediction and correctness
    """
    # Select system prompt
    if use_cot:
        if use_optimized_prompt:
            system_prompt = OPTIMIZED_SYSTEM_PROMPT_COT
        else:
            system_prompt = """You are a medical expert evaluating biomedical research.
Based on the research abstract provided, analyze the evidence and answer the question.

First, briefly analyze the key findings in the abstract (2-3 sentences).
Then, provide your final answer on a new line starting with "Answer: " followed by exactly one word: yes, no, or maybe.

- Answer "yes" if the evidence clearly supports the claim
- Answer "no" if the evidence clearly contradicts the claim
- Answer "maybe" if the evidence is inconclusive or mixed"""
        max_tokens = 250
    else:
        if use_optimized_prompt:
            system_prompt = OPTIMIZED_SYSTEM_PROMPT
        else:
            system_prompt = """You are a medical expert evaluating biomedical research.
Based on the research abstract provided, answer the question.

You MUST answer with exactly one word: "yes", "no", or "maybe".
- Answer "yes" if the evidence clearly supports the claim
- Answer "no" if the evidence clearly contradicts the claim
- Answer "maybe" if the evidence is inconclusive or mixed

Only output the single word answer, nothing else."""
        max_tokens = 10

    # Build user prompt
    if use_few_shot:
        user_prompt = build_few_shot_prompt(item)
    else:
        user_prompt = f"""Research Abstract:
{item['context']}

Question: {item['question']}

Answer (yes/no/maybe):"""

    start_time = time.time()

    try:
        response = provider.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=max_tokens,
            temperature=temperature
        )
        generation_time = (time.time() - start_time) * 1000

        raw_response = response.content.strip().lower()

        # For CoT, extract the final answer
        if use_cot and "answer:" in raw_response:
            # Get text after "answer:"
            answer_part = raw_response.split("answer:")[-1].strip()
            predicted = answer_part.split()[0] if answer_part else raw_response
        else:
            predicted = raw_response

        # Normalize to yes/no/maybe
        if "yes" in predicted:
            predicted = "yes"
        elif "no" in predicted:
            predicted = "no"
        else:
            predicted = "maybe"

        correct = predicted == item['answer'].lower()
        error = None

    except Exception as e:
        generation_time = (time.time() - start_time) * 1000
        predicted = "error"
        correct = False
        error = str(e)

    return {
        'question_id': item['question_id'],
        'pubmed_id': item.get('pubmed_id'),
        'question': item['question'],
        'gold_answer': item['answer'],
        'predicted_answer': predicted,
        'correct': correct,
        'generation_time_ms': generation_time,
        'error': error,
    }


def evaluate_with_self_consistency(
    item: Dict[str, Any],
    provider,
    n_samples: int = 5,
    temperature: float = 0.7,
    use_cot: bool = False,
    use_few_shot: bool = False,
    use_optimized_prompt: bool = False,
) -> Dict[str, Any]:
    """Evaluate with self-consistency voting (multiple samples + majority vote).

    Args:
        item: Question dictionary
        provider: LLM provider instance
        n_samples: Number of samples to generate
        temperature: Generation temperature (should be >0 for diversity)
        use_cot: Whether to use chain-of-thought
        use_few_shot: Whether to use few-shot examples
        use_optimized_prompt: Whether to use optimized prompts

    Returns:
        Result dictionary with voted prediction
    """
    start_time = time.time()
    answers = []
    errors = []

    for _ in range(n_samples):
        result = evaluate_question(
            item, provider, temperature,
            use_cot=use_cot,
            use_few_shot=use_few_shot,
            use_optimized_prompt=use_optimized_prompt
        )
        if result['predicted_answer'] != 'error':
            answers.append(result['predicted_answer'])
        else:
            errors.append(result.get('error'))

    generation_time = (time.time() - start_time) * 1000

    if not answers:
        # All samples errored
        return {
            'question_id': item['question_id'],
            'pubmed_id': item.get('pubmed_id'),
            'question': item['question'],
            'gold_answer': item['answer'],
            'predicted_answer': 'error',
            'correct': False,
            'generation_time_ms': generation_time,
            'error': errors[0] if errors else 'All samples failed',
            'vote_distribution': {},
            'confidence': 0.0,
        }

    # Majority vote
    vote_counts = Counter(answers)
    final_answer = vote_counts.most_common(1)[0][0]
    confidence = vote_counts[final_answer] / len(answers)

    return {
        'question_id': item['question_id'],
        'pubmed_id': item.get('pubmed_id'),
        'question': item['question'],
        'gold_answer': item['answer'],
        'predicted_answer': final_answer,
        'correct': final_answer == item['answer'].lower(),
        'generation_time_ms': generation_time,
        'error': None,
        'vote_distribution': dict(vote_counts),
        'confidence': confidence,
    }


def run_evaluation(
    data: List[Dict[str, Any]],
    model_name: str,
    temperature: float = 0.0,
    use_cot: bool = False,
    use_few_shot: bool = False,
    use_optimized_prompt: bool = False,
    use_self_consistency: bool = False,
    n_samples: int = 5,
) -> Dict[str, Any]:
    """Run full evaluation on PubMedQA dataset.

    Args:
        data: List of question dictionaries
        model_name: Name of the model to use
        temperature: Generation temperature
        use_cot: Whether to use chain-of-thought prompting
        use_few_shot: Whether to use few-shot examples
        use_optimized_prompt: Whether to use SOTA-optimized prompts
        use_self_consistency: Whether to use self-consistency voting
        n_samples: Number of samples for self-consistency

    Returns:
        Dictionary with results and metrics
    """
    # Determine evaluation mode string
    mode_parts = []
    if use_few_shot:
        mode_parts.append("few-shot")
    if use_cot:
        mode_parts.append("CoT")
    if use_optimized_prompt:
        mode_parts.append("optimized-prompt")
    if use_self_consistency:
        mode_parts.append(f"self-consistency(n={n_samples})")
    mode_str = " + ".join(mode_parts) if mode_parts else "zero-shot"

    print(f"\n{'='*60}")
    print(f"PubMedQA Evaluation")
    print(f"{'='*60}")
    print(f"Model: {model_name}")
    print(f"Mode: {mode_str}")
    print(f"Questions: {len(data)}")
    print(f"Temperature: {temperature}")
    print(f"{'='*60}\n")

    # Initialize provider
    provider = get_provider(model_name)

    # Evaluate
    results = []
    correct = 0
    total_time = 0

    # Track by answer type
    by_gold_answer = {'yes': {'correct': 0, 'total': 0},
                      'no': {'correct': 0, 'total': 0},
                      'maybe': {'correct': 0, 'total': 0}}

    for item in tqdm(data, desc="Evaluating"):
        if use_self_consistency:
            # Use higher temperature for diversity
            sc_temp = max(temperature, 0.7)
            result = evaluate_with_self_consistency(
                item, provider, n_samples, sc_temp,
                use_cot=use_cot,
                use_few_shot=use_few_shot,
                use_optimized_prompt=use_optimized_prompt
            )
        else:
            result = evaluate_question(
                item, provider, temperature,
                use_cot=use_cot,
                use_few_shot=use_few_shot,
                use_optimized_prompt=use_optimized_prompt
            )
        results.append(result)

        if result['correct']:
            correct += 1
        total_time += result['generation_time_ms']

        # Track by gold answer
        gold = item['answer'].lower()
        if gold in by_gold_answer:
            by_gold_answer[gold]['total'] += 1
            if result['correct']:
                by_gold_answer[gold]['correct'] += 1

    # Calculate metrics
    accuracy = correct / len(results) if results else 0
    avg_time = total_time / len(results) if results else 0

    # Calculate per-answer-type accuracy
    for answer_type in by_gold_answer:
        stats = by_gold_answer[answer_type]
        stats['accuracy'] = stats['correct'] / stats['total'] if stats['total'] > 0 else 0

    # Print results
    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"Model: {model_name}")
    print(f"Overall Accuracy: {accuracy:.1%} ({correct}/{len(results)})")
    print(f"\nBreakdown by answer type:")
    for answer_type, stats in by_gold_answer.items():
        print(f"  {answer_type:6s}: {stats['accuracy']:.1%} ({stats['correct']}/{stats['total']})")
    print(f"\nAvg generation time: {avg_time:.0f}ms")
    print(f"{'='*60}")

    return {
        'model': model_name,
        'timestamp': datetime.now().isoformat(),
        'accuracy': accuracy,
        'correct': correct,
        'total': len(results),
        'by_answer_type': by_gold_answer,
        'avg_generation_time_ms': avg_time,
        'temperature': temperature,
        'mode': mode_str,
        'use_cot': use_cot,
        'use_few_shot': use_few_shot,
        'use_optimized_prompt': use_optimized_prompt,
        'use_self_consistency': use_self_consistency,
        'n_samples': n_samples if use_self_consistency else None,
        'results': results
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate LLMs on PubMedQA with SOTA techniques",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic zero-shot evaluation
  python evaluation.py --model gpt-4o-mini --limit 10

  # Few-shot evaluation (SOTA technique #1)
  python evaluation.py --model claude-sonnet-4-20250514 --few-shot

  # Self-consistency voting (SOTA technique #2)
  python evaluation.py --model gpt-4o --self-consistency --n-samples 5

  # Combined SOTA techniques
  python evaluation.py --model claude-sonnet-4-20250514 --few-shot --optimized-prompt

  # Full SOTA pipeline
  python evaluation.py --model gpt-4o --few-shot --self-consistency --optimized-prompt
        """
    )
    parser.add_argument(
        '--model',
        default=DEFAULTS.llm_model,
        help=f'Model to evaluate (default: {DEFAULTS.llm_model})'
    )
    parser.add_argument(
        '--data',
        default='data/pubmedqa/pubmedqa_labeled.jsonl',
        help='Path to PubMedQA data file'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of questions (for testing)'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.0,
        help='Generation temperature (default: 0.0)'
    )
    parser.add_argument(
        '--output-dir',
        default='results',
        help='Output directory for results (default: results)'
    )

    # SOTA technique flags
    parser.add_argument(
        '--cot',
        action='store_true',
        help='Use chain-of-thought prompting'
    )
    parser.add_argument(
        '--few-shot',
        action='store_true',
        help='Use few-shot prompting with curated examples (+4-5%% accuracy)'
    )
    parser.add_argument(
        '--optimized-prompt',
        action='store_true',
        help='Use SOTA-optimized system prompts'
    )
    parser.add_argument(
        '--self-consistency',
        action='store_true',
        help='Use self-consistency voting with multiple samples (+2-3%% accuracy)'
    )
    parser.add_argument(
        '--n-samples',
        type=int,
        default=5,
        help='Number of samples for self-consistency (default: 5)'
    )

    args = parser.parse_args()

    # Load data
    data = load_pubmedqa(args.data)
    print(f"Loaded {len(data)} questions from {args.data}")

    if args.limit:
        data = data[:args.limit]
        print(f"Limited to {len(data)} questions")

    # Run evaluation
    results = run_evaluation(
        data,
        args.model,
        args.temperature,
        use_cot=args.cot,
        use_few_shot=args.few_shot,
        use_optimized_prompt=args.optimized_prompt,
        use_self_consistency=args.self_consistency,
        n_samples=args.n_samples,
    )

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Build filename suffix based on techniques used
    model_clean = args.model.replace('/', '_').replace(':', '_')
    suffix_parts = []
    if args.few_shot:
        suffix_parts.append("fs")
    if args.cot:
        suffix_parts.append("cot")
    if args.optimized_prompt:
        suffix_parts.append("opt")
    if args.self_consistency:
        suffix_parts.append(f"sc{args.n_samples}")
    suffix = "_" + "_".join(suffix_parts) if suffix_parts else ""
    output_path = output_dir / f"pubmedqa_{model_clean}{suffix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
