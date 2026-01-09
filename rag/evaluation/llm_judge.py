"""LLM-as-a-Judge evaluation for RAG system."""

import sys
from pathlib import Path
from typing import Tuple, Optional
import re

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.providers import get_provider


JUDGE_SYSTEM_PROMPT = """You are an expert evaluator for question-answering systems. Your task is to judge whether a predicted answer is correct compared to a gold/reference answer.

Guidelines:
1. Focus on factual correctness, not exact wording
2. For numerical answers, allow reasonable formatting differences (e.g., "$1,577" vs "1577 million" vs "$1.577 billion")
3. For percentage/ratio questions, ensure the numbers are equivalent
4. Partial credit is acceptable - give 0.5 for partially correct answers
5. Consider that the predicted answer may contain additional context that doesn't make it wrong

Scoring:
- 1.0: Fully correct answer
- 0.5: Partially correct (main fact correct but missing details, or close but not exact)
- 0.0: Incorrect answer

You MUST respond in the following format:
SCORE: <score>
JUSTIFICATION: <brief explanation>"""


JUDGE_USER_PROMPT_TEMPLATE = """Question: {question}

Gold Answer: {gold_answer}

Predicted Answer: {predicted_answer}

Evaluate the predicted answer against the gold answer and provide your score and justification."""


def parse_judge_response(response: str) -> Tuple[float, str]:
    """Parse the judge response to extract score and justification.

    Args:
        response: Raw response from the judge LLM

    Returns:
        Tuple of (score, justification)
    """
    score = 0.0
    justification = response

    # Try to extract score
    score_patterns = [
        r"SCORE:\s*([0-9.]+)",
        r"Score:\s*([0-9.]+)",
        r"score:\s*([0-9.]+)",
    ]

    for pattern in score_patterns:
        match = re.search(pattern, response)
        if match:
            try:
                score = float(match.group(1))
                score = max(0.0, min(1.0, score))  # Clamp to [0, 1]
                break
            except ValueError:
                continue

    # Try to extract justification
    justification_patterns = [
        r"JUSTIFICATION:\s*(.+?)(?:\n|$)",
        r"Justification:\s*(.+?)(?:\n|$)",
        r"justification:\s*(.+?)(?:\n|$)",
    ]

    for pattern in justification_patterns:
        match = re.search(pattern, response, re.DOTALL)
        if match:
            justification = match.group(1).strip()
            break

    return score, justification


def llm_as_judge(
    question: str,
    gold_answer: str,
    predicted_answer: str,
    judge_model: str = "claude-sonnet-4-5-20250514",
) -> Tuple[float, str]:
    """Use an LLM to judge whether the predicted answer is correct.

    Args:
        question: The original question
        gold_answer: The gold/reference answer
        predicted_answer: The predicted answer to evaluate
        judge_model: Model to use for judging (default: Claude Sonnet 4.5)

    Returns:
        Tuple of (score between 0-1, justification string)
    """
    if not predicted_answer or not gold_answer:
        return 0.0, "Empty answer"

    try:
        # Get provider for the judge model
        provider = get_provider(judge_model)

        # Format the prompt
        user_prompt = JUDGE_USER_PROMPT_TEMPLATE.format(
            question=question,
            gold_answer=gold_answer,
            predicted_answer=predicted_answer,
        )

        # Generate judge response
        response = provider.generate(
            system_prompt=JUDGE_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            max_tokens=256,
            temperature=0.0,
        )

        # Parse the response
        score, justification = parse_judge_response(response.content)

        return score, justification

    except Exception as e:
        error_msg = f"Judge error: {str(e)}"
        print(f"Warning: {error_msg}")
        return 0.0, error_msg


def llm_as_judge_batch(
    questions: list,
    gold_answers: list,
    predicted_answers: list,
    judge_model: str = "claude-sonnet-4-5-20250514",
) -> list:
    """Evaluate multiple predictions at once.

    Args:
        questions: List of questions
        gold_answers: List of gold answers
        predicted_answers: List of predicted answers
        judge_model: Model to use for judging

    Returns:
        List of (score, justification) tuples
    """
    results = []
    for q, gold, pred in zip(questions, gold_answers, predicted_answers):
        score, justification = llm_as_judge(q, gold, pred, judge_model)
        results.append((score, justification))
    return results
