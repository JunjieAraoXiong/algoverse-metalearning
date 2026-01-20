"""Judge Agent: Evaluates answers and decides whether to retry.

Includes a deterministic verification gate that runs BEFORE LLM-as-judge
to catch answers lacking proper evidence citations.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document

from .base import AgentDecision, BaseAgent
from evaluation.deterministic_verify import (
    VerificationResult,
    deterministic_verify,
    format_verification_feedback,
)


class JudgeAgent(BaseAgent):
    """Agent C: Evaluates answer quality and decides whether to retry.

    This agent:
    1. Compares the generated answer to the gold answer (when available)
    2. Evaluates answer quality using LLM-as-judge
    3. Decides whether to trigger a retry based on score threshold
    4. Can adjust threshold on different attempts

    The agent provides interpretable decisions about:
    - Whether the answer is correct
    - Why the answer succeeded or failed
    - Whether a retry is warranted
    """

    def __init__(
        self,
        judge_model: str = None,
        retry_threshold: float = 0.5,
        min_threshold: float = 0.3,
        enable_deterministic_gate: bool = True,
        require_all_numbers_cited: bool = True,
    ):
        """Initialize the judge agent.

        Args:
            judge_model: Model to use for LLM-as-judge evaluation
            retry_threshold: Score below which to trigger retry (default 0.5)
            min_threshold: Minimum threshold even after escalation (default 0.3)
            enable_deterministic_gate: Run deterministic verification before LLM judge
            require_all_numbers_cited: Require all numerical claims to have citations
        """
        super().__init__("JudgeAgent")

        from src.config import DEFAULTS
        self.judge_model = judge_model or DEFAULTS.judge_model
        self.retry_threshold = retry_threshold
        self.min_threshold = min_threshold
        self.enable_deterministic_gate = enable_deterministic_gate
        self.require_all_numbers_cited = require_all_numbers_cited

        # Track scores across attempts
        self._attempt_scores: list = []
        # Track verification results for retry feedback
        self._last_verification: Optional[VerificationResult] = None

    def evaluate(
        self,
        question: str,
        predicted_answer: str,
        gold_answer: str = None,
    ) -> Tuple[float, str]:
        """Evaluate the predicted answer.

        Args:
            question: The original question
            predicted_answer: The model's answer
            gold_answer: The reference answer (if available)

        Returns:
            Tuple of (score, justification)
        """
        if not predicted_answer:
            return 0.0, "Empty answer"

        if gold_answer:
            # Use LLM-as-judge with gold answer
            from evaluation.llm_judge import llm_as_judge
            score, justification = llm_as_judge(
                question=question,
                gold_answer=gold_answer,
                predicted_answer=predicted_answer,
                judge_model=self.judge_model,
            )
        else:
            # Self-evaluation without gold answer (check for coherence, specificity)
            score, justification = self._self_evaluate(question, predicted_answer)

        return score, justification

    def _self_evaluate(self, question: str, answer: str) -> Tuple[float, str]:
        """Self-evaluate answer quality without gold answer.

        Args:
            question: The question
            answer: The generated answer

        Returns:
            Tuple of (score, justification)
        """
        from src.providers import get_provider

        system_prompt = """You are evaluating the quality of an answer to a question.
You do NOT have access to the correct answer, so evaluate based on:
1. Coherence: Is the answer clear and well-structured?
2. Specificity: Does it provide concrete information (numbers, names, dates)?
3. Relevance: Does it actually address what the question asks?
4. Confidence: Does it avoid excessive hedging or refusals?

Score from 0.0 to 1.0:
- 1.0: Excellent - specific, coherent, directly addresses the question
- 0.7: Good - mostly specific with minor issues
- 0.5: Acceptable - somewhat vague but on topic
- 0.3: Poor - vague, hedging, or partially off-topic
- 0.0: Failure - refused to answer or completely off-topic

Respond with:
SCORE: <score>
JUSTIFICATION: <explanation>"""

        user_prompt = f"""Question: {question}

Answer: {answer}

Evaluate the quality of this answer."""

        try:
            provider = get_provider(self.judge_model)
            response = provider.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=200,
                temperature=0.0,
            )

            # Parse response
            from evaluation.llm_judge import parse_judge_response
            return parse_judge_response(response.content)
        except Exception as e:
            return 0.0, f"Self-evaluation failed: {str(e)}"

    def run_deterministic_verification(
        self,
        answer: str,
        docs: List[Document],
    ) -> VerificationResult:
        """Run deterministic verification as a hard gate before LLM judge.

        This checks that all numerical claims have proper [DocX: 'quote'] citations.

        Args:
            answer: The generated answer
            docs: Source documents used for the answer

        Returns:
            VerificationResult with pass/fail and details
        """
        result = deterministic_verify(
            answer=answer,
            docs=docs,
            require_all_numbers_cited=self.require_all_numbers_cited,
        )
        self._last_verification = result
        return result

    def get_verification_feedback(self) -> str:
        """Get feedback from last verification for retry prompt.

        Returns:
            Formatted feedback string or empty if passed
        """
        if self._last_verification is None:
            return ""
        return format_verification_feedback(self._last_verification)

    def should_retry(self, score: float, attempt: int, max_retries: int) -> bool:
        """Decide whether to retry based on score and attempt number.

        Args:
            score: The evaluation score
            attempt: Current attempt number
            max_retries: Maximum allowed retries

        Returns:
            True if should retry, False otherwise
        """
        # Don't retry if we've reached max attempts
        if attempt >= max_retries:
            return False

        # Adjust threshold based on attempt (lower threshold on later attempts)
        # This avoids infinite loops on consistently low-scoring questions
        adjusted_threshold = max(
            self.min_threshold,
            self.retry_threshold - (attempt * 0.1)
        )

        # Check if score is below threshold
        if score < adjusted_threshold:
            # Additional check: if score is improving, maybe don't retry
            if len(self._attempt_scores) > 0:
                if score >= self._attempt_scores[-1] + 0.2:
                    # Significant improvement, accept even if below threshold
                    return False
            return True

        return False

    def decide(self, context: Dict[str, Any]) -> AgentDecision:
        """Evaluate the answer and decide whether to retry.

        Includes a deterministic verification gate that runs BEFORE LLM judge.
        If verification fails, triggers retry without calling the LLM judge.
        If max retries reached and still failing verification, returns abstain.

        Args:
            context: Must contain 'question', 'predicted_answer'.
                     Optional: 'gold_answer', 'attempt', 'max_retries', 'documents'

        Returns:
            AgentDecision with evaluation and retry decision
        """
        question = context["question"]
        predicted_answer = context.get("predicted_answer", "")
        gold_answer = context.get("gold_answer")
        attempt = context.get("attempt", self._attempt)
        max_retries = context.get("max_retries", 1)
        docs = context.get("documents", [])

        # DETERMINISTIC GATE: Run verification before LLM judge
        verification_passed = True
        verification_message = ""

        if self.enable_deterministic_gate and docs:
            verification_result = self.run_deterministic_verification(
                predicted_answer, docs
            )
            verification_passed = verification_result.passed
            verification_message = verification_result.message

            if not verification_passed:
                # Deterministic check failed - decide based on retry budget
                if attempt >= max_retries:
                    # Max retries reached, ABSTAIN
                    return self._create_abstain_decision(
                        attempt=attempt,
                        reason=f"Deterministic verification failed after {attempt + 1} attempts: {verification_message}",
                    )
                else:
                    # Trigger retry without calling LLM judge
                    return self._create_retry_decision(
                        attempt=attempt,
                        reason=f"Deterministic verification failed: {verification_message}",
                        verification_feedback=self.get_verification_feedback(),
                    )

        # Deterministic check passed - proceed to LLM-as-judge evaluation
        score, justification = self.evaluate(question, predicted_answer, gold_answer)

        # Track score for this attempt
        self._attempt_scores.append(score)

        # Decide whether to retry
        should_retry = self.should_retry(score, attempt, max_retries)

        # Determine pass/fail
        passed = score >= 0.5

        # Build reasoning
        if should_retry:
            reasoning = (
                f"Score {score:.2f} below threshold. "
                f"Triggering retry. {justification}"
            )
        elif passed:
            reasoning = f"Answer accepted with score {score:.2f}. {justification}"
        else:
            reasoning = (
                f"Answer failed with score {score:.2f}, "
                f"but max retries reached. {justification}"
            )

        decision = AgentDecision(
            agent_name=self.name,
            decision_type="evaluation",
            decision_value={
                "score": score,
                "pass": passed,
                "retry": should_retry,
                "abstain": False,
                "justification": justification,
                "verification_passed": verification_passed,
            },
            confidence=score,  # Use score as confidence
            reasoning=reasoning,
            metadata={
                "attempt": attempt,
                "threshold": self.retry_threshold,
                "has_gold_answer": gold_answer is not None,
                "attempt_scores": self._attempt_scores.copy(),
                "verification_message": verification_message,
            }
        )

        self.log_decision(decision)
        return decision

    def _create_retry_decision(
        self,
        attempt: int,
        reason: str,
        verification_feedback: str = "",
    ) -> AgentDecision:
        """Create a decision to retry due to verification failure.

        Args:
            attempt: Current attempt number
            reason: Reason for retry
            verification_feedback: Feedback to include in retry prompt

        Returns:
            AgentDecision indicating retry needed
        """
        decision = AgentDecision(
            agent_name=self.name,
            decision_type="evaluation",
            decision_value={
                "score": 0.0,
                "pass": False,
                "retry": True,
                "abstain": False,
                "justification": reason,
                "verification_passed": False,
                "verification_feedback": verification_feedback,
            },
            confidence=0.0,
            reasoning=f"Retry triggered: {reason}",
            metadata={
                "attempt": attempt,
                "deterministic_gate_triggered": True,
            }
        )
        self.log_decision(decision)
        return decision

    def _create_abstain_decision(
        self,
        attempt: int,
        reason: str,
    ) -> AgentDecision:
        """Create a decision to abstain due to insufficient evidence.

        This is returned when max retries are reached but the deterministic
        verification still fails, indicating the answer cannot be grounded
        in the available documents.

        Args:
            attempt: Current attempt number
            reason: Reason for abstention

        Returns:
            AgentDecision indicating abstention
        """
        decision = AgentDecision(
            agent_name=self.name,
            decision_type="evaluation",
            decision_value={
                "score": 0.0,
                "pass": False,
                "retry": False,
                "abstain": True,
                "justification": "Insufficient evidence in retrieved corpus",
                "verification_passed": False,
            },
            confidence=0.0,
            reasoning=f"ABSTAIN: {reason}",
            metadata={
                "attempt": attempt,
                "abstain_reason": reason,
                "deterministic_gate_triggered": True,
            }
        )
        self.log_decision(decision)
        return decision

    def reset(self) -> None:
        """Reset agent state for a new question."""
        super().reset()
        self._attempt_scores = []
        self._last_verification = None
