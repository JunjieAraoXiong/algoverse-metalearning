"""Bulk testing framework for RAG system evaluation."""

import sys
import time
import json
import argparse
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import retrieval components
from langchain_chroma import Chroma

# Import custom modules
from dataset_adapters import BaseDatasetAdapter, FinanceBenchAdapter, PubMedQAAdapter
from evaluation.metrics import (
    embedding_similarity,
    calculate_aggregate_metrics,
    format_metrics_summary
)
from evaluation.llm_judge import llm_as_judge
from src.postprocessing.numeric_verify import verify_numeric_answer
from src.retrieval_tools.tool_registry import (
    build_pipeline,
    build_retriever_for_pipeline,
)
from src.retrieval_tools.router import build_routed_pipeline
from src.config import (
    DEFAULTS,
    PIPELINES,
    EMBEDDINGS,
    get_model_abbrev,
    get_provider_for_model,
    get_embedding_model,
)
from src.providers import get_provider

# Load environment variables
load_dotenv()


@dataclass
class BulkTestConfig:
    """Configuration for bulk testing runs."""

    # Dataset settings
    dataset_name: str

    # Retrieval policy
    pipeline_id: str = DEFAULTS.pipeline_id

    # Model settings
    model_name: str = DEFAULTS.llm_model
    embedding_model: str = DEFAULTS.embedding_model

    # Retrieval settings
    top_k_retrieval: int = DEFAULTS.top_k
    initial_k_factor: float = DEFAULTS.initial_k_factor

    # Reranker settings
    reranker_model: str = DEFAULTS.reranker_model

    # Generation settings
    temperature: float = DEFAULTS.temperature
    max_tokens: int = DEFAULTS.max_tokens

    # Evaluation settings
    use_llm_judge: bool = False
    judge_model: str = DEFAULTS.judge_model
    use_numeric_verify: bool = False

    # Paths
    chroma_path: str = DEFAULTS.chroma_path
    output_dir: str = DEFAULTS.output_dir

    # Router settings (for pipeline_id="routed")
    router_classifier_model: str = DEFAULTS.router_classifier_model
    router_hyde_model: str = DEFAULTS.router_hyde_model
    use_rule_router: bool = False  # Use free rule-based router instead of LLM

    # Runtime metadata
    timestamp: str = None

    def __post_init__(self):
        """Generate timestamp if not provided and resolve paths relative to project root."""
        if self.timestamp is None:
            self.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        if not self.pipeline_id:
            self.pipeline_id = DEFAULTS.pipeline_id

        # Resolve paths relative to project root
        base_dir = Path(__file__).parent.parent
        self.chroma_path = str(base_dir / self.chroma_path)
        self.output_dir = str(base_dir / self.output_dir)

    def get_model_abbrev(self) -> str:
        """Get abbreviated model name for filename."""
        return get_model_abbrev(self.model_name)

    def generate_filename(self, dataset_abbrev: str) -> str:
        """Generate output filename from configuration."""
        model_abbrev = self.get_model_abbrev()
        temp_str = f"t{self.temperature}".replace(".", "")
        return f"{self.timestamp}_{dataset_abbrev}_{model_abbrev}_k{self.top_k_retrieval}_{temp_str}.csv"


class BulkTestRunner:
    """Main bulk testing runner."""

    def __init__(self, config: BulkTestConfig):
        self.config = config
        self.retriever = None
        self.embeddings = None
        self.llm_provider = None
        self.pipeline = None

    def initialize_framework(self):
        """Initialize RAG framework components."""
        try:
            print("\nInitializing RAG framework...")

            # Initialize embeddings (uses FREE local model by default)
            emb_config = EMBEDDINGS.get(self.config.embedding_model)
            if emb_config:
                print(f"  Loading embeddings: {emb_config.model_id} ({emb_config.provider})")
            else:
                print(f"  Loading embeddings: {self.config.embedding_model}")
            self.embeddings = get_embedding_model(self.config.embedding_model)

            # Load ChromaDB
            print(f"  Loading ChromaDB from: {self.config.chroma_path}")
            db = Chroma(
                persist_directory=self.config.chroma_path,
                embedding_function=self.embeddings
            )

            print(f"  Creating retriever for pipeline: {self.config.pipeline_id}")

            if self.config.pipeline_id == "routed":
                # Use the question-type router for adaptive retrieval
                self.pipeline = build_routed_pipeline(
                    db=db,
                    embedding_fn=self.embeddings,
                    classifier_model=self.config.router_classifier_model,
                    hyde_model=self.config.router_hyde_model,
                    reranker_model=self.config.reranker_model,
                    use_rule_router=self.config.use_rule_router,
                )
            else:
                # Standard pipeline
                retriever, set_k_fn, take_top_k_fn = build_retriever_for_pipeline(
                    self.config.pipeline_id, db, top_k=self.config.top_k_retrieval
                )
                self.retriever = retriever
                self.pipeline = build_pipeline(
                    pipeline_id=self.config.pipeline_id,
                    retriever=retriever,
                    top_k=self.config.top_k_retrieval,
                    initial_k_factor=self.config.initial_k_factor,
                    set_k_fn=set_k_fn,
                    take_top_k_fn=take_top_k_fn,
                    reranker_model=self.config.reranker_model,
                )

            # Initialize LLM provider
            provider_name = get_provider_for_model(self.config.model_name)
            print(f"  Initializing LLM: {self.config.model_name} (provider: {provider_name})")
            self.llm_provider = get_provider(self.config.model_name)

            print("Framework initialization complete!\n")
            return True

        except Exception as e:
            print(f"ERROR: Framework initialization failed: {str(e)}")
            return False

    def process_single_question(self, question: str, question_id: Any) -> Dict[str, Any]:
        """Process a single question through the RAG pipeline."""
        result = {
            'predicted_answer': None,
            'sources': None,
            'retrieval_time_ms': 0,
            'generation_time_ms': 0,
            'error': None
        }

        try:
            # Retrieval phase
            retrieval_start = time.time()
            docs = self.pipeline.retrieve(question) if self.pipeline else []
            result['retrieval_time_ms'] = (time.time() - retrieval_start) * 1000

            if not docs:
                result['error'] = "No relevant documents found"
                return result

            # Extract context and sources
            context = "\n\n".join(d.page_content for d in docs)
            sources = [doc.metadata for doc in docs]
            result['sources'] = sources

            # Generation phase
            generation_start = time.time()

            system_prompt = (
                "You are a precise financial analysis assistant who approaches every question methodically. "
                "ALWAYS enter PLAN MODE before answering: first analyze what information is needed, "
                "identify relevant data points in the context, then formulate your answer. "
                "Be accurate with numbers, dates, and company names. "
                "ALWAYS provide your best answer based on the available context - "
                "never refuse to answer or say you cannot find the information."
            )

            user_prompt = f"""Answer the following question using the information from the provided context.

PLAN MODE REQUIRED - Before answering, you MUST:
1. IDENTIFY: What specific information does this question ask for? (number, explanation, comparison, etc.)
2. LOCATE: Find the relevant data points, figures, or facts in the context
3. VERIFY: Check that the data matches the correct company, time period, and fiscal year
4. CALCULATE: If math is needed, show your work step-by-step
5. ANSWER: Only then provide your final answer

IMPORTANT INSTRUCTIONS:
- ALWAYS provide an answer - even if the context seems incomplete, give your best hypothesis based on available information
- Use precise numbers, dates, and company names from the context when available
- Do NOT use information from other companies or fiscal years unless explicitly asked
- Pay close attention to fiscal years and time periods mentioned in both the question and context
- For numerical questions requiring a specific number, percentage, or ratio as the answer:
  * After your planning steps, provide ONLY the numerical value with appropriate units
  * Format examples: "$1,577 million" or "65.4%" or "24.26"
  * Do NOT add explanatory sentences like "The answer is..." or "According to the context..."
- For non-numerical or explanatory questions, provide full context and reasoning
- NEVER say "The provided context does not contain sufficient information" - always attempt an answer

Context:
{context}

Question: {question}

Plan and Answer:"""

            # Use the provider abstraction
            response = self.llm_provider.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
            )

            result['generation_time_ms'] = (time.time() - generation_start) * 1000

            if response.content:
                result['predicted_answer'] = response.content

                # Numeric verification - check if numbers in answer exist in sources
                verification = verify_numeric_answer(response.content, docs)
                result['numeric_score'] = verification.score
                result['flagged_numbers'] = verification.flagged_numbers
            else:
                result['error'] = "Empty response from LLM"

        except Exception as e:
            result['error'] = f"Error: {str(e)}"

        return result

    def run_bulk_test(self, adapter: BaseDatasetAdapter) -> pd.DataFrame:
        """Run bulk test on a dataset."""
        print("\n" + "=" * 60)
        print("STARTING BULK TEST")
        print("=" * 60)

        # Load dataset
        try:
            df = adapter.load_dataset()
        except Exception as e:
            print(f"ERROR: Failed to load dataset: {str(e)}")
            sys.exit(1)

        # Get column names
        question_col = adapter.get_question_column()
        answer_col = adapter.get_answer_column()
        question_type_col = adapter.get_question_type_column()
        metadata_cols = adapter.get_metadata_columns()

        print(f"Dataset loaded: {len(df)} questions")
        print(f"Question column: {question_col}")
        print(f"Answer column: {answer_col}")

        # Initialize framework
        if not self.initialize_framework():
            print("ERROR: Framework initialization failed. Exiting.")
            sys.exit(1)

        # Prepare results storage
        results = []

        # Process questions with progress bar
        print("\nProcessing questions...")
        start_time = time.time()

        try:
            for idx, row in tqdm(df.iterrows(), total=len(df), desc="Questions"):
                question = row[question_col]
                gold_answer = row[answer_col]

                # Process question
                result = self.process_single_question(question, idx)

                # Calculate semantic similarity
                if result['predicted_answer'] is not None:
                    sem_sim = embedding_similarity(
                        result['predicted_answer'],
                        gold_answer,
                        self.embeddings
                    )
                else:
                    sem_sim = 0.0

                # LLM judge if enabled
                judge_score = None
                judge_justification = None
                if self.config.use_llm_judge and result['predicted_answer'] is not None:
                    judge_score, judge_justification = llm_as_judge(
                        question=question,
                        gold_answer=gold_answer,
                        predicted_answer=result['predicted_answer'],
                        judge_model=self.config.judge_model
                    )

                # Format sources
                sources_str = None
                if result['sources']:
                    source_names = [s.get('source', 'unknown') for s in result['sources']]
                    sources_str = "; ".join(source_names)

                # Build result row
                result_row = {
                    'question_id': idx,
                    'question': question,
                    'gold_answer': gold_answer,
                    'predicted_answer': result['predicted_answer'],
                    'semantic_similarity': sem_sim,
                    'numeric_score': result.get('numeric_score'),
                    'flagged_numbers': str(result.get('flagged_numbers', [])),
                    'retrieval_time_ms': result['retrieval_time_ms'],
                    'generation_time_ms': result['generation_time_ms'],
                    'sources': sources_str,
                    'error': result['error']
                }

                if self.config.use_llm_judge:
                    result_row['judge_score'] = judge_score
                    result_row['judge_justification'] = judge_justification

                if question_type_col and question_type_col in row:
                    result_row['question_type'] = row[question_type_col]

                for col in metadata_cols:
                    if col in row:
                        result_row[col] = row[col]

                results.append(result_row)

        except KeyboardInterrupt:
            print(f"\n\nInterrupted! Saving {len(results)} partial results...")
            if results:
                results_df = pd.DataFrame(results)
                self._save_results(results_df, adapter, partial=True)
            sys.exit(0)

        print(f"\nProcessing complete! Total time: {time.time() - start_time:.2f}s")
        return pd.DataFrame(results)

    def _save_results(self, results_df: pd.DataFrame, adapter: BaseDatasetAdapter, partial: bool = False):
        """Save results to CSV and summary to JSON."""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(exist_ok=True)

        filename = self.config.generate_filename(adapter.name)
        if partial:
            filename = filename.replace('.csv', '_PARTIAL.csv')

        output_path = output_dir / filename

        # Save CSV
        results_df.to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path}")

        # Calculate and save metrics
        metrics = calculate_aggregate_metrics(results_df)
        metrics['config'] = asdict(self.config)

        summary_path = output_path.with_suffix('.json')
        with open(summary_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        print(f"Summary saved to: {summary_path}")
        print(format_metrics_summary(metrics))

    def save_results(self, results_df: pd.DataFrame, adapter: BaseDatasetAdapter):
        """Public method to save results."""
        self._save_results(results_df, adapter, partial=False)


def main():
    """Main entry point for bulk testing."""
    parser = argparse.ArgumentParser(description="Run bulk testing on RAG framework")

    parser.add_argument(
        '--dataset', type=str, default='financebench',
        help='Dataset to test on (default: financebench)'
    )
    parser.add_argument(
        '--pipeline', type=str, default=DEFAULTS.pipeline_id, choices=PIPELINES,
        help=f'Retrieval pipeline (default: {DEFAULTS.pipeline_id})'
    )
    parser.add_argument(
        '--model', type=str, default=DEFAULTS.llm_model,
        help=f'LLM model (default: {DEFAULTS.llm_model})'
    )
    parser.add_argument(
        '--top-k', type=int, default=DEFAULTS.top_k,
        help=f'Documents to retrieve (default: {DEFAULTS.top_k})'
    )
    parser.add_argument(
        '--initial-k-factor', type=float, default=DEFAULTS.initial_k_factor,
        help=f'Initial retrieval multiplier (default: {DEFAULTS.initial_k_factor})'
    )
    parser.add_argument(
        '--reranker', type=str, default=DEFAULTS.reranker_model,
        help=f'Reranker model (default: {DEFAULTS.reranker_model})'
    )
    parser.add_argument(
        '--temperature', type=float, default=DEFAULTS.temperature,
        help=f'Generation temperature (default: {DEFAULTS.temperature})'
    )
    parser.add_argument(
        '--max-tokens', type=int, default=DEFAULTS.max_tokens,
        help=f'Max tokens (default: {DEFAULTS.max_tokens})'
    )
    parser.add_argument(
        '--subset', type=str, default=None,
        help='Path to subset questions CSV'
    )
    parser.add_argument(
        '--use-llm-judge', action='store_true',
        help='Enable LLM-as-a-Judge evaluation'
    )
    parser.add_argument(
        '--use-numeric-verify', action='store_true',
        help='Enable numeric verification to detect hallucinated numbers'
    )
    parser.add_argument(
        '--judge-model', type=str, default=DEFAULTS.judge_model,
        help=f'Judge model (default: {DEFAULTS.judge_model})'
    )
    parser.add_argument(
        '--embedding', type=str, default=DEFAULTS.embedding_model,
        help=f'Embedding model (default: {DEFAULTS.embedding_model}). Use "openai-large" for ChromaDB built with OpenAI embeddings.'
    )
    parser.add_argument(
        '--chroma-path', type=str, default=None,
        help='Path to ChromaDB directory (overrides default dataset-based path)'
    )
    parser.add_argument(
        '--router-classifier-model', type=str, default=DEFAULTS.router_classifier_model,
        help=f'Model for question classification in routed pipeline (default: {DEFAULTS.router_classifier_model})'
    )
    parser.add_argument(
        '--router-hyde-model', type=str, default=DEFAULTS.router_hyde_model,
        help=f'Model for HyDE generation in routed pipeline (default: {DEFAULTS.router_hyde_model})'
    )
    parser.add_argument(
        '--use-rule-router', action='store_true',
        help='Use free rule-based router instead of LLM classifier (instant, no API cost)'
    )

    args = parser.parse_args()

    # Create configuration
    config = BulkTestConfig(
        dataset_name=args.dataset,
        pipeline_id=args.pipeline,
        model_name=args.model,
        embedding_model=args.embedding,
        top_k_retrieval=args.top_k,
        initial_k_factor=args.initial_k_factor,
        reranker_model=args.reranker,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        use_llm_judge=args.use_llm_judge,
        judge_model=args.judge_model,
        router_classifier_model=args.router_classifier_model,
        router_hyde_model=args.router_hyde_model,
        use_rule_router=args.use_rule_router,
    )

    # Handle chroma path - explicit override takes precedence
    if args.chroma_path:
        config.chroma_path = str(Path(__file__).parent.parent / args.chroma_path)
    else:
        # Auto-adjust chroma path for known datasets
        ds = args.dataset.lower()
        if config.chroma_path.endswith(DEFAULTS.chroma_path):
            if ds == 'pubmedqa':
                config.chroma_path = str(Path(__file__).parent.parent / "chroma_pubmedqa")
            elif ds == 'financebench':
                config.chroma_path = str(Path(__file__).parent.parent / "chroma")

    # Select dataset adapter
    ds = args.dataset.lower()
    if ds == 'financebench':
        adapter = FinanceBenchAdapter(subset_csv=args.subset)
    elif ds == 'pubmedqa':
        adapter = PubMedQAAdapter(subset_csv=args.subset)
    else:
        print(f"ERROR: Unknown dataset '{args.dataset}'")
        print("Available datasets: financebench, pubmedqa")
        sys.exit(1)

    # Create runner and execute
    runner = BulkTestRunner(config)

    try:
        results_df = runner.run_bulk_test(adapter)
        runner.save_results(results_df, adapter)
    except Exception as e:
        print(f"\nFATAL ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
