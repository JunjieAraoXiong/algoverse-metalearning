"""FinQA dataset adapter for numerical reasoning over financial tables."""

import json
from pathlib import Path
from typing import List, Optional
import pandas as pd

from dataset_adapters.base import BaseDatasetAdapter


class FinQAAdapter(BaseDatasetAdapter):
    """Adapter for the FinQA financial question-answering dataset.

    FinQA is a benchmark for numerical reasoning over financial data,
    containing questions that require extracting and computing values
    from financial tables and text.

    Dataset format (from HuggingFace: financialdatasets/finqa):
    - id: Unique identifier
    - question: The question text
    - answer: Gold answer (text form)
    - exe_ans: Executed numerical answer
    - table: 2D list representing the financial table
    - pre_text: Text appearing before the table
    - post_text: Text appearing after the table
    - program: DSL program representing the reasoning steps

    The adapter can load from:
    1. HuggingFace datasets (default)
    2. Local JSONL files (if provided)
    """

    # Default path for local JSONL (optional)
    DEFAULT_QUESTIONS_PATH = "data/finqa/finqa_questions.jsonl"

    def __init__(
        self,
        questions_path: Optional[str] = None,
        subset_csv: Optional[str] = None,
        split: str = "dev",
        use_huggingface: bool = True,
    ):
        """Initialize the FinQA adapter.

        Args:
            questions_path: Path to local JSONL file. If None and use_huggingface=True,
                           loads from HuggingFace datasets.
            subset_csv: Optional path to CSV with subset of question IDs.
            split: Dataset split to use ("train", "dev", "test"). Default: "dev"
            use_huggingface: If True, load from HuggingFace. If False, use local file.
        """
        super().__init__(subset_csv=subset_csv)

        self.split = split
        self.use_huggingface = use_huggingface

        # Resolve paths relative to project root
        project_root = Path(__file__).parent.parent
        self.questions_path = questions_path or str(project_root / self.DEFAULT_QUESTIONS_PATH)

    def load_dataset(self) -> pd.DataFrame:
        """Load the FinQA dataset.

        Returns:
            DataFrame with questions and answers
        """
        if self._df is not None:
            return self._df

        if self.use_huggingface:
            df = self._load_from_huggingface()
        else:
            df = self._load_from_local()

        # Apply subset filter if specified
        df = self._apply_subset_filter(df)

        self._df = df
        return df

    def _load_from_huggingface(self) -> pd.DataFrame:
        """Load FinQA from HuggingFace datasets."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "HuggingFace datasets library required. Install with: pip install datasets"
            )

        print(f"Loading FinQA from HuggingFace (split: {self.split})...")

        # Load the dataset
        # FinQA is available at: https://huggingface.co/datasets/financialdatasets/finqa
        # Alternatively try: ibm/finqa or dreamerdeo/finqa
        try:
            dataset = load_dataset("financialdatasets/finqa", split=self.split)
        except Exception as e:
            print(f"Failed to load from financialdatasets/finqa: {e}")
            print("Trying alternative source: dreamerdeo/finqa...")
            try:
                dataset = load_dataset("dreamerdeo/finqa", split=self.split)
            except Exception as e2:
                print(f"Failed to load from dreamerdeo/finqa: {e2}")
                print("Trying alternative source: ibm/finqa...")
                dataset = load_dataset("ibm/finqa", split=self.split)

        # Convert to DataFrame
        records = []
        for idx, item in enumerate(dataset):
            record = {
                "question_id": item.get("id", f"finqa_{idx}"),
                "question": item["question"],
                "answer": str(item.get("answer", item.get("exe_ans", ""))),
                "exe_ans": item.get("exe_ans"),
                "program": item.get("program", ""),
            }

            # Store table as JSON string for metadata
            if "table" in item:
                record["table"] = json.dumps(item["table"])

            # Combine pre_text, table, and post_text as context
            context_parts = []
            if item.get("pre_text"):
                if isinstance(item["pre_text"], list):
                    context_parts.extend(item["pre_text"])
                else:
                    context_parts.append(item["pre_text"])

            if item.get("post_text"):
                if isinstance(item["post_text"], list):
                    context_parts.extend(item["post_text"])
                else:
                    context_parts.append(item["post_text"])

            record["context"] = " ".join(context_parts) if context_parts else ""

            records.append(record)

        print(f"Loaded {len(records)} questions from FinQA {self.split} split")
        return pd.DataFrame(records)

    def _load_from_local(self) -> pd.DataFrame:
        """Load FinQA from local JSONL file."""
        if not Path(self.questions_path).exists():
            raise FileNotFoundError(
                f"Local FinQA file not found at {self.questions_path}. "
                "Either provide the file or use use_huggingface=True."
            )

        records = []
        with open(self.questions_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    item = json.loads(line)
                    record = {
                        "question_id": item.get("id", item.get("question_id", "")),
                        "question": item["question"],
                        "answer": str(item.get("answer", item.get("exe_ans", ""))),
                        "exe_ans": item.get("exe_ans"),
                        "program": item.get("program", ""),
                        "table": json.dumps(item.get("table", [])),
                        "context": item.get("context", ""),
                    }
                    records.append(record)

        return pd.DataFrame(records)

    def get_question_column(self) -> str:
        """Return the question column name."""
        return "question"

    def get_answer_column(self) -> str:
        """Return the answer column name."""
        return "answer"

    def get_question_type_column(self) -> Optional[str]:
        """Return the question type column name.

        FinQA doesn't have explicit question types, but we could infer them
        from the program DSL (e.g., "add", "subtract", "divide", "greater").
        """
        return None

    def get_metadata_columns(self) -> List[str]:
        """Return additional metadata columns."""
        return [
            "question_id",
            "exe_ans",
            "program",
            "table",
            "context",
        ]

    @property
    def name(self) -> str:
        """Return the dataset name."""
        return "finqa"

    def get_table_for_question(self, question_id: str) -> Optional[List[List[str]]]:
        """Get the table data for a specific question.

        Args:
            question_id: The question ID

        Returns:
            2D list representing the table, or None if not found
        """
        df = self.load_dataset()
        row = df[df['question_id'] == question_id]

        if row.empty:
            return None

        table_str = row.iloc[0].get('table', '[]')
        try:
            return json.loads(table_str)
        except json.JSONDecodeError:
            return None

    def get_context_for_question(self, question_id: str) -> str:
        """Get the text context for a specific question.

        Args:
            question_id: The question ID

        Returns:
            Combined pre_text and post_text context
        """
        df = self.load_dataset()
        row = df[df['question_id'] == question_id]

        if row.empty:
            return ""

        return row.iloc[0].get('context', '')
