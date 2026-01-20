"""CUAD (Contract Understanding Atticus Dataset) adapter for legal domain.

CUAD is a legal contract QA dataset with 510 contracts and 41 clause types.
Used as held-out domain for cross-domain meta-learning evaluation.

Dataset: https://www.atticusprojectai.org/cuad
HuggingFace: https://huggingface.co/datasets/cuad
"""

import json
from pathlib import Path
from typing import List, Optional

import pandas as pd

from dataset_adapters.base import BaseDatasetAdapter


# Default path relative to rag/ directory
DEFAULT_DATA_PATH = Path(__file__).parent.parent / "data" / "cuad" / "cuad_questions.jsonl"


class CUADAdapter(BaseDatasetAdapter):
    """Adapter for the CUAD legal contract QA dataset.

    CUAD contains extractive QA questions about legal contracts.
    Each question asks about a specific clause type in a contract.

    Note: CUAD is extractive (find span in text), so we reformulate
    questions for RAG evaluation: "What is the [clause_type] in this contract?"

    Fields:
    - question: The question about a clause
    - context: Contract text (or relevant section)
    - answer: The extracted clause text (or "Not found")
    - clause_type: One of 41 clause types
    - contract_id: Identifier for the source contract
    """

    # Common clause types in CUAD for question type classification
    CLAUSE_CATEGORIES = {
        "financial": [
            "Anti-Assignment",
            "Cap On Liability",
            "Competitive Restriction Exception",
            "Liquidated Damages",
            "Minimum Commitment",
            "Price Restrictions",
            "Revenue/Profit Sharing",
        ],
        "termination": [
            "Expiration Date",
            "Renewal Term",
            "Termination For Convenience",
            "Post-Termination Services",
            "Rofr/Rofo/Rofn",
        ],
        "ip_rights": [
            "Ip Ownership Assignment",
            "Joint Ip Ownership",
            "License Grant",
            "Non-Compete",
            "Non-Solicitation Of Employees",
        ],
        "liability": [
            "Audit Rights",
            "Change Of Control",
            "Insurance",
            "Limitation Of Liability",
            "Uncapped Liability",
            "Warranty Duration",
        ],
    }

    def __init__(
        self,
        data_path: Optional[str] = None,
        subset_csv: Optional[str] = None,
        clause_types: Optional[List[str]] = None,
    ):
        """Initialize the CUAD adapter.

        Args:
            data_path: Path to the CUAD JSONL file.
            subset_csv: Optional path to CSV with subset of question IDs.
            clause_types: Optional list of clause types to filter.
        """
        super().__init__(subset_csv=subset_csv)
        self.data_path = Path(data_path) if data_path else DEFAULT_DATA_PATH
        self.clause_types = clause_types

    def load_dataset(self) -> pd.DataFrame:
        """Load the CUAD dataset from JSONL.

        Returns:
            DataFrame with questions, answers, and metadata

        Raises:
            FileNotFoundError: If data file doesn't exist
        """
        if not self.data_path.exists():
            raise FileNotFoundError(
                f"CUAD data not found at {self.data_path}. "
                f"Run scripts/prepare_cuad.py to download and prepare the dataset."
            )

        records = []
        with open(self.data_path, "r") as f:
            for line in f:
                record = json.loads(line.strip())
                records.append(record)

        df = pd.DataFrame(records)

        # Filter by clause types if specified
        if self.clause_types and "clause_type" in df.columns:
            df = df[df["clause_type"].isin(self.clause_types)]

        # Apply subset filtering if provided
        df = self._apply_subset_filter(df)

        return df

    def get_question_column(self) -> str:
        """Return the question column name."""
        return "question"

    def get_answer_column(self) -> str:
        """Return the answer column name."""
        return "answer"

    def get_question_type_column(self) -> Optional[str]:
        """Return the question type column name (clause_type)."""
        return "clause_type"

    def get_context_column(self) -> Optional[str]:
        """Return the context column name (contract text)."""
        return "context"

    def get_metadata_columns(self) -> List[str]:
        """Return additional metadata columns."""
        return ["question_id", "contract_id", "clause_type", "context"]

    def get_clause_category(self, clause_type: str) -> str:
        """Map clause type to broader category for analysis."""
        for category, clauses in self.CLAUSE_CATEGORIES.items():
            if clause_type in clauses:
                return category
        return "other"

    @property
    def name(self) -> str:
        """Return the dataset name."""
        return "cuad"


def prepare_cuad_from_huggingface(output_path: Optional[str] = None):
    """Download and prepare CUAD dataset from HuggingFace.

    This function downloads the CUAD dataset and converts it to our
    JSONL format for use with the adapter.

    Args:
        output_path: Path to save the prepared JSONL file.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Please install datasets: pip install datasets")

    output_path = Path(output_path) if output_path else DEFAULT_DATA_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Downloading CUAD from HuggingFace (chenghao/cuad_qa)...")
    # Use chenghao/cuad_qa which has standard format (no deprecated scripts)
    dataset = load_dataset("chenghao/cuad_qa", split="test")

    print(f"Processing {len(dataset)} examples...")
    records = []

    for idx, example in enumerate(dataset):
        # chenghao/cuad_qa format:
        # - id: "ContractName__ClauseType"
        # - title: contract filename
        # - context: full contract text
        # - question: clause type name
        # - answers: {"text": [...], "answer_start": [...]}
        question = example["question"]
        context = example["context"]
        answers = example["answers"]
        contract_title = example.get("title", "unknown")

        # Get the first answer (or "Not found" if empty)
        if answers["text"]:
            answer = answers["text"][0]
        else:
            answer = "Not found in contract"

        # Extract clause type from the question field (it's the clause name)
        clause_type = question

        # Create a question that's more suitable for RAG
        # Original question is just the clause type name like "Document Name"
        rag_question = f"What is the {question} clause in this contract?"

        record = {
            "question_id": f"CUAD{idx:05d}",
            "contract_id": contract_title,
            "question": rag_question,
            "context": context[:8000],  # Keep more context for legal contracts
            "answer": answer,
            "clause_type": clause_type,
            "question_type": "legal_extractive",
        }
        records.append(record)

    # Save to JSONL
    with open(output_path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    print(f"Saved {len(records)} questions to {output_path}")
    return output_path


if __name__ == "__main__":
    # Run this file directly to prepare the dataset
    import argparse

    parser = argparse.ArgumentParser(description="Prepare CUAD dataset")
    parser.add_argument("--output", default=None, help="Output path for JSONL")
    args = parser.parse_args()

    prepare_cuad_from_huggingface(args.output)
