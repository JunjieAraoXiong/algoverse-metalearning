"""BioASQ dataset adapter for medical domain.

Loads yes/no questions from BioASQ (part of MIRAGE benchmark).
Used for cross-domain testing of retrieval pipelines.
"""

import json
from pathlib import Path
from typing import List, Optional

import pandas as pd

from dataset_adapters.base import BaseDatasetAdapter


MIRAGE_PATH = Path(__file__).parent.parent.parent / "MIRAGE"
BIOASQ_PATH = MIRAGE_PATH / "rawdata" / "bioasq"


class BioASQAdapter(BaseDatasetAdapter):
    """Adapter for BioASQ yes/no questions from MIRAGE.

    BioASQ contains biomedical yes/no questions with PubMed snippets
    as evidence. For RAG testing, we use only the questions (not snippets).
    """

    def __init__(
        self,
        data_path: Optional[str] = None,
        subset_csv: Optional[str] = None,
    ):
        super().__init__(subset_csv=subset_csv)
        self.data_path = Path(data_path) if data_path else BIOASQ_PATH

    def load_dataset(self) -> pd.DataFrame:
        """Load BioASQ yes/no questions from MIRAGE rawdata."""
        records = []

        for task_folder in self.data_path.glob("Task*"):
            for json_file in task_folder.glob("*.json"):
                if json_file.name == "readme.txt":
                    continue

                with open(json_file) as f:
                    data = json.load(f)

                for question in data.get("questions", []):
                    # Only yes/no questions
                    if question.get("type") != "yesno":
                        continue

                    exact_answer = question.get("exact_answer", "")
                    if isinstance(exact_answer, list):
                        exact_answer = exact_answer[0] if exact_answer else ""

                    # Get ideal answer for evaluation
                    ideal = question.get("ideal_answer", [""])
                    if isinstance(ideal, list):
                        ideal = ideal[0] if ideal else ""

                    records.append({
                        "question_id": question.get("id", ""),
                        "question": question.get("body", ""),
                        "answer": exact_answer,  # yes/no
                        "long_answer": ideal,  # Full explanation
                        "question_type": "biomedical_yes_no",
                    })

        df = pd.DataFrame(records)
        df = self._apply_subset_filter(df)
        return df

    def get_question_column(self) -> str:
        return "question"

    def get_answer_column(self) -> str:
        return "long_answer"  # Use full answer for similarity scoring

    def get_question_type_column(self) -> Optional[str]:
        return "question_type"

    def get_metadata_columns(self) -> List[str]:
        return ["question_id", "answer"]

    @property
    def name(self) -> str:
        return "bioasq"
