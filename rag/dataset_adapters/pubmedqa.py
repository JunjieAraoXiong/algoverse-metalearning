"""PubMedQA dataset adapter (stub for meta-learning).

This is a placeholder for the PubMedQA healthcare QA dataset,
which will be used for meta-learning experiments to test
cross-domain generalization of retrieval routing.
"""

from typing import List, Optional
import pandas as pd

from dataset_adapters.base import BaseDatasetAdapter


class PubMedQAAdapter(BaseDatasetAdapter):
    """Adapter for the PubMedQA biomedical QA dataset.

    PubMedQA contains yes/no/maybe questions about biomedical research
    abstracts from PubMed.

    Note: This is currently a stub implementation. Full implementation
    will be added when the meta-learning component is developed.
    """

    def __init__(
        self,
        data_path: Optional[str] = None,
        subset_csv: Optional[str] = None,
    ):
        """Initialize the PubMedQA adapter.

        Args:
            data_path: Path to the PubMedQA data file.
            subset_csv: Optional path to CSV with subset of question IDs.
        """
        super().__init__(subset_csv=subset_csv)
        self.data_path = data_path

    def load_dataset(self) -> pd.DataFrame:
        """Load the PubMedQA dataset.

        Returns:
            DataFrame with questions and answers

        Raises:
            NotImplementedError: This is a stub implementation
        """
        raise NotImplementedError(
            "PubMedQA adapter is not yet implemented. "
            "This is a placeholder for meta-learning experiments."
        )

    def get_question_column(self) -> str:
        """Return the question column name."""
        return "question"

    def get_answer_column(self) -> str:
        """Return the answer column name."""
        return "answer"

    def get_question_type_column(self) -> Optional[str]:
        """Return the question type column name."""
        return "question_type"

    def get_metadata_columns(self) -> List[str]:
        """Return additional metadata columns."""
        return ["pubmed_id", "context"]

    @property
    def name(self) -> str:
        """Return the dataset name."""
        return "pubmedqa"
