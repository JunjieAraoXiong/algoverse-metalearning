"""Adapter for the PubMedQA benchmark."""

from typing import Optional, List
import pandas as pd
from datasets import load_dataset

from .base_adapter import BaseDatasetAdapter


class PubMedQAAdapter(BaseDatasetAdapter):
    """Adapter for PubMedQA (pqa_labeled or pqa_l)."""

    def __init__(self, split: str = "train", subset_csv: Optional[str] = None, config: str = "pqa_l"):
        """
        Args:
            split: Dataset split to load (default: 'train')
            subset_csv: Optional path to CSV with subset of questions to test
            config: PubMedQA config ('pqa_l' for labeled abstracts)
        """
        self.split = split
        self.dataset_id = "pubmedqa"
        self.config = config
        self.subset_csv = subset_csv

    def load_dataset(self) -> pd.DataFrame:
        """Load PubMedQA dataset.

        Returns:
            pd.DataFrame with columns: question, long_answer, context, final_decision, pmid, title
        """
        if self.subset_csv:
            df = pd.read_csv(self.subset_csv)
            return df

        ds = load_dataset(self.dataset_id, self.config, split=self.split)
        df = ds.to_pandas()

        # Normalize column names to expected fields
        # PubMedQA provides: 'id', 'question', 'context', 'long_answer', 'final_decision'
        # We add 'pmid' (id) and 'title' if present in context metadata (not always available)
        if 'id' in df.columns:
            df['pmid'] = df['id']
        if 'title' not in df.columns:
            df['title'] = None

        return df

    def get_question_column(self) -> str:
        return "question"

    def get_answer_column(self) -> str:
        # Use long_answer for richer semantic comparison; fallback to final_decision (yes/no/maybe)
        return "long_answer"

    def get_question_type_column(self) -> Optional[str]:
        return None

    def get_metadata_columns(self) -> List[str]:
        cols = []
        for c in ["pmid", "title", "context", "final_decision"]:
            cols.append(c)
        return cols

    def get_dataset_name(self) -> str:
        return "pmqa"
