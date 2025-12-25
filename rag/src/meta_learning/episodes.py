"""Episode/task containers for meta-learning experiments (stub)."""

from dataclasses import dataclass
from typing import Any
import pandas as pd


@dataclass
class Episode:
    """Support/query split for a task."""

    support: pd.DataFrame
    query: pd.DataFrame
    corpus_id: str
    domain_id: str
    metadata: Any | None = None
