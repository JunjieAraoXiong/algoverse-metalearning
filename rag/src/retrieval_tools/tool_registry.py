"""Registry for retrieval pipelines/policies."""

from typing import List, Tuple
from langchain_core.documents import Document

from .base import RetrievalPipeline
from .semantic import build_semantic_retriever, set_retriever_k as set_semantic_k, take_top_k as take_semantic_top_k
from .hybrid import build_hybrid_retriever, set_retriever_k as set_hybrid_k, take_top_k as take_hybrid_top_k
from .metadata_filter import filter_with_question_metadata
from .rerank import get_reranker


class SimplePipeline(RetrievalPipeline):
    """Composable retrieval pipeline."""

    def __init__(
        self,
        retriever,
        top_k: int,
        use_metadata_filter: bool,
        use_rerank: bool,
        initial_k_factor: float,
        set_k_fn,
        take_top_k_fn,
        reranker_model: str = None,
    ):
        self.retriever = retriever
        self.top_k = top_k
        self.use_metadata_filter = use_metadata_filter
        self.use_rerank = use_rerank
        self.initial_k_factor = max(1.0, float(initial_k_factor))
        self._set_k = set_k_fn
        self._take_top_k = take_top_k_fn
        self._reranker = None
        self._reranker_model = reranker_model

    def retrieve(self, question: str) -> List[Document]:
        multiplier = self.initial_k_factor if (self.use_metadata_filter or self.use_rerank) else 1.0
        initial_k = max(self.top_k, int(self.top_k * multiplier))
        self._set_k(self.retriever, initial_k)

        docs = self.retriever.invoke(question)

        if self.use_metadata_filter:
            filtered_docs, used_metadata = filter_with_question_metadata(question, docs)
            if used_metadata:
                docs = filtered_docs
            elif filtered_docs:
                docs = filtered_docs

        if self.use_rerank:
            self._reranker = get_reranker(self._reranker, model_name=self._reranker_model) if self._reranker_model else get_reranker(self._reranker)
            docs = self._reranker.rerank(question, docs, self.top_k)
        else:
            docs = self._take_top_k(docs, self.top_k)

        return docs


def _pipeline_flags(pipeline_id: str) -> Tuple[bool, bool, bool]:
    """Return (use_hybrid, use_filter, use_rerank) for a pipeline id."""
    mapping = {
        "semantic": (False, False, False),
        "hybrid": (True, False, False),
        "hybrid_filter": (True, True, False),
        "hybrid_filter_rerank": (True, True, True),
    }
    if pipeline_id not in mapping:
        raise ValueError(f"Unknown pipeline_id '{pipeline_id}'")
    return mapping[pipeline_id]


def build_retriever_for_pipeline(pipeline_id: str, db, top_k: int):
    """Return a retriever and helpers for the given pipeline."""
    use_hybrid, _, _ = _pipeline_flags(pipeline_id)
    if use_hybrid:
        retriever = build_hybrid_retriever(db, top_k=top_k)
        return retriever, set_hybrid_k, take_hybrid_top_k
    retriever = build_semantic_retriever(db, top_k=top_k)
    return retriever, set_semantic_k, take_semantic_top_k


def build_pipeline(
    pipeline_id: str,
    retriever,
    top_k: int,
    initial_k_factor: float,
    set_k_fn,
    take_top_k_fn,
    reranker_model: str = None,
) -> SimplePipeline:
    """Construct a SimplePipeline for the pipeline id."""
    use_hybrid, use_filter, use_rerank = _pipeline_flags(pipeline_id)
    # use_hybrid is already baked into retriever selection
    return SimplePipeline(
        retriever=retriever,
        top_k=top_k,
        use_metadata_filter=use_filter,
        use_rerank=use_rerank,
        initial_k_factor=initial_k_factor,
        set_k_fn=set_k_fn,
        take_top_k_fn=take_top_k_fn,
        reranker_model=reranker_model,
    )


def list_pipelines() -> List[str]:
    """Return supported pipeline ids."""
    return ["semantic", "hybrid", "hybrid_filter", "hybrid_filter_rerank", "routed"]


# Re-export routed pipeline builder for convenience
from .router import build_routed_pipeline
