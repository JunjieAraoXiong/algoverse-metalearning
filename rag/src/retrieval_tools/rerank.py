"""Cross-encoder reranking tool."""

from typing import List, Optional
from langchain_core.documents import Document


# Default to stronger BGE reranker
DEFAULT_RERANKER = "BAAI/bge-reranker-large"

# Available reranker models (in order of quality/speed tradeoff)
RERANKER_MODELS = {
    "BAAI/bge-reranker-large": "High quality, slower",
    "BAAI/bge-reranker-base": "Good quality, medium speed",
    "cross-encoder/ms-marco-MiniLM-L-6-v2": "Fast, lower quality",
}


class Reranker:
    """Wraps a cross-encoder reranker."""

    def __init__(self, model_name: str = DEFAULT_RERANKER):
        from sentence_transformers import CrossEncoder
        import torch

        self.model_name = model_name
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  Loading reranker: {model_name} on {device}")
        self.model = CrossEncoder(model_name, device=device)

    def rerank(self, question: str, docs: List[Document], top_k: int) -> List[Document]:
        """Rerank docs and return top_k."""
        if not docs:
            return docs

        pairs = [[question, doc.page_content] for doc in docs]
        scores = self.model.predict(pairs)

        scored_docs = list(zip(docs, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in scored_docs[:top_k]]


# Global reranker cache to avoid reloading
_reranker_cache: dict = {}


def get_reranker(existing: Optional["Reranker"] = None, model_name: str = DEFAULT_RERANKER) -> "Reranker":
    """Return an existing reranker or create/cache a new one."""
    if existing and existing.model_name == model_name:
        return existing

    # Check cache
    if model_name in _reranker_cache:
        return _reranker_cache[model_name]

    # Create and cache
    reranker = Reranker(model_name=model_name)
    _reranker_cache[model_name] = reranker
    return reranker
