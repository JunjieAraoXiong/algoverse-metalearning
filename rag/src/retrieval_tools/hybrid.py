"""Hybrid BM25 + semantic retriever builder."""

from typing import Any, List
from langchain_chroma import Chroma
from langchain_core.documents import Document


def build_hybrid_retriever(db: Chroma, top_k: int) -> Any:
    """Create an ensemble of BM25 + semantic retrievers."""
    from langchain_community.retrievers import BM25Retriever
    from langchain.retrievers import EnsembleRetriever

    all_docs = db.get()
    from langchain_core.documents import Document as LCDocument

    documents: List[LCDocument] = [
        LCDocument(page_content=text, metadata=meta)
        for text, meta in zip(all_docs["documents"], all_docs["metadatas"])
    ]

    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = top_k

    semantic_retriever = db.as_retriever(search_kwargs={"k": top_k})

    return EnsembleRetriever(
        retrievers=[bm25_retriever, semantic_retriever],
        weights=[0.5, 0.5],
    )


def set_retriever_k(retriever: Any, k: int) -> None:
    """Update k for an ensemble retriever."""
    if hasattr(retriever, "retrievers") and len(retriever.retrievers) >= 2:
        # BM25
        if hasattr(retriever.retrievers[0], "k"):
            retriever.retrievers[0].k = k
        # Semantic
        if hasattr(retriever.retrievers[1], "search_kwargs"):
            retriever.retrievers[1].search_kwargs["k"] = k


def take_top_k(docs: list[Document], k: int) -> list[Document]:
    """Return the first k documents."""
    return docs[:k]
