"""Build a Chroma index for PubMedQA (pqa_l)."""

from pathlib import Path
from typing import List
from dotenv import load_dotenv

from datasets import load_dataset
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_chroma import Chroma

load_dotenv()


def load_pubmedqa(split: str = "train", config: str = "pqa_l"):
    """Load PubMedQA split."""
    return load_dataset("pubmedqa", config, split=split)


def to_documents(ds) -> List[Document]:
    docs: List[Document] = []
    for row in ds:
        title = row.get("title") or ""
        context = row.get("context") or ""
        pmid = row.get("id") or row.get("pmid") or ""
        question = row.get("question") or ""

        content = f"Title: {title}\nQuestion: {question}\nAbstract: {context}"
        meta = {
            "source": f"pubmedqa_{pmid}",
            "pmid": pmid,
            "title": title,
        }
        docs.append(Document(page_content=content, metadata=meta))
    return docs


def chunk_documents(documents: List[Document], chunk_size: int = 800, chunk_overlap: int = 100) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,
    )
    return splitter.split_documents(documents)


def build_index(split: str = "train", config: str = "pqa_l", persist_dir: Path | None = None):
    if persist_dir is None:
        persist_dir = Path(__file__).parent.parent / "chroma_pubmedqa"
    persist_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading PubMedQA ({config}, split={split})...")
    ds = load_pubmedqa(split=split, config=config)
    documents = to_documents(ds)
    print(f"Loaded {len(documents)} documents")

    chunks = chunk_documents(documents)
    print(f"Chunked into {len(chunks)} chunks (size ~{len(chunks[0].page_content) if chunks else 0} chars)")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    print(f"Embedding and writing to {persist_dir} ...")
    db = Chroma.from_documents(chunks, embeddings, persist_directory=str(persist_dir))
    print(f"Done. Total chunks: {len(chunks)}. Persisted at {persist_dir}")
    return db


if __name__ == "__main__":
    build_index()
