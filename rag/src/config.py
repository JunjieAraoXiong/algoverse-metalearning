"""Central configuration for RAG system."""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from dotenv import load_dotenv

load_dotenv()


# =============================================================================
# Embedding Configuration
# =============================================================================

@dataclass
class EmbeddingConfig:
    """Configuration for embedding models."""
    name: str
    model_id: str
    provider: str  # "local" or "openai"
    dimension: int
    description: str


# Embedding registry - local models are FREE
EMBEDDINGS: Dict[str, EmbeddingConfig] = {
    # FREE local models (recommended)
    "bge-large": EmbeddingConfig(
        name="bge-large",
        model_id="BAAI/bge-large-en-v1.5",
        provider="local",
        dimension=1024,
        description="Best free option - comparable to OpenAI",
    ),
    "bge-base": EmbeddingConfig(
        name="bge-base",
        model_id="BAAI/bge-base-en-v1.5",
        provider="local",
        dimension=768,
        description="Good quality, smaller/faster",
    ),
    "gte-large": EmbeddingConfig(
        name="gte-large",
        model_id="Alibaba-NLP/gte-large-en-v1.5",
        provider="local",
        dimension=1024,
        description="Excellent quality, slightly slower",
    ),
    "nomic": EmbeddingConfig(
        name="nomic",
        model_id="nomic-ai/nomic-embed-text-v1.5",
        provider="local",
        dimension=768,
        description="Good quality, fast",
    ),
    # Paid OpenAI models (expensive - avoid)
    "openai-large": EmbeddingConfig(
        name="openai-large",
        model_id="text-embedding-3-large",
        provider="openai",
        dimension=3072,
        description="OpenAI - $0.13/1M tokens (PAID)",
    ),
    "openai-small": EmbeddingConfig(
        name="openai-small",
        model_id="text-embedding-3-small",
        provider="openai",
        dimension=1536,
        description="OpenAI - $0.02/1M tokens (PAID)",
    ),
}


def get_embedding_model(embedding_name: str = "bge-large"):
    """Get an embedding model instance (lazy loaded)."""
    if embedding_name not in EMBEDDINGS:
        raise ValueError(f"Unknown embedding: {embedding_name}. Available: {list(EMBEDDINGS.keys())}")

    config = EMBEDDINGS[embedding_name]

    if config.provider == "local":
        from langchain_huggingface import HuggingFaceEmbeddings
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device} for embeddings")
        return HuggingFaceEmbeddings(
            model_name=config.model_id,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True},
        )
    elif config.provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model=config.model_id)
    else:
        raise ValueError(f"Unknown provider: {config.provider}")


# =============================================================================
# LLM Provider Configuration
# =============================================================================

@dataclass
class ProviderConfig:
    """Configuration for an LLM provider."""
    name: str
    base_url: Optional[str]
    api_key_env: str
    models: List[str]

    @property
    def api_key(self) -> Optional[str]:
        return os.environ.get(self.api_key_env)


# Provider registry - add new providers here
PROVIDERS: Dict[str, ProviderConfig] = {
    "openai": ProviderConfig(
        name="openai",
        base_url=None,
        api_key_env="OPENAI_API_KEY",
        models=["gpt-5.2", "gpt-5.2-mini", "gpt-4o", "gpt-4o-mini"],
    ),
    "anthropic": ProviderConfig(
        name="anthropic",
        base_url=None,
        api_key_env="ANTHROPIC_API_KEY",
        models=[
            "claude-sonnet-4-5-20250514",
            "claude-opus-4-5-20250514",
            "claude-sonnet-4-20250514",
        ],
    ),
    "google": ProviderConfig(
        name="google",
        base_url=None,
        api_key_env="GOOGLE_API_KEY",
        models=["gemini-3-pro", "gemini-3-flash", "gemini-2.0-flash"],
    ),
    "together": ProviderConfig(
        name="together",
        base_url="https://api.together.xyz/v1",
        api_key_env="TOGETHER_API_KEY",
        models=[
            "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
            "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
            "deepseek-ai/DeepSeek-V3",
        ],
    ),
    "deepseek": ProviderConfig(
        name="deepseek",
        base_url="https://api.deepseek.com/v1",
        api_key_env="DEEPSEEK_API_KEY",
        models=["deepseek-chat", "deepseek-reasoner"],
    ),
    "local-vllm": ProviderConfig(
        name="local-vllm",
        base_url="http://localhost:8000/v1",
        api_key_env="EMPTY_KEY",  # vLLM doesn't need a real key, but provider might check existence
        models=["meta-llama/Meta-Llama-3.1-70B-Instruct"],
    ),
}


def get_provider_for_model(model_name: str) -> str:
    """Determine which provider to use based on model name."""
    model_lower = model_name.lower()
    if model_lower.startswith("gpt-"):
        return "openai"
    elif model_lower.startswith("claude-"):
        return "anthropic"
    elif model_lower.startswith("gemini-"):
        return "google"
    elif "deepseek" in model_lower and not model_lower.startswith("deepseek-ai/"):
        return "deepseek"
    elif "meta-llama" in model_lower:
        if "turbo" in model_lower:
            return "together"  # Turbo models use Together API
        return "local-vllm"
    return "together"


def get_provider_config(model_name: str) -> ProviderConfig:
    """Get provider configuration for a model."""
    provider_name = get_provider_for_model(model_name)
    return PROVIDERS[provider_name]


# =============================================================================
# Reranker Configuration
# =============================================================================

@dataclass
class RerankerConfig:
    """Configuration for reranker models."""
    name: str
    description: str


RERANKERS: Dict[str, RerankerConfig] = {
    "BAAI/bge-reranker-large": RerankerConfig(
        name="BAAI/bge-reranker-large",
        description="High quality, slower",
    ),
    "BAAI/bge-reranker-base": RerankerConfig(
        name="BAAI/bge-reranker-base",
        description="Good quality, medium speed",
    ),
    "cross-encoder/ms-marco-MiniLM-L-6-v2": RerankerConfig(
        name="cross-encoder/ms-marco-MiniLM-L-6-v2",
        description="Fast, lower quality",
    ),
}


# =============================================================================
# Router Configuration
# =============================================================================

@dataclass
class RouteConfig:
    """Configuration for a specific question type route."""
    pipeline_id: str
    top_k: int
    initial_k_factor: float
    use_hyde: bool = False
    use_table_preference: bool = False
    table_quota_ratio: float = 0.6


ROUTES: Dict[str, RouteConfig] = {
    "metrics-generated": RouteConfig(
        pipeline_id="hybrid_filter_rerank",
        top_k=10,  # Was 5 - increased to retrieve more table chunks
        initial_k_factor=6.0,  # Was 4.0 - retrieve 60 docs initially
        use_hyde=False,
        use_table_preference=True,
        table_quota_ratio=0.9,  # Was 0.6 - prioritize 90% table chunks
    ),
    "domain-relevant": RouteConfig(
        pipeline_id="hybrid_filter_rerank",
        top_k=5,
        initial_k_factor=3.0,
        use_hyde=False,
        use_table_preference=False,
    ),
    "novel-generated": RouteConfig(
        pipeline_id="hybrid_filter_rerank",
        top_k=8,
        initial_k_factor=3.0,
        use_hyde=True,
        use_table_preference=False,
    ),
}


# =============================================================================
# Default Settings
# =============================================================================

@dataclass
class Defaults:
    """Default values for the RAG system."""
    # Model defaults
    llm_model: str = "claude-sonnet-4-5-20250514"
    embedding_model: str = "bge-large"  # FREE local embedding (was text-embedding-3-large)
    reranker_model: str = "BAAI/bge-reranker-large"  # Also FREE local
    judge_model: str = "claude-sonnet-4-5-20250514"

    # Retrieval defaults
    top_k: int = 5
    initial_k_factor: float = 3.0
    pipeline_id: str = "hybrid_filter_rerank"
    ensemble_weights: tuple = (0.3, 0.7)  # (BM25, semantic) - favor semantic for better table matching
    rerank_threshold: float = 0.0  # Minimum reranker score (0.0 = no filtering, try 0.1-0.3)

    # Generation defaults
    temperature: float = 0.0
    max_tokens: int = 512

    # Router defaults
    router_classifier_model: str = "gpt-4o-mini"
    router_hyde_model: str = "gpt-4o-mini"

    # Paths (relative to project root)
    chroma_path: str = "chroma"
    output_dir: str = "bulk_runs"


DEFAULTS = Defaults()


# =============================================================================
# Model Name Abbreviations (for filenames)
# =============================================================================

MODEL_ABBREVS: Dict[str, str] = {
    # Claude
    "claude-sonnet-4-5": "claude45-sonnet",
    "claude-opus-4-5": "claude45-opus",
    "claude-sonnet-4": "claude4-sonnet",
    # GPT
    "gpt-5.2-mini": "gpt52-mini",
    "gpt-5.2": "gpt52",
    "gpt-4o-mini": "gpt4o-mini",
    "gpt-4o": "gpt4o",
    # Gemini
    "gemini-3-flash": "gemini3-flash",
    "gemini-3-pro": "gemini3-pro",
    "gemini-2": "gemini2-flash",
    # Llama
    "llama-4": "llama4",
    "llama-3.1-70b": "llama31-70b",
    "llama-3.1-8b": "llama31-8b",
    # DeepSeek
    "deepseek-v3": "deepseek-v3",
    "deepseek-chat": "deepseek-chat",
    "deepseek-reasoner": "deepseek-r1",
}


def get_model_abbrev(model_name: str) -> str:
    """Get abbreviated model name for filenames."""
    model_lower = model_name.lower()
    for pattern, abbrev in MODEL_ABBREVS.items():
        if pattern in model_lower:
            return abbrev
    # Fallback: use last part of model name
    return model_name.split("/")[-1][:20]


# =============================================================================
# Pipeline Configuration
# =============================================================================

PIPELINES = ["semantic", "hybrid", "hybrid_filter", "hybrid_filter_rerank", "routed"]


def get_pipeline_flags(pipeline_id: str) -> tuple:
    """Return (use_hybrid, use_filter, use_rerank) for a pipeline id."""
    mapping = {
        "semantic": (False, False, False),
        "hybrid": (True, False, False),
        "hybrid_filter": (True, True, False),
        "hybrid_filter_rerank": (True, True, True),
    }
    if pipeline_id not in mapping:
        raise ValueError(f"Unknown pipeline_id '{pipeline_id}'. Available: {PIPELINES}")
    return mapping[pipeline_id]
