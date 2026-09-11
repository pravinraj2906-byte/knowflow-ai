"""
RAG package for KnowFlow AI.
Includes embeddings, vector storage, retrieval, and grounded generation.
"""

from src.rag.embeddings import (
    GeminiEmbedder,
    embed_documents,
    embed_query,
    get_default_embedder,
    EmbeddingError,
)
from src.rag.vector_store import VectorStore, IncompatibleEmbeddingDimensionError
from src.rag.retriever import (
    Retriever,
    RetrievalResult,
    INSUFFICIENT_CONTEXT_MESSAGE,
)
from src.rag.generator import GroundedGenerator, GenerationResult

__all__ = [
    "GeminiEmbedder",
    "embed_documents",
    "embed_query",
    "get_default_embedder",
    "EmbeddingError",
    "VectorStore",
    "IncompatibleEmbeddingDimensionError",
    "Retriever",
    "RetrievalResult",
    "INSUFFICIENT_CONTEXT_MESSAGE",
    "GroundedGenerator",
    "GenerationResult",
]
