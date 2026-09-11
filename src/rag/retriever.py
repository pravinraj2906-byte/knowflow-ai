"""
Retriever module for KnowFlow AI.
Embeds user queries, searches the ChromaDB vector store, and applies
relevance thresholds for hallucination prevention.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.rag.embeddings import GeminiEmbedder, get_default_embedder
from src.rag.vector_store import VectorStore

logger = logging.getLogger(__name__)

# Grounded fallback message required across the application
INSUFFICIENT_CONTEXT_MESSAGE = (
    "I couldn't find enough information in the uploaded documents to answer this question."
)

# Cosine distance threshold (chunks with distance > threshold are considered irrelevant)
DEFAULT_MAX_DISTANCE_THRESHOLD = 0.70
DEFAULT_TOP_K = 5


@dataclass
class RetrievalResult:
    """Structured container for retrieval outputs and relevance metadata."""
    query: str
    chunks: List[Dict[str, Any]] = field(default_factory=list)
    has_relevant_context: bool = False
    best_distance: Optional[float] = None
    best_similarity: Optional[float] = None
    total_docs_searched: int = 0
    fallback_message: str = INSUFFICIENT_CONTEXT_MESSAGE


class Retriever:
    """
    Retrieves the most semantically relevant chunks for a user query,
    enforcing hallucination control via distance thresholds.
    """

    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        embedder: Optional[GeminiEmbedder] = None,
        default_top_k: int = DEFAULT_TOP_K,
        relevance_threshold: float = DEFAULT_MAX_DISTANCE_THRESHOLD,
    ):
        self.vector_store = vector_store or VectorStore()
        self.embedder = embedder or get_default_embedder()
        self.default_top_k = default_top_k
        self.relevance_threshold = relevance_threshold

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        relevance_threshold: Optional[float] = None,
    ) -> RetrievalResult:
        """
        Execute semantic retrieval for a user question.
        
        Args:
            query: User question.
            top_k: Number of chunks to retrieve (defaults to self.default_top_k).
            relevance_threshold: Max cosine distance allowed for a chunk to be relevant.
            
        Returns:
            RetrievalResult containing matched chunks and relevance status.
        """
        k = top_k or self.default_top_k
        threshold = (
            relevance_threshold
            if relevance_threshold is not None
            else self.relevance_threshold
        )

        total_chunks = self.vector_store.count()
        if not query or not query.strip() or total_chunks == 0:
            return RetrievalResult(
                query=query,
                chunks=[],
                has_relevant_context=False,
                total_docs_searched=total_chunks,
            )

        # 1. Embed query
        try:
            query_embedding = self.embedder.embed_query(query.strip())
        except Exception as e:
            logger.error("Failed to generate query embedding: %s", e)
            raise

        # 2. Query vector store
        raw_chunks = self.vector_store.similarity_search(
            query_embedding=query_embedding,
            top_k=k,
        )

        if not raw_chunks:
            return RetrievalResult(
                query=query,
                chunks=[],
                has_relevant_context=False,
                total_docs_searched=total_chunks,
            )

        best_chunk = raw_chunks[0]
        best_distance = best_chunk.get("distance", 1.0)
        best_similarity = best_chunk.get("similarity", 0.0)

        # 3. Filter chunks by relevance threshold (hallucination guardrail)
        relevant_chunks = [
            chunk for chunk in raw_chunks
            if chunk.get("distance", 1.0) <= threshold
        ]

        has_relevant = len(relevant_chunks) > 0

        logger.info(
            "Query '%s' retrieved %d chunks (best distance: %.3f, threshold: %.3f, relevant: %s)",
            query,
            len(raw_chunks),
            best_distance,
            threshold,
            has_relevant,
        )

        return RetrievalResult(
            query=query,
            chunks=relevant_chunks,
            has_relevant_context=has_relevant,
            best_distance=best_distance,
            best_similarity=best_similarity,
            total_docs_searched=total_chunks,
        )
