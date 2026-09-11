"""
Embedding module for KnowFlow AI using the modern Google GenAI SDK.
Supports embedding document chunks and user queries with gemini-embedding-2.
Includes robust batching, exponential backoff retries for transient/SSL errors,
and strict 1-to-1 chunk-to-embedding mapping without deprecated task_type.
"""

import logging
import os
import random
import ssl
import time
from typing import List, Optional

from google import genai
from google.genai import types

from src.utils.helpers import load_api_key

logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_MODEL = "gemini-embedding-2"
DEFAULT_BATCH_SIZE = 10
MAX_RETRIES = 3
INITIAL_BACKOFF = 1.0
BACKOFF_FACTOR = 2.0


class EmbeddingError(Exception):
    """Raised when an embedding request fails."""
    pass


class GeminiEmbedder:
    """
    Isolated embedding provider using Google GenAI SDK.
    Embeds document chunks and user queries into the shared gemini-embedding-2 vector space.
    Does NOT use deprecated task_type parameter.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_retries: int = MAX_RETRIES,
        initial_backoff: float = INITIAL_BACKOFF,
        backoff_factor: float = BACKOFF_FACTOR,
    ):
        self.api_key = api_key or load_api_key()
        self.model_name = (
            model_name
            or os.getenv("EMBEDDING_MODEL")
            or DEFAULT_EMBEDDING_MODEL
        )
        self.batch_size = max(1, batch_size)
        self.max_retries = max(1, max_retries)
        self.initial_backoff = initial_backoff
        self.backoff_factor = backoff_factor
        self._client: Optional[genai.Client] = None

    @property
    def client(self) -> genai.Client:
        """Lazy initialization of Google GenAI Client."""
        if self._client is None:
            if not self.api_key:
                raise ValueError(
                    "Gemini API key is missing. Please set GEMINI_API_KEY in your .env file or environment."
                )
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    @client.setter
    def client(self, value: Optional[genai.Client]) -> None:
        self._client = value

    def _embed_batch_with_retry(
        self,
        contents: List[types.Content],
        expected_count: int,
        batch_idx: int = 0,
    ) -> List[List[float]]:
        """
        Execute embed_content for a batch of Content objects with exponential backoff.
        Refreshes client on SSL/socket disconnects to ensure a clean TLS handshake.
        """
        last_error: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 1):
            try:
                # Call modern embed_content with contents list and NO task_type
                response = self.client.models.embed_content(
                    model=self.model_name,
                    contents=contents,
                )

                if not response or not response.embeddings:
                    raise EmbeddingError(
                        f"Gemini API returned empty embeddings response for batch at index {batch_idx}."
                    )

                if len(response.embeddings) != expected_count:
                    raise EmbeddingError(
                        f"Embedding count mismatch for batch at index {batch_idx}: "
                        f"expected {expected_count} embeddings, received {len(response.embeddings)}."
                    )

                # Extract float vector values for each chunk
                return [item.values for item in response.embeddings]

            except Exception as e:
                last_error = e
                err_str = str(e)

                # Classify transient errors: SSL EOF, socket reset, timeouts, rate limits (429), or 5xx
                is_ssl_or_conn = (
                    isinstance(e, (ssl.SSLError, ConnectionError, TimeoutError, OSError))
                    or "ssl" in err_str.lower()
                    or "eof" in err_str.lower()
                    or "connection" in err_str.lower()
                    or "timeout" in err_str.lower()
                    or "broken pipe" in err_str.lower()
                )
                is_rate_or_server = (
                    "429" in err_str
                    or "500" in err_str
                    or "502" in err_str
                    or "503" in err_str
                    or "504" in err_str
                    or "resource_exhausted" in err_str.lower()
                    or "unavailable" in err_str.lower()
                )

                # Reset client on connection/SSL failures to force clean TLS handshake on retry
                if is_ssl_or_conn:
                    logger.warning(
                        "SSL/Socket error detected on batch at index %d: %s. Resetting client for next attempt.",
                        batch_idx,
                        err_str,
                    )
                    self._client = None

                if attempt < self.max_retries and (is_ssl_or_conn or is_rate_or_server):
                    delay = self.initial_backoff * (self.backoff_factor ** (attempt - 1)) + random.uniform(0.1, 0.4)
                    logger.warning(
                        "Transient error on batch index %d (attempt %d/%d): %s. Retrying in %.2fs...",
                        batch_idx,
                        attempt,
                        self.max_retries,
                        err_str,
                        delay,
                    )
                    time.sleep(delay)
                else:
                    logger.error(
                        "Embedding failed for batch at index %d on attempt %d/%d: %s",
                        batch_idx,
                        attempt,
                        self.max_retries,
                        err_str,
                    )
                    if not (is_ssl_or_conn or is_rate_or_server):
                        # Non-transient error (e.g. invalid auth, invalid model), abort immediately
                        raise EmbeddingError(f"Failed to embed document chunks: {err_str}") from e

        raise EmbeddingError(
            f"Failed to embed document batch (chunks {batch_idx} to {batch_idx + expected_count}) "
            f"after {self.max_retries} attempts. Last error: {str(last_error)}"
        ) from last_error

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of document chunks.
        
        Guarantees:
        - Exactly one embedding per input chunk (1-to-1 mapping).
        - Does NOT aggregate all chunks into one single content.
        - Does NOT pass task_type (compatible with gemini-embedding-2).
        - Automatically batches requests into safe chunk batches.
        - Retries transient network/SSL failures with exponential backoff.
        
        Args:
            texts: List of chunk texts to embed.
            
        Returns:
            List of float vectors, exactly matching the length of texts.
        """
        if not texts:
            return []

        all_embeddings: List[List[float]] = []

        # Batch texts to prevent socket timeouts and respect API payload limits
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]

            # Construct explicit Content objects per chunk to prevent the SDK
            # from collapsing multiple strings into a single UserContent with multiple parts
            contents = [
                types.Content(parts=[types.Part.from_text(text=t)])
                for t in batch
            ]

            batch_embeddings = self._embed_batch_with_retry(
                contents=contents,
                expected_count=len(batch),
                batch_idx=i,
            )

            all_embeddings.extend(batch_embeddings)

        if len(all_embeddings) != len(texts):
            raise EmbeddingError(
                f"Fatal embedding count mismatch: expected {len(texts)} embeddings, "
                f"produced {len(all_embeddings)}."
            )

        return all_embeddings

    def embed_query(self, query: str) -> List[float]:
        """
        Embed a single user search query into the same vector space as documents.
        Does NOT use task_type.
        
        Args:
            query: The user question or search phrase.
            
        Returns:
            Single float vector representing the query embedding.
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty for embedding.")

        contents = [types.Content(parts=[types.Part.from_text(text=query.strip())])]

        embeddings = self._embed_batch_with_retry(
            contents=contents,
            expected_count=1,
            batch_idx=0,
        )

        return embeddings[0]


# Module-level convenience functions
_default_embedder: Optional[GeminiEmbedder] = None


def get_default_embedder() -> GeminiEmbedder:
    """Singleton getter for the default embedder."""
    global _default_embedder
    if _default_embedder is None:
        _default_embedder = GeminiEmbedder()
    return _default_embedder


def embed_documents(texts: List[str]) -> List[List[float]]:
    """Convenience function to embed documents using default embedder."""
    return get_default_embedder().embed_documents(texts)


def embed_query(query: str) -> List[float]:
    """Convenience function to embed a query using default embedder."""
    return get_default_embedder().embed_query(query)
