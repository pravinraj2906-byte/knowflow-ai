"""
Vector store implementation for KnowFlow AI using ChromaDB.
Provides persistent storage, deduplication, metadata tracking, and similarity search.
Includes automatic detection and prevention of mixed embedding model dimensionalities.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import chromadb
from chromadb.config import Settings

logger = logging.getLogger(__name__)

DEFAULT_PERSIST_DIR = Path("data/chroma").resolve()
COLLECTION_NAME = "knowflow_collection"


class IncompatibleEmbeddingDimensionError(Exception):
    """Raised when incoming embeddings do not match the existing collection's dimensionality."""
    pass


class VectorStore:
    """
    ChromaDB persistent vector store manager for document chunks.
    Ensures persistent storage outside .venv, stable chunk IDs, and zero duplication.
    Guarantees dimensionality consistency across embedding updates.
    """

    def __init__(
        self,
        persist_directory: Optional[Union[str, Path]] = None,
        collection_name: str = COLLECTION_NAME,
    ):
        self.persist_directory = Path(persist_directory or DEFAULT_PERSIST_DIR).resolve()
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name

        # Initialize ChromaDB persistent client
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(anonymized_telemetry=False),
        )

        # Use cosine distance for normalized similarity comparisons
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "Initialized ChromaDB collection '%s' at '%s'. Total chunks: %d",
            self.collection_name,
            self.persist_directory,
            self.count(),
        )

    def count(self) -> int:
        """Return total number of chunks stored in the collection."""
        try:
            return self.collection.count()
        except Exception as e:
            logger.error("Error counting chunks: %s", e)
            return 0

    def get_existing_dimension(self) -> Optional[int]:
        """Inspect the dimensionality of existing vectors in the collection, if any."""
        if self.count() == 0:
            return None
        try:
            existing = self.collection.get(limit=1, include=["embeddings"])
            embeddings = existing.get("embeddings")
            if embeddings is not None and len(embeddings) > 0 and len(embeddings[0]) > 0:
                return len(embeddings[0])
        except Exception as e:
            logger.warning("Unable to determine existing embedding dimension: %s", e)
        return None

    def add_chunks(
        self,
        chunks: List[Dict[str, Any]],
        embeddings: List[List[float]],
    ) -> int:
        """
        Store document chunks with their embeddings and metadata.
        Prevents duplication by replacing any prior chunks for the same source document.
        Validates that incoming embeddings match the dimensionality of the collection.
        
        Args:
            chunks: List of chunk dictionaries containing text, metadata, and chunk_id.
            embeddings: List of embedding vectors corresponding to each chunk.
            
        Returns:
            Number of chunks successfully added.
        """
        if not chunks or not embeddings:
            return 0

        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Mismatch: received {len(chunks)} chunks and {len(embeddings)} embeddings."
            )

        new_dim = len(embeddings[0])

        # Verify dimensionality compatibility with existing chunks
        existing_dim = self.get_existing_dimension()
        if existing_dim is not None and existing_dim != new_dim:
            error_msg = (
                f"Incompatible embedding dimensionality: the existing knowledge base was indexed with "
                f"{existing_dim}-dimensional vectors, but the current model produces {new_dim}-dimensional "
                f"vectors. Mixing vectors from different embedding models is invalid. "
                f"Please click 'Clear Knowledge Base' in the sidebar or call reset_collection() before re-indexing."
            )
            logger.error(error_msg)
            raise IncompatibleEmbeddingDimensionError(error_msg)

        # Identify unique source documents to clean up any older versions
        sources_to_update = {c.get("source") for c in chunks if c.get("source")}
        for src in sources_to_update:
            self.delete_document(src)

        # Prepare lists for ChromaDB
        ids: List[str] = []
        documents: List[str] = []
        metadatas: List[Dict[str, Any]] = []

        for chunk in chunks:
            chunk_id = chunk.get("chunk_id")
            text = chunk.get("text", "")
            source = chunk.get("source", "unknown")
            page = chunk.get("page")
            chunk_index = chunk.get("chunk_index", 0)

            # ChromaDB requires metadata values to be str, int, float, or bool
            meta: Dict[str, Any] = {
                "source": str(source),
                "page": int(page) if page is not None and isinstance(page, int) else -1,
                "chunk_index": int(chunk_index),
            }

            ids.append(chunk_id)
            documents.append(text)
            metadatas.append(meta)

        # Add to ChromaDB in batches to prevent payload bottlenecks
        batch_size = 100
        for i in range(0, len(ids), batch_size):
            try:
                self.collection.upsert(
                    ids=ids[i : i + batch_size],
                    documents=documents[i : i + batch_size],
                    metadatas=metadatas[i : i + batch_size],
                    embeddings=embeddings[i : i + batch_size],
                )
            except Exception as e:
                err_str = str(e)
                if "dimension" in err_str.lower():
                    raise IncompatibleEmbeddingDimensionError(
                        f"ChromaDB dimension mismatch error: {err_str}. "
                        "The vector collection contains vectors from a different embedding model. "
                        "Please click 'Clear Knowledge Base' to reset and re-index."
                    ) from e
                raise

        logger.info("Successfully indexed %d chunks across %d document(s).", len(ids), len(sources_to_update))
        return len(ids)

    def similarity_search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Query ChromaDB for the most similar chunks.
        
        Args:
            query_embedding: Embedding vector of the query.
            top_k: Number of nearest neighbors to retrieve.
            where: Optional ChromaDB metadata filter.
            
        Returns:
            List of result dicts with chunk_id, text, metadata, distance, and similarity score.
        """
        total_available = self.count()
        if total_available == 0:
            return []

        # Validate query dimension matches existing collection
        existing_dim = self.get_existing_dimension()
        if existing_dim is not None and len(query_embedding) != existing_dim:
            raise IncompatibleEmbeddingDimensionError(
                f"Query embedding dimensionality ({len(query_embedding)}) does not match "
                f"collection dimensionality ({existing_dim}). Please reset the knowledge base."
            )

        # Clamp top_k to total available
        k = min(top_k, total_available)
        if k <= 0:
            return []

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        formatted_results: List[Dict[str, Any]] = []

        ids = results.get("ids", [[]])[0]
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for chunk_id, doc_text, meta, dist in zip(ids, docs, metas, distances):
            # Clean up page in metadata (-1 -> None)
            cleaned_meta = dict(meta) if meta else {}
            if cleaned_meta.get("page") == -1:
                cleaned_meta["page"] = None

            # With cosine distance: distance in [0, 2], similarity in [0, 1]
            similarity = max(0.0, min(1.0, 1.0 - dist))

            formatted_results.append({
                "chunk_id": chunk_id,
                "text": doc_text,
                "metadata": cleaned_meta,
                "distance": float(dist),
                "similarity": float(similarity),
            })

        return formatted_results

    def delete_document(self, source: str) -> int:
        """
        Delete all chunks associated with a specific document source.
        """
        try:
            existing = self.collection.get(where={"source": str(source)})
            existing_ids = existing.get("ids", [])
            if existing_ids:
                self.collection.delete(ids=existing_ids)
                logger.info("Deleted %d existing chunks for '%s'.", len(existing_ids), source)
                return len(existing_ids)
        except Exception as e:
            logger.warning("Failed to delete chunks for source '%s': %s", source, e)
        return 0

    def list_documents(self) -> List[Dict[str, Any]]:
        """
        Inspect all documents currently indexed in the vector store.
        
        Returns:
            List of document summaries:
            [{"source": str, "file_type": str, "chunk_count": int, "pages": list, "status": "Indexed"}]
        """
        total = self.count()
        if total == 0:
            return []

        # Retrieve all items (up to total)
        data = self.collection.get(include=["metadatas"])
        metadatas = data.get("metadatas", [])

        doc_summary: Dict[str, Dict[str, Any]] = {}

        for meta in metadatas:
            if not meta:
                continue
            src = meta.get("source", "unknown")
            page = meta.get("page")
            
            if src not in doc_summary:
                ext = Path(src).suffix.lower() or "unknown"
                doc_summary[src] = {
                    "source": src,
                    "file_type": ext.replace(".", "").upper(),
                    "chunk_count": 0,
                    "pages": set(),
                    "status": "Indexed",
                }

            doc_summary[src]["chunk_count"] += 1
            if page and page != -1:
                doc_summary[src]["pages"].add(page)

        # Convert sets to sorted lists
        result = []
        for src, info in doc_summary.items():
            info["pages"] = sorted(list(info["pages"]))
            result.append(info)

        return sorted(result, key=lambda x: x["source"].lower())

    def reset_collection(self) -> None:
        """
        Clear and reset the knowledge base collection.
        """
        try:
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info("Collection '%s' reset successfully.", self.collection_name)
        except Exception as e:
            logger.error("Error resetting collection: %s", e)
            raise
