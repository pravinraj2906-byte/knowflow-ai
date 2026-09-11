"""
Chunking module for KnowFlow AI.
Cleans unnecessary whitespace, preserves paragraph boundaries, and splits
documents into overlapping chunks with stable IDs and attached metadata.
"""

import logging
from typing import Any, Dict, List, Optional

from src.utils.helpers import generate_chunk_id, sanitize_text

logger = logging.getLogger(__name__)

# Sensible default parameters
DEFAULT_CHUNK_SIZE = 600
DEFAULT_CHUNK_OVERLAP = 120
DEFAULT_MIN_CHUNK_SIZE = 30


def split_text_with_overlap(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    min_chunk_size: int = DEFAULT_MIN_CHUNK_SIZE,
) -> List[str]:
    """
    Split text into chunks of approximately `chunk_size` characters with `chunk_overlap`.
    Preserves paragraph breaks (\n\n) and sentence boundaries where possible.
    Filters out chunks smaller than `min_chunk_size`.
    """
    if not text:
        return []

    # If the total text fits comfortably within chunk_size, return directly
    if len(text) <= chunk_size:
        return [text] if len(text) >= min_chunk_size else []

    # First split into paragraphs
    paragraphs = text.split("\n\n")
    chunks: List[str] = []
    current_chunk: List[str] = []
    current_len = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # If a single paragraph exceeds chunk_size, split by sentences or space
        if len(para) > chunk_size:
            # If we had buffered content, flush it
            if current_chunk:
                combined = "\n\n".join(current_chunk)
                if len(combined) >= min_chunk_size:
                    chunks.append(combined)
                current_chunk = []
                current_len = 0

            # Split large paragraph by sentences
            words = para.split(" ")
            sub_chunk: List[str] = []
            sub_len = 0

            for word in words:
                word_len = len(word) + 1  # count space
                if sub_len + word_len > chunk_size and sub_chunk:
                    chunk_str = " ".join(sub_chunk)
                    if len(chunk_str) >= min_chunk_size:
                        chunks.append(chunk_str)
                    
                    # Compute overlap: take trailing words that fit inside chunk_overlap
                    overlap_words = []
                    acc = 0
                    for w in reversed(sub_chunk):
                        if acc + len(w) + 1 <= chunk_overlap:
                            overlap_words.insert(0, w)
                            acc += len(w) + 1
                        else:
                            break
                    sub_chunk = overlap_words
                    sub_len = sum(len(w) + 1 for w in sub_chunk)

                sub_chunk.append(word)
                sub_len += word_len

            if sub_chunk:
                chunk_str = " ".join(sub_chunk)
                if len(chunk_str) >= min_chunk_size:
                    chunks.append(chunk_str)

        else:
            # Check if adding this paragraph exceeds chunk_size
            para_len = len(para) + (2 if current_chunk else 0)
            if current_len + para_len > chunk_size and current_chunk:
                combined = "\n\n".join(current_chunk)
                if len(combined) >= min_chunk_size:
                    chunks.append(combined)

                # Overlap: keep the last paragraph if it fits in chunk_overlap
                last_para = current_chunk[-1]
                if len(last_para) <= chunk_overlap:
                    current_chunk = [last_para, para]
                    current_len = len(last_para) + 2 + len(para)
                else:
                    current_chunk = [para]
                    current_len = len(para)
            else:
                current_chunk.append(para)
                current_len += para_len

    # Flush any remaining buffer
    if current_chunk:
        combined = "\n\n".join(current_chunk)
        if len(combined) >= min_chunk_size:
            chunks.append(combined)

    return chunks


def chunk_text(
    text: str,
    source: str,
    page: Optional[int] = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    min_chunk_size: int = DEFAULT_MIN_CHUNK_SIZE,
    start_chunk_idx: int = 0,
) -> List[Dict[str, Any]]:
    """
    Chunk a text block and attach full metadata and deterministic chunk IDs.
    """
    cleaned_text = sanitize_text(text)
    if not cleaned_text:
        return []

    raw_chunks = split_text_with_overlap(
        cleaned_text,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        min_chunk_size=min_chunk_size,
    )

    structured_chunks: List[Dict[str, Any]] = []
    for idx, chunk in enumerate(raw_chunks, start=start_chunk_idx):
        chunk_id = generate_chunk_id(source, page, idx, chunk)
        structured_chunks.append({
            "chunk_id": chunk_id,
            "text": chunk,
            "source": source,
            "page": page,
            "chunk_index": idx,
        })

    return structured_chunks


def chunk_documents(
    extracted_docs: List[Dict[str, Any]],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    min_chunk_size: int = DEFAULT_MIN_CHUNK_SIZE,
) -> List[Dict[str, Any]]:
    """
    Process a list of extracted document items and split into chunks.
    
    Args:
        extracted_docs: List of dicts, each with keys {"text", "source", "page"}
        chunk_size: Maximum chunk size in characters
        chunk_overlap: Overlap length in characters
        min_chunk_size: Minimum chunk size to retain
        
    Returns:
        Flat list of structured chunks with metadata and stable IDs.
    """
    all_chunks: List[Dict[str, Any]] = []
    chunk_counter = 0

    for doc in extracted_docs:
        text = doc.get("text", "")
        source = doc.get("source", "document")
        page = doc.get("page")

        chunks = chunk_text(
            text=text,
            source=source,
            page=page,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            min_chunk_size=min_chunk_size,
            start_chunk_idx=chunk_counter,
        )
        all_chunks.extend(chunks)
        chunk_counter += len(chunks)

    return all_chunks
