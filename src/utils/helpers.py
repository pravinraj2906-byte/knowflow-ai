"""
Helper utilities for configuration, text hygiene, chunk identification,
and source formatting across KnowFlow AI.
"""

import hashlib
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional
from dotenv import load_dotenv

# Automatically load .env file from project root if present
load_dotenv()


def load_api_key() -> Optional[str]:
    """
    Safely retrieve the Gemini API key from environment variables or .env.
    
    Returns:
        The valid API key string, or None if missing or unset/placeholder.
    """
    # Reload in case .env was updated at runtime
    load_dotenv(override=False)
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    
    # Check for empty or placeholder values
    if not api_key or api_key in ("your_api_key_here", "YOUR_API_KEY_HERE", "your_key"):
        return None
    return api_key


def clean_filename(filename: str) -> str:
    """
    Sanitizes a filename to prevent path traversal or unwanted characters.
    """
    # Keep only the base filename
    base = Path(filename).name
    # Remove potentially dangerous characters
    cleaned = re.sub(r'[\\/*?:"<>|]', "", base).strip()
    return cleaned or "document"


def sanitize_text(text: str) -> str:
    """
    Clean raw extracted text while preserving meaningful structure:
    - Normalizes diverse newline formats (\r\n -> \n)
    - Replaces consecutive blank lines with a double newline (paragraph boundary)
    - Collapses horizontal spaces and tabs into single spaces
    - Strips leading and trailing whitespace
    """
    if not text:
        return ""

    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Normalize unicode spaces and remove null bytes
    text = text.replace("\x00", "")
    text = re.sub(r"[\t\f\v ]+", " ", text)

    # Preserve paragraph breaks: collapse 3+ newlines into 2
    paragraphs = text.split("\n\n")
    cleaned_paras = []
    for para in paragraphs:
        # Join wrapped single lines within a paragraph with single space
        lines = [line.strip() for line in para.split("\n") if line.strip()]
        if lines:
            cleaned_paras.append(" ".join(lines))

    return "\n\n".join(cleaned_paras).strip()


def generate_chunk_id(source: str, page: Optional[int], chunk_idx: int, content: str) -> str:
    """
    Generate a deterministic, stable chunk ID using a SHA-256 hash.
    
    This ensures that re-indexing the same file content produces identical IDs,
    preventing duplicate chunk inserts.
    """
    page_str = f"p{page}" if page is not None else "p0"
    base_id = f"{source}_{page_str}_c{chunk_idx}"
    content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()[:10]
    return f"{base_id}_{content_hash}"


def format_source_citation(metadata: Dict[str, Any]) -> str:
    """
    Format a clean, readable citation label from chunk metadata.
    
    Rules:
    - If page number is available and valid (> 0): 'filename.pdf — Page X'
    - Otherwise (e.g. TXT, DOCX): 'filename.ext'
    """
    source = metadata.get("source", "Unknown Source")
    page = metadata.get("page")
    
    # Page must be integer and positive
    if page is not None and isinstance(page, int) and page > 0:
        return f"{source} — Page {page}"
    return str(source)
