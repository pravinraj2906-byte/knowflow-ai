"""
Grounded response generator for KnowFlow AI using Google GenAI SDK.
Enforces strict grounding, source extraction, and hallucination prevention.
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from google import genai
from google.genai import types

from src.rag.retriever import INSUFFICIENT_CONTEXT_MESSAGE
from src.utils.helpers import format_source_citation, load_api_key

logger = logging.getLogger(__name__)

DEFAULT_GENERATION_MODEL = "gemini-3.6-flash"

SYSTEM_INSTRUCTION = """You are KnowFlow AI, an intelligent, strictly grounded document assistant.
Your job is to answer the user's question using ONLY the provided context from the uploaded documents.

CRITICAL RULES:
1. Rely EXCLUSIVELY on the facts explicitly stated in the context.
2. Do NOT extrapolate, speculate, or bring in outside knowledge.
3. If the provided context does not contain enough information to answer the question, you MUST respond with EXACTLY:
   "I couldn't find enough information in the uploaded documents to answer this question."
4. Do NOT invent facts, statistics, page numbers, or citations.
5. Keep your answer clear, direct, and factual. Mention nuance or uncertainty when the context indicates it.
"""


@dataclass
class GenerationResult:
    """Structured response containing generated answer, sources, and metadata."""
    answer: str
    sources: List[str] = field(default_factory=list)
    retrieved_chunks: List[Dict[str, Any]] = field(default_factory=list)
    model_used: str = DEFAULT_GENERATION_MODEL
    is_fallback: bool = False


class GroundedGenerator:
    """
    Orchestrates prompt creation, Gemini model inference with temperature=0.0,
    and extraction of precise source references.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        self.api_key = api_key or load_api_key()
        self.model_name = (
            model_name
            or os.getenv("GEMINI_MODEL")
            or DEFAULT_GENERATION_MODEL
        )
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

    @staticmethod
    def extract_sources(chunks: List[Dict[str, Any]]) -> List[str]:
        """
        Deduplicate and format source citations from retrieved chunks.
        
        Example output:
        - project_guidelines.pdf — Page 4
        - handbook.docx
        """
        seen = set()
        citations = []
        for chunk in chunks:
            metadata = chunk.get("metadata", {})
            citation = format_source_citation(metadata)
            if citation not in seen:
                seen.add(citation)
                citations.append(citation)
        return citations

    @staticmethod
    def format_context(chunks: List[Dict[str, Any]]) -> str:
        """
        Format retrieved chunks into a clear, labeled context block.
        """
        formatted_blocks = []
        for idx, chunk in enumerate(chunks, start=1):
            metadata = chunk.get("metadata", {})
            source = metadata.get("source", "document")
            page = metadata.get("page")
            
            page_label = f" | Page: {page}" if page is not None and page > 0 else ""
            header = f"--- Document Section {idx} [Source: {source}{page_label}] ---"
            content = chunk.get("text", "").strip()
            formatted_blocks.append(f"{header}\n{content}")

        return "\n\n".join(formatted_blocks)

    def generate_answer(
        self,
        question: str,
        retrieved_chunks: List[Dict[str, Any]],
        force_fallback: bool = False,
    ) -> GenerationResult:
        """
        Generate a strictly grounded response to the question.
        
        Args:
            question: The user's query.
            retrieved_chunks: List of relevant chunks retrieved from vector store.
            force_fallback: If True (e.g. relevance threshold failed), bypass LLM and return fallback.
            
        Returns:
            GenerationResult with grounded answer, sources, and metadata.
        """
        # Hallucination control check
        if force_fallback or not retrieved_chunks:
            return GenerationResult(
                answer=INSUFFICIENT_CONTEXT_MESSAGE,
                sources=[],
                retrieved_chunks=[],
                model_used=self.model_name,
                is_fallback=True,
            )

        context_str = self.format_context(retrieved_chunks)
        sources = self.extract_sources(retrieved_chunks)

        user_prompt = f"""Context from uploaded documents:
========================================
{context_str}
========================================

Question: {question}

Answer based ONLY on the context above. If the context does not contain the answer, reply:
"{INSUFFICIENT_CONTEXT_MESSAGE}"
"""

        try:
            config = types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                temperature=0.0,  # Zero temperature for deterministic grounding
            )

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=user_prompt,
                config=config,
            )

            raw_answer = (response.text or "").strip()
            if not raw_answer:
                raw_answer = INSUFFICIENT_CONTEXT_MESSAGE

            is_fallback = INSUFFICIENT_CONTEXT_MESSAGE.lower() in raw_answer.lower()

            return GenerationResult(
                answer=raw_answer,
                sources=[] if is_fallback else sources,
                retrieved_chunks=retrieved_chunks,
                model_used=self.model_name,
                is_fallback=is_fallback,
            )

        except Exception as e:
            logger.error("Error generating answer from Gemini: %s", e)
            return GenerationResult(
                answer=f"An error occurred while generating the answer: {str(e)}",
                sources=sources,
                retrieved_chunks=retrieved_chunks,
                model_used=self.model_name,
                is_fallback=True,
            )
