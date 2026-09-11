# KnowFlow AI: Concept Summary

## 1. Problem Statement

In academic, clinical, and corporate environments, critical institutional knowledge is fragmented across heterogeneous documents—PDF reports, Word specifications, and plain text notes. Professionals and students face significant friction when extracting insights from these files:

- **Keyword Search Inefficiency**: Standard keyword searches (such as Ctrl+F) match literal character sequences, failing to recognize synonyms, conceptual relationships, or semantic intent.
- **Cross-Document Dispersal**: Information required to resolve a single inquiry is frequently scattered across distinct files and pages, requiring laborious manual cross-referencing.
- **LLM Hallucination Risk**: Off-the-shelf Large Language Models (LLMs) operate purely on parametric memory derived during pretraining. When queried about specialized, private, or current documents, standard LLMs routinely fabricate plausible-sounding yet completely fictitious facts (hallucinations), invent nonexistent citations, and answer with unfounded confidence.

---

## 2. Retrieval-Augmented Generation (RAG) & KnowFlow AI's Solution

**Retrieval-Augmented Generation (RAG)** resolves parametric hallucination by separating knowledge storage from generative reasoning. Rather than expecting the language model to memorize proprietary information, RAG dynamically retrieves pertinent passages from a verified corpus and provides them as contextual grounding for generation.

**KnowFlow AI** is a specialized, production-oriented RAG knowledge assistant designed to deliver factual, verifiable answers strictly grounded in user-provided documents.

```
Uploaded Documents (PDF / DOCX / TXT)
                  │
                  ▼
         [Document Ingestion]
                  │
                  ▼
         [Text Extraction]
                  │
                  ▼
     [Paragraph-Aware Chunking] ──► Stable SHA-256 Chunk IDs
                  │
                  ▼
       [gemini-embedding-2] ──────► 1-to-1 Content Embeddings
                  │
                  ▼
        [ChromaDB Vector Store]
                  ▲
                  │  (Cosine Similarity Search)
                  │
   User Question ─┴─► [Query Embedding] ──► [Distance Threshold Filter]
                                                    │
                             ┌──────────────────────┴──────────────────────┐
                             ▼                                             ▼
                 [Insufficient Context]                           [Relevant Context]
                             │                                             │
                             ▼                                             ▼
                 "I couldn't find enough..."                     [gemini-3.6-flash]
                                                               (temperature = 0.0)
                                                                           │
                                                                           ▼
                                                             Grounded Answer + Citations
```

---

## 3. End-to-End Technical Workflow

### Document Ingestion & Extraction
KnowFlow AI accepts three foundational document formats via a Streamlit interface:
- **PDF Documents**: Parsed page-by-page using `PyMuPDF` (`pymupdf`), capturing raw text while tracking exact 1-indexed page numbers.
- **DOCX Documents**: Parsed using `python-docx`, extracting structural paragraphs and table cells into plain text.
- **TXT Files**: Decoded with fallback encoding handling (UTF-8, Latin-1, CP1252).
Extraction errors are captured gracefully, preventing an unreadable document from crashing the pipeline.

### Paragraph-Aware Chunking
Raw text is cleaned (whitespace normalized, null bytes stripped) and split into manageable, overlapping windows (default size: 600 characters, overlap: 120 characters, minimum threshold: 30 characters). Natural paragraph boundaries (`\n\n`) are preserved. Every chunk is tagged with metadata:
- Source filename
- Page number (or `None` for DOCX/TXT)
- Chunk index
- Deterministic chunk ID generated via a SHA-256 hash of source, page, index, and content to guarantee idempotence.

### Vector Embeddings with `gemini-embedding-2`
Document chunks and user questions are converted into vector representations using Google's modern `gemini-embedding-2` model via the official `google-genai` SDK. 
- **1-to-1 Explicit Content Mapping**: Each chunk is wrapped as an explicit `types.Content` object, preventing the SDK from collapsing multi-chunk batches into a single composite representation.
- **Batched Execution**: Requests are processed in safe batches of 10 chunks with exponential backoff retries and TLS client refreshes to eliminate socket drops.
- **Unified Vector Space**: Neither chunks nor queries pass legacy `task_type` parameters, ensuring unified embedding space alignment.

### Persistent Vector Storage with ChromaDB
Vector embeddings, metadata, chunk texts, and deterministic IDs are stored in a local, persistent **ChromaDB** collection (`data/chroma/`) outside `.venv`. 
- **Deduplication**: Re-uploading a document replaces prior chunks, eliminating duplicates.
- **Dimensionality Safety**: Vector dimension compatibility is validated prior to insertion, preventing corruption from mixed embedding models.

### Semantic Retrieval & Hallucination Prevention
Upon receiving a user query:
1. The query is embedded via `gemini-embedding-2`.
2. ChromaDB evaluates the nearest neighbors using cosine distance.
3. **Hallucination Distance Threshold**: Chunks with cosine distance exceeding `0.70` (similarity $< 0.30$) are filtered out. If no retrieved chunks satisfy this threshold or the collection is empty, generation is halted immediately, returning the exact verified fallback:
   > *"I couldn't find enough information in the uploaded documents to answer this question."*

### Grounded Generation with `gemini-3.6-flash`
When relevant context exists, it is labeled by document and page number and passed to Google Gemini (`gemini-3.6-flash`) with strict grounding instructions and `temperature=0.0`. Gemini is directed to answer using exclusively the supplied passages.

### Granular Source Attribution
Generated answers display explicit supporting source tags:
- `document.pdf — Page 4`
- `handbook.docx`
Page numbers are never fabricated. Expandable evidence cards allow users to review the exact snippets and similarity scores.

---

## 4. Benefits and Limitations

### Key Benefits
- **Strict Factual Reliability**: Elimination of parametric hallucinations through vector thresholding and zero-temperature grounding.
- **Auditability**: Transparent source citations allow users to verify answers directly against original pages.
- **Local Persistence & Privacy**: Raw vectors and ChromaDB databases remain strictly on the local machine.
- **Resilient Pipeline**: Batched ingestion and exponential backoff prevent network/TLS failures.

### Current Limitations
- **Text-Only Extraction**: Scanned documents containing images or rasterized text require optical character recognition (OCR) before ingestion.
- **Single-Machine Scale**: Local ChromaDB is suitable for thousands of chunks; enterprise multi-terabyte corpora require distributed vector search.

---

## 5. Future Improvements

1. **OCR Integration**: Incorporating Tesseract or Surya OCR for scanned PDFs.
2. **Hybrid Search**: Combining BM25 keyword matching with dense vector retrieval using Reciprocal Rank Fusion (RRF).
3. **Cross-Encoder Reranking**: Reordering Top-K candidates via cross-encoders before prompt injection.
4. **Bounding-Box Highlights**: Directly linking citations to visual bounding boxes on original PDF pages.
