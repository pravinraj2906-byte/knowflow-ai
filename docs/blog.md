# KnowFlow AI: Building a Grounded RAG-Based Knowledge Assistant

*By the Lead Developer — Week 2 AI Internship Project*

---

## 1. Introduction

During Week 2 of my AI Engineering internship, I built **KnowFlow AI**: a production-grade Retrieval-Augmented Generation (RAG) assistant designed to ingest multi-format organizational documents, index them into a persistent local vector database, and generate factually verified answers with granular source citations.

Building an AI assistant that answers questions is straightforward; building one that **refuses to hallucinate**, transparently proves its sources, and reliably processes multi-page documents under real-world network constraints requires disciplined systems engineering. This post details the architecture, technical hurdles, and solutions developed while building KnowFlow AI with Python, Streamlit, Google Gemini (`gemini-3.6-flash`), `gemini-embedding-2`, and ChromaDB.

---

## 2. The Problem: Information Fragmentation

In academic and enterprise environments, organizational knowledge is fragmented across multiple formats:
- Lengthy PDF guidelines, handbooks, and regulatory standards.
- Microsoft Word (`.docx`) requirement briefs and meeting summaries.
- Plain text (`.txt`) developer memos and project notes.

When teams need a concrete answer—such as a submission deadline or technical parameter—they must manually search through dozens of pages or rely on keyword search (Ctrl+F). Keyword search fails whenever queries involve synonyms or conceptual phrasing rather than exact substring matches.

---

## 3. Why Hallucination Is a Critical Problem

When standard Large Language Models (LLMs) generate responses without contextual grounding, they rely exclusively on internal parametric weights. When queried on proprietary documentation, standard LLMs encounter severe failure modes:
1. **Plausible Fabrication**: LLMs predict probable token sequences. When lacking specific data, they generate plausible-sounding falsehoods with high confidence.
2. **Fabricated Citations**: Users asking for proof frequently receive hallucinated page numbers, fake titles, or non-existent authors.
3. **Absence of Fallbacks**: General models rarely state *"I do not know"* if they can predict a plausible response.

In compliance, research, or engineering workflows, an ungrounded hallucination introduces serious risk.

---

## 4. The RAG Architecture

**Retrieval-Augmented Generation (RAG)** addresses parametric hallucination by decoupling knowledge retrieval from generative reasoning. Rather than expecting the LLM to memorize proprietary facts, RAG dynamically retrieves pertinent passages from a verified corpus and provides them as contextual grounding for generation.

```
[Uploaded Files: PDF, DOCX, TXT]
               │
               ▼
      [Document Extractor] ──► Page Numbers (PDF) & Structural Text
               │
               ▼
      [Paragraph Chunker]  ──► Overlapping Windows & Stable SHA-256 IDs
               │
               ▼
    [gemini-embedding-2]   ──► 1-to-1 Vector Generation (Batched)
               │
               ▼
     [ChromaDB Vector DB]  ──► Local Persistent Storage (data/chroma/)
               ▲
               │ (Cosine Distance Search)
               │
User Question ─┴─► [Query Vector] ──► [Distance Threshold Gate (< 0.70)]
                                                  │
                      ┌───────────────────────────┴───────────────────────────┐
                      ▼                                                       ▼
          [Context Insufficient]                                     [Context Verified]
                      │                                                       │
                      ▼                                                       ▼
        "I couldn't find enough..."                                  [gemini-3.6-flash]
                                                                    (temperature = 0.0)
                                                                              │
                                                                              ▼
                                                                Grounded Answer + Citations
```

---

## 5. KnowFlow AI Technical Workflow

### Document Ingestion & Extraction
Document ingestion is handled by `src/ingestion/extractor.py`:
- **PDF Extraction**: Uses `PyMuPDF` (`pymupdf`) to iterate through pages, extracting text while maintaining 1-indexed page metadata.
- **DOCX Extraction**: Uses `python-docx` to iterate through paragraphs and table cells, concatenating structured tabular data.
- **TXT Extraction**: Decodes UTF-8 with automatic fallback to Latin-1 and CP1252.

Extraction errors are handled gracefully without aborting the application.

### Chunking & Deterministic Identifiers
In `src/ingestion/chunker.py`, text is split using a paragraph-aware windowing algorithm:
- **Chunk Size & Overlap**: 600 characters with 120-character overlap, preserving paragraph boundaries (`\n\n`).
- **Minimum Filter**: Chunks smaller than 30 characters are pruned to eliminate trivial whitespace.
- **Deterministic Hashing**: Every chunk receives an ID derived from `SHA-256(source + page + chunk_index + content)`. Re-indexing a file produces identical IDs, preventing duplicate entries in ChromaDB.

### Embeddings with `gemini-embedding-2`
In `src/rag/embeddings.py`, vectors are generated using Google's modern `gemini-embedding-2` model via the official `google-genai` SDK (`from google import genai`). Both document chunks and user queries are embedded using this same model without legacy task types, guaranteeing that queries and candidate passages reside in the identical geometric vector space.

### Persistent Vector Storage in ChromaDB
KnowFlow AI employs **ChromaDB** (`src/rag/vector_store.py`) configured with persistent local storage under `data/chroma/` (outside `.venv`). Cosine distance is used as the distance metric. To ensure idempotence, `VectorStore` automatically removes previous chunks belonging to an updated filename before upserting new vectors.

### Semantic Retrieval & Hallucination Control
In `src/rag/retriever.py`, user queries are embedded and compared against the ChromaDB index. To prevent the assistant from hallucinating when asked unanswerable questions, KnowFlow AI enforces a **Relevance Distance Threshold**:
- If the best-matching chunk has a cosine distance $> 0.70$ (similarity $< 0.30$) or if the database is empty, the retriever flags context as insufficient.
- The pipeline immediately halts and returns the verified fallback message:
  > *"I couldn't find enough information in the uploaded documents to answer this question."*

### Grounded Gemini Generation & Citations
When relevant chunks are retrieved, `src/rag/generator.py` constructs a strict prompt incorporating document titles, page numbers, and passage excerpts. The prompt is dispatched to `gemini-3.6-flash` configured with `temperature=0.0`:
- **Deterministic Grounding**: The system instruction explicitly prohibits drawing on outside training knowledge.
- **Granular Attribution**: Supporting documents are parsed into citation pills (e.g., `project_guidelines.pdf — Page 4`, `handbook.docx`). Page numbers are never fabricated for formats where pagination does not exist.

---

## 6. User Interface

Built with **Streamlit** in `app.py`, the user interface prioritizes clarity and operational transparency:
- **Branded Sidebar**: Features multi-file uploaders, knowledge base statistics (document counts, total chunks), and a single-click "Clear Knowledge Base" button.
- **Conversational Feed**: Built using `st.chat_message` and `st.chat_input`, preserving chat history across the session.
- **Evidence Inspection Cards**: Every assistant response includes an expandable "View Retrieved Context Evidence" section showing raw text snippets, similarity metrics, and source page numbers.
- **Persistent Error Alerts**: Ingestion errors are captured in session state and displayed with actionable recovery advice rather than vanishing upon page reruns.

---

## 7. Testing & Quality Assurance

In `tests/test_rag.py`, a comprehensive test suite was developed comprising **26 automated pytest tests**:
- **Chunking & Boundary Tests**: Verifies chunk sizes, overlap windows, and paragraph preservation.
- **Metadata Integrity**: Confirms that page numbers and filenames persist across extracted chunks.
- **Multi-Format Parsers**: Tests real in-memory PDF, DOCX, and TXT streams without filesystem dependencies.
- **Deterministic Hashing**: Asserts that identical content produces identical chunk IDs.
- **Empty Document Handling**: Validates that empty files or whitespace strings produce zero chunks without errors.
- **Hallucination Fallback Verification**: Tests that queries exceeding the distance threshold trigger the fallback message.
- **Offline Mocking**: External Gemini API calls are mocked using `unittest.mock`, ensuring tests run 100% offline, reliably, and without consuming API quota.

---

## 8. Technical Challenges Encountered & Engineering Solutions

During development and end-to-end testing with real-world documents—specifically a 14-page document producing 58 chunks (`Pathway PS (1).pdf`)—we solved three critical engineering challenges:

### Challenge 1: The `task_type` Incompatibility in `gemini-embedding-2`
- **Issue**: Our initial implementation passed `types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")`, which functioned under legacy models like `text-embedding-004`. However, `gemini-embedding-2` rejected this parameter, causing the backend API to terminate connections unexpectedly.
- **Solution**: We removed `task_type` entirely from both chunk and query embeddings. Both now call `embed_content` directly, aligning them into the identical vector space.

### Challenge 2: SDK Content List Aggregation (`t.t_contents`)
- **Issue**: When passing a list of raw strings (`contents = ["chunk1", "chunk2", ...]`) to the `google-genai` SDK, the internal transformer `t.t_contents()` automatically merged all consecutive strings into a single `UserContent` object with multiple parts. This caused the API to treat the batch as one giant document and calculate only one composite embedding for all 50 chunks.
- **Solution**: We wrapped each text chunk as an explicit `types.Content(parts=[types.Part.from_text(text=t)])`. This prevented SDK aggregation, guaranteeing a strict 1-to-1 chunk-to-embedding relationship.

### Challenge 3: Socket Dropouts and TLS Protocol Violations
- **Issue**: Uploading 58 chunks in a single large batch of 50 chunks caused socket read timeouts and `[SSL: UNEXPECTED_EOF_WHILE_READING]` errors. Once an SSL socket dropped, Python's persistent connection pool retained the damaged connection state.
- **Solution**: We reduced the batch size to **10 chunks per request** and introduced an **exponential backoff retry mechanism** (up to 3 attempts with random jitter). Crucially, upon detecting a connection drop, the embedder automatically resets its internal client (`self._client = None`), forcing a fresh TLS handshake on the retry.

---

## 9. Results & Key Takeaways

The final implementation achieved all core objectives:
- **100% Passing Test Suite**: 26 unit and integration tests executing cleanly in under 2.5 seconds.
- **Flawless Ingestion**: Verified on multi-page PDFs, DOCX tables, and text files.
- **Zero Hallucination Leaks**: Unanswerable queries reliably trigger the grounded fallback response.
- **Persistent Local Database**: ChromaDB retains indexed knowledge across application restarts without cloud lock-in.

---

## 10. Future Improvements

While KnowFlow AI is ready for deployment, several extensions remain on the roadmap:
1. **Optical Character Recognition (OCR)**: Integrating Tesseract or Surya OCR to extract text from scanned documents.
2. **Hybrid Retrieval**: Merging BM25 lexical keyword matching with dense vector search via Reciprocal Rank Fusion (RRF).
3. **Cross-Encoder Reranking**: Introducing a secondary reranking model to refine Top-K retrieved passages before generation.
4. **Interactive PDF Highlights**: Rendering PDF pages directly in the UI with bounding boxes highlighting the exact sentence cited.

---

## 11. Conclusion

Building **KnowFlow AI** demonstrated that robust RAG is not simply about connecting an API key to an LLM. Production-ready systems require meticulous attention to text hygiene, chunking strategies, vector dimensionality safeguards, and resilient network error handling. By combining ChromaDB's local persistence with Google Gemini's grounded reasoning, KnowFlow AI delivers a reliable, verifiable, and transparent document knowledge assistant.
