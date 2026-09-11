# KnowFlow AI

> **RAG-Based Knowledge Assistant**  
> An intelligent, hallucination-resistant knowledge assistant that ingests multi-format documents, performs semantic vector retrieval with ChromaDB, and generates strictly grounded responses using Google Gemini.

---

## Problem Statement

In modern academic and enterprise environments, critical information is scattered across fragmented documents—PDF reports, Word specifications, and plain text notes. Traditional keyword search (Ctrl+F) suffers from critical limitations:
- It fails to capture semantic meaning or synonyms.
- It cannot aggregate information dispersed across multiple pages or separate files.
- Users waste valuable time manually reading through dense documentation to find answers to specific questions.

Furthermore, standard Large Language Models (LLMs) used in isolation suffer from **hallucinations**—they may authoritatively state incorrect facts, speculate on missing data, or lack access to private, unpublished organizational documents.

---

## Solution

**KnowFlow AI** implements a production-grade **Retrieval-Augmented Generation (RAG)** pipeline:
1. **Multi-Document Ingestion**: Parses PDF, DOCX, and TXT files, extracting raw text while preserving essential metadata such as page numbers and document names.
2. **Deterministic Chunking & Embedding**: Splits documents into overlapping segments with paragraph preservation, transforming them into high-dimensional vector representations via Google's modern `gemini-embedding-2` model.
3. **Persistent Vector Storage**: Stores chunks in a local, persistent **ChromaDB** vector store outside `.venv` with deterministic hashing to eliminate duplicates.
4. **Grounded Semantic Retrieval & Hallucination Guardrails**: Evaluates question relevance against a distance threshold. If the uploaded files lack sufficient context, KnowFlow AI refuses to fabricate answers.
5. **Grounded Gemini Generation**: Employs Google Gemini (`gemini-3.6-flash`) at `temperature=0.0` with explicit grounding prompts and granular source attribution.

---

## Week 2 Submission Materials

Comprehensive project documentation and technical records prepared for the Week 2 internship review:
- 📄 **[Concept Summary](docs/concept_summary.md)** — Problem statement, RAG pipeline mechanics, and grounded generation breakdown.
- ✍️ **[Technical Blog](docs/blog.md)** — *"KnowFlow AI: Building a Grounded RAG-Based Knowledge Assistant"* (Architecture deep-dive, technical challenges, and solutions).
- 🛠️ **[Setup Instructions](docs/setup_instructions.md)** — Step-by-step local environment configuration, testing, and troubleshooting guide.
- ⚖️ **[Sources and Licenses](docs/sources_and_licenses.md)** — Registry of third-party libraries, foundation models, and licensing compliance.

---

## Features

- **Multi-Format Document Support**: Seamlessly processes PDF (with page-level tracking), Word (`.docx`), and plain text (`.txt`) documents.
- **Multi-Document Knowledge Base**: Index and query multiple files simultaneously within a unified vector store.
- **Semantic Vector Search**: Powered by ChromaDB using cosine distance metrics for accurate contextual matching.
- **Gemini-Powered Grounded Answers**: Answers generated using Google's state-of-the-art Gemini models with strict grounding instructions.
- **ChromaDB Vector Database**: Local persistent storage with zero cloud dependencies for embeddings.
- **Granular Source Attribution**: Every response displays exact supporting citations (e.g., `project_guidelines.pdf — Page 4`, `handbook.docx`).
- **Strict Hallucination Control**: Configurable similarity/distance thresholding ensures the assistant returns a verified fallback message when context is insufficient:
  > *"I couldn't find enough information in the uploaded documents to answer this question."*
- **Polished Streamlit Chat Experience**: Clean conversational interface featuring chat history, document management statistics, evidence inspection cards, and single-click knowledge base reset.

---

## Architecture

```
User
 ↓
Streamlit UI
 ↓
Document Ingestion
 ↓
Text Extraction
 ↓
Chunking
 ↓
Embeddings
 ↓
ChromaDB
 ↓
User Question
 ↓
Query Embedding
 ↓
Similarity Search
 ↓
Relevant Context
 ↓
Gemini
 ↓
Grounded Answer + Sources
```

---

## Tech Stack

- **Python**: Core programming language for application logic and RAG pipeline.
- **Streamlit**: Modern interactive web framework powering the user interface and chat experience.
- **Google Gemini API**: State-of-the-art generative AI model (`gemini-3.6-flash`) for grounded reasoning and synthesis.
- **google-genai SDK**: Official modern Google GenAI Python SDK (`from google import genai`).
- **ChromaDB**: High-performance local vector database for persistent document chunk embeddings.
- **PyMuPDF (`pymupdf`)**: High-speed, robust PDF parsing engine preserving page boundaries.
- **python-docx**: Library for extracting text, paragraphs, and tables from Microsoft Word documents.
- **NumPy**: Numerical operations and vector processing.
- **python-dotenv**: Environment variable management for secure API key loading.
- **pytest**: Test automation framework for unit and integration testing.

---

## Project Structure

```
KNOWFLOWAI/
├── .env                  # Environment configuration (API key)
├── .gitignore            # Git exclusion rules
├── .venv/                # Virtual environment (managed)
├── app.py                # Streamlit web application
├── pytest.ini            # Pytest configuration
├── README.md             # Project documentation
├── requirements.txt      # Project dependencies
│
├── docs/
│   ├── concept_summary.md    # Conceptual summary & RAG architecture breakdown
│   ├── blog.md               # Technical internship blog & engineering deep dive
│   ├── setup_instructions.md # Local environment & troubleshooting guide
│   └── sources_and_licenses.md # Dependency, model & license registry
│
├── src/
│   ├── __init__.py
│   │
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── extractor.py   # Multi-format document parser (PDF/DOCX/TXT)
│   │   └── chunker.py     # Text cleaning, boundary preservation & stable chunk IDs
│   │
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── embeddings.py  # Google GenAI gemini-embedding-2 wrapper
│   │   ├── vector_store.py# ChromaDB persistent store & deduplication
│   │   ├── retriever.py   # Semantic search & relevance thresholding
│   │   └── generator.py   # Grounded Gemini answer generator & citations
│   │
│   └── utils/
│       ├── __init__.py
│       └── helpers.py     # API key loader, sanitization, hashing, citation formatters
│
├── tests/
│   └── test_rag.py       # Unit and integration test suite (26 tests)
│
└── data/
    └── chroma/           # Persistent ChromaDB vector store directory (auto-created)
```

---

## Installation

### 1. Create and activate virtual environment

```bash
python -m venv .venv
```

**Windows activation:**
```powershell
.venv\Scripts\activate
```

**macOS / Linux activation:**
```bash
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

Create a `.env` file in the root directory:

```env
GEMINI_API_KEY=your_key_here
GEMINI_MODEL=gemini-3.6-flash
EMBEDDING_MODEL=gemini-embedding-2
```

> **Note**: Replace `your_key_here` with a valid Google Gemini API key obtained from [Google AI Studio](https://aistudio.google.com/).

### 4. Run the application

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

---

## Testing

Execute the automated pytest suite:

```bash
pytest
```

For verbose output:

```bash
pytest -v
```

All 26 tests run fully offline with mocked external API calls, testing:
- Text chunking boundaries, minimum chunk filters, and overlap.
- Metadata preservation across PDF pages and DOCX/TXT sources.
- Multi-format file parsing (PDF, DOCX, TXT, unsupported formats).
- Deterministic chunk ID generation.
- Empty document and whitespace handling.
- Hallucination control and distance threshold fallback triggers.
- Vector store persistence, document deduplication, and collection reset.

---

## Security

- **API Keys**: API keys are loaded exclusively from `.env` or current session state and are **never** hardcoded in source code or logged.
- **Git Exclusion**: `.env`, `.venv/`, `data/chroma/`, and cache directories are explicitly excluded in `.gitignore` to prevent leaking credentials or private document vectors.
- **Local Persistence**: Vector data is stored strictly on your local filesystem in `data/chroma/` without third-party cloud data transmission.

---

## Future Improvements

- **OCR Integration**: Add Tesseract / Surya OCR to extract text from scanned PDFs and images.
- **Hybrid Search**: Combine BM25 keyword matching with dense vector retrieval using Reciprocal Rank Fusion (RRF).
- **Reranking**: Integrate cross-encoder rerankers to improve Top-K retrieval precision.
- **User Authentication**: Add role-based multi-tenant authentication.
- **Cloud Vector Database**: Add connectors for Google Cloud Vertex AI Vector Search or Pinecone for massive-scale enterprise storage.
- **Document Versioning**: Track document update history and semantic diffs across document iterations.
- **Advanced In-Text Citations**: Direct inline sentence-level citations with PDF coordinate bounding-box highlights.
