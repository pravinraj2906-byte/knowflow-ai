# KnowFlow AI: Local Setup Instructions

This document provides step-by-step instructions for configuring, running, and testing **KnowFlow AI** in a local development environment.

---

## 1. Prerequisites & Environment

- **Operating System**: Windows, macOS, or Linux
- **Python Version**: Python 3.10 to Python 3.14 (Python 3.14 verified)
- **Package Manager**: `pip` (bundled with standard Python installations)
- **Google AI Studio Account**: Required to generate a free Gemini API key

---

## 2. Virtual Environment Setup

Always isolate project dependencies inside a dedicated virtual environment.

### Step 1: Clone or Navigate to the Repository
Open a terminal (PowerShell on Windows, Bash on macOS/Linux) and navigate to the project directory:

```bash
cd /path/to/KNOWFLOWAI
```

### Step 2: Create the Virtual Environment
Create a virtual environment named `.venv`:

```bash
python -m venv .venv
```

### Step 3: Activate the Virtual Environment

**Windows (PowerShell):**
```powershell
.\.venv\Scripts\activate
```
*(If you encounter execution policy restrictions on Windows, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first).*

**Windows (Command Prompt):**
```cmd
.venv\Scripts\activate.bat
```

**macOS / Linux:**
```bash
source .venv/bin/activate
```

Upon successful activation, your shell prompt will display `(.venv)`.

---

## 3. Install Dependencies

Upgrade `pip` and install the pinned dependencies from `requirements.txt`:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Core Installed Packages:
- `streamlit` — Web interface and conversational feed
- `google-genai` — Official modern Google GenAI SDK
- `chromadb` — Local persistent vector database
- `pymupdf` — High-speed PDF text and page extraction
- `python-docx` — Microsoft Word parsing
- `numpy` — Array and vector calculations
- `python-dotenv` — Environment variable loader
- `pytest` — Automated testing suite

---

## 4. Configure Environment Variables (`.env`)

KnowFlow AI requires a Google Gemini API key to calculate vector embeddings and generate grounded responses.

Create a file named `.env` in the root directory:

```env
# Google Gemini API Key (Obtain from https://aistudio.google.com/)
GEMINI_API_KEY=your_actual_gemini_api_key

# Model Configuration
GEMINI_MODEL=gemini-3.6-flash
EMBEDDING_MODEL=gemini-embedding-2
```

> [!CAUTION]
> **Never commit your actual API key to version control.** The `.env` file is explicitly ignored in `.gitignore`.

### Alternative Sidebar Key Entry
If you prefer not to write your key to a file on disk, you can leave `.env` with a placeholder and input your key securely into the password input field in the Streamlit application sidebar during runtime. The key is kept strictly in session memory.

---

## 5. Running the Test Suite

Before starting the web application, verify your installation by executing the automated test suite:

```bash
python -m pytest -v
```

### What Is Tested:
- Multi-format file extraction (PDF, DOCX, TXT)
- 1-indexed page preservation for PDFs
- Paragraph boundary and overlapping text chunking
- Deterministic SHA-256 chunk IDs
- `gemini-embedding-2` 1-to-1 chunk-to-embedding mapping
- Transient network/SSL error retry with exponential backoff
- Vector store persistence and deduplication
- ChromaDB dimensionality mismatch detection
- Hallucination distance threshold fallback triggers

> **Note**: All 26 tests use offline mocks for the Gemini API. No real API quota is consumed during testing.

---

## 6. Running the Streamlit Application

Start the KnowFlow AI application using Streamlit:

```bash
streamlit run app.py
```

Streamlit will start a local web server. Open your web browser to:

```
http://localhost:8501
```

---

## 7. Supported File Types & Ingestion Guide

KnowFlow AI supports three document formats via the sidebar uploader:
1. **PDF (`.pdf`)**: Standard text-based PDF documents. Automatically tracks individual page numbers.
2. **Word (`.docx`)**: Microsoft Word files. Extracts text paragraphs and table cell contents.
3. **Plain Text (`.txt`)**: UTF-8 and ANSI text documents.

### Ingestion Best Practices:
- You can upload multiple files at once.
- Each document is automatically split into chunks, converted to vector embeddings, and stored in `data/chroma/`.
- If you upload a revised version of a file with the same name, KnowFlow AI replaces the prior chunks automatically to prevent duplicate answers.

---

## 8. Troubleshooting Common Errors

### 1. `ModuleNotFoundError: No module named 'src'`
- **Cause**: Python cannot find the root directory on `sys.path`.
- **Solution**: Ensure your virtual environment is active and run commands from the project root (`KNOWFLOWAI/`). When running tests, use `python -m pytest -v`.

### 2. `IncompatibleEmbeddingDimensionError` / ChromaDB Dimension Conflict
- **Cause**: The local ChromaDB database (`data/chroma/`) was previously populated with vectors of a different dimensionality (e.g. 768 dimensions from `text-embedding-004` vs 1536/3072 from `gemini-embedding-2`).
- **Solution**: Click the **"Clear Knowledge Base"** button in the Streamlit sidebar. This resets the local collection, allowing fresh embeddings to index cleanly.

### 3. `[SSL: UNEXPECTED_EOF_WHILE_READING] EOF occurred in violation of protocol`
- **Cause**: Network packet drop, firewall/proxy TLS inspection, or oversized batch during embedding generation.
- **Solution**: KnowFlow AI includes built-in exponential backoff retries with a safe batch size of 10 chunks and TLS client refresh. If your network continues dropping packets:
  - Check whether a corporate VPN or strict firewall is intercepting Google API connections.
  - Verify your Internet connection stability.

### 4. `API Key Required` Alert in Sidebar
- **Cause**: `GEMINI_API_KEY` is not set or still contains the template placeholder string.
- **Solution**: Add your key to `.env` as `GEMINI_API_KEY=AIzaSy...` or type it directly into the sidebar key input.

### 5. `I couldn't find enough information in the uploaded documents to answer this question.`
- **Cause**: This is **expected behavior** when a query is off-topic or when the uploaded documents lack sufficient context. KnowFlow AI's distance threshold prevents hallucinating answers.
- **Solution**: Upload documents that contain the information needed to address your query.
