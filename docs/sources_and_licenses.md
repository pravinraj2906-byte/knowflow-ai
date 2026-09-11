# KnowFlow AI: Sources and Licenses Record

This record documents the third-party technologies, libraries, foundational models, datasets, and licensing considerations for **KnowFlow AI**.

---

## 1. Core Technologies & Dependencies

| Technology | Purpose in Project | Official Project / Documentation URL | License Status |
|---|---|---|---|
| **Python** | Runtime environment and language for application logic, pipeline orchestration, and test execution. | [python.org](https://www.python.org/) | Python Software Foundation (PSF) License |
| **Streamlit** | Interactive frontend framework powering the web interface, sidebar document management, and chat. | [streamlit.io](https://streamlit.io/)<br>[github.com/streamlit/streamlit](https://github.com/streamlit/streamlit) | Apache License 2.0 *(Verify from official project repository/package metadata)* |
| **google-genai** | Official modern Google GenAI Python SDK for executing embeddings and generation calls via Gemini API. | [github.com/googleapis/python-genai](https://github.com/googleapis/python-genai)<br>[ai.google.dev/gemini-api/docs](https://ai.google.dev/gemini-api/docs) | Apache License 2.0 *(Verify from official project repository/package metadata)* |
| **ChromaDB** | Local persistent vector database storing chunk embeddings, metadata, and executing cosine similarity search. | [trychroma.com](https://www.trychroma.com/)<br>[github.com/chroma-core/chroma](https://github.com/chroma-core/chroma) | Apache License 2.0 *(Confirmed in package metadata)* |
| **PyMuPDF (`pymupdf`)** | High-performance document parser extracting text and tracking 1-indexed page boundaries from PDFs. | [pymupdf.readthedocs.io](https://pymupdf.readthedocs.io/)<br>[github.com/pymupdf/PyMuPDF](https://github.com/pymupdf/PyMuPDF) | Dual Licensed: GNU AGPL v3.0 / Artifex Commercial License *(Confirmed in package metadata)* |
| **python-docx** | Document parser extracting text paragraphs and table cell contents from Microsoft Word (.docx) files. | [python-docx.readthedocs.io](https://python-docx.readthedocs.io/)<br>[github.com/python-openxml/python-docx](https://github.com/python-openxml/python-docx) | MIT License *(Confirmed in package metadata)* |
| **NumPy** | Vector operations, array manipulation, and numerical normalization. | [numpy.org](https://numpy.org/)<br>[github.com/numpy/numpy](https://github.com/numpy/numpy) | BSD 3-Clause License *(Verify from official project repository/package metadata)* |
| **python-dotenv** | Securely loads configuration variables from local `.env` files into environment variables. | [github.com/theskumar/python-dotenv](https://github.com/theskumar/python-dotenv) | BSD 3-Clause License *(Confirmed in package metadata)* |
| **pytest** | Test automation framework and assertion engine for pipeline unit and integration verification. | [pytest.org](https://pytest.org/)<br>[github.com/pytest-dev/pytest](https://github.com/pytest-dev/pytest) | MIT License *(Verify from official project repository/package metadata)* |

---

## 2. Foundational Models

| Model | Provider | Purpose | Terms & Licensing |
|---|---|---|---|
| **`gemini-embedding-2`** | Google AI Studio | Vector embedding model used to map document chunks and user queries into a shared vector space. | Governed by Google Gemini API Terms of Service ([ai.google.dev/terms](https://ai.google.dev/terms)). |
| **`gemini-3.6-flash`** | Google AI Studio | Fast, high-reasoning foundation model configured with `temperature=0.0` for strictly grounded response synthesis. | Governed by Google Gemini API Terms of Service ([ai.google.dev/terms](https://ai.google.dev/terms)). |

---

## 3. Project Code & Intellectual Property

- **Original Implementation**: All pipeline code contained in `src/ingestion/`, `src/rag/`, `src/utils/`, `app.py`, and `tests/test_rag.py` is original work authored for the Week 2 AI Internship Project.
- **Architectural Design**: Follows standard Retrieval-Augmented Generation best practices without relying on high-level proprietary orchestration frameworks (e.g., LangChain, LlamaIndex), maximizing clarity, maintainability, and transparency.

---

## 4. User-Provided Documents & Data

- **Data Privacy & Ownership**: User-uploaded documents (`.pdf`, `.docx`, `.txt`) remain the exclusive property of the user.
- **Local Data Storage**: Extracted chunks and generated vector embeddings are stored strictly on the local machine in `data/chroma/`. No document files or vector databases are sent to external storage or cloud vector databases.
- **Ephemeral API Transmission**: Text snippets are transmitted via TLS to Google AI Studio strictly for real-time embedding calculation and context-grounded response generation.

---

## 5. Fonts, Graphics & UI Components

- **Icons & Emojis**: System Unicode emojis (e.g., 🧠, 📄, 📊, 🔍, 🔒, 🗑️) are used for interface navigation without third-party icon font dependencies.
- **CSS Styling**: Custom CSS styles declared in `app.py` use native browser CSS properties (flexbox, CSS grid, borders, border-radius) and system sans-serif font stacks (`-apple-system`, `BlinkMacSystemFont`, `Segoe UI`, `Roboto`, `sans-serif`).

---

## 6. Generated Content

- **Grounding Mandate**: Responses produced by KnowFlow AI are generated dynamically using retrieved context.
- **Content Liability**: The system explicitly flags context absence via the standardized message: *"I couldn't find enough information in the uploaded documents to answer this question."* Outputs must be validated by the user against the provided source citations.

---

## 7. License Considerations & Distribution Guidance

When distributing or deploying KnowFlow AI, the following compliance considerations apply:
1. **PyMuPDF AGPL-3.0 License**: `PyMuPDF` is licensed under GNU AGPL v3.0. If KnowFlow AI is modified and hosted as a publicly accessible network service, AGPL requires releasing the corresponding source code under AGPL, or obtaining a commercial license from Artifex. For internal development, academic research, and personal evaluation, standard AGPL terms apply.
2. **API Key Security**: Users and developers must supply their own API keys via `.env` or the web sidebar. No shared credentials may be bundled with the software repository.
3. **Open-Source Compatibility**: Core utility code, chunking logic, and ChromaDB integrations use permissive open-source licenses (MIT, Apache 2.0, BSD-3-Clause).
