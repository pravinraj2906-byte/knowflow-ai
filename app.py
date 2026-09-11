"""
KnowFlow AI: RAG-Based Knowledge Assistant
Streamlit Web Application
"""

import os
import sys
from pathlib import Path
from typing import List

import streamlit as st

# Ensure root directory is on Python path
ROOT_DIR = Path(__file__).parent.resolve()
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.ingestion.extractor import extract_documents
from src.ingestion.chunker import chunk_documents
from src.rag.embeddings import GeminiEmbedder, EmbeddingError
from src.rag.vector_store import VectorStore
from src.rag.retriever import Retriever, INSUFFICIENT_CONTEXT_MESSAGE
from src.rag.generator import GroundedGenerator
from src.utils.helpers import load_api_key

# Page Configuration
st.set_page_config(
    page_title="KnowFlow AI - RAG Knowledge Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for polished interface
CUSTOM_CSS = """
<style>
    /* Global Card & Container Styling */
    .metric-container {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 12px 16px;
        margin-bottom: 10px;
    }
    .dark .metric-container {
        background-color: #1e293b;
        border-color: #334155;
    }
    .doc-pill {
        display: inline-block;
        background-color: #e0f2fe;
        color: #0369a1;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-right: 6px;
    }
    .source-tag {
        display: inline-flex;
        align-items: center;
        background-color: #f1f5f9;
        border: 1px solid #cbd5e1;
        color: #334155;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.85rem;
        font-weight: 500;
        margin: 4px 6px 4px 0;
    }
    .confidence-badge {
        font-size: 0.75rem;
        padding: 2px 6px;
        border-radius: 4px;
        background-color: #dcfce7;
        color: #15803d;
        font-weight: 600;
        margin-left: 6px;
    }
    .warning-box {
        background-color: #fef2f2;
        border-left: 4px solid #ef4444;
        color: #991b1b;
        padding: 12px;
        border-radius: 4px;
        margin: 8px 0;
        font-size: 0.9rem;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def init_session_state():
    """Initialize persistent session states."""
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "processed_files" not in st.session_state:
        st.session_state.processed_files = set()

    if "vector_store" not in st.session_state:
        st.session_state.vector_store = VectorStore()

    if "api_key" not in st.session_state:
        st.session_state.api_key = load_api_key() or ""


init_session_state()

# Verify API key
active_api_key = st.session_state.api_key or load_api_key()

# Initialize RAG components with active key
embedder = GeminiEmbedder(api_key=active_api_key) if active_api_key else None
retriever = Retriever(vector_store=st.session_state.vector_store, embedder=embedder) if embedder else None
generator = GroundedGenerator(api_key=active_api_key) if active_api_key else None


# -----------------------------------------------------------------------------
# SIDEBAR
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("🧠 KnowFlow AI")
    st.caption("RAG-Based Knowledge Assistant")
    st.markdown("---")

    # API Key Configuration Section
    if not active_api_key:
        st.error("🔑 API Key Required")
        user_key = st.text_input(
            "Enter Gemini API Key",
            type="password",
            help="Your key is kept only in current session and never stored permanently in git.",
        )
        if user_key.strip():
            st.session_state.api_key = user_key.strip()
            os.environ["GEMINI_API_KEY"] = user_key.strip()
            st.success("API Key updated!")
            st.rerun()
        st.markdown(
            "💡 Alternatively, set `GEMINI_API_KEY` in your `.env` file."
        )
        st.markdown("---")
    else:
        st.success("🔒 Gemini API Key Active")

    # Document Uploader Section
    st.subheader("📂 Document Ingestion")
    st.caption("Supported: **PDF**, **DOCX**, **TXT**")
    
    uploaded_files = st.file_uploader(
        "Upload files",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        help="Upload one or multiple documents to index into the local vector store.",
    )

    # Process newly uploaded files
    if uploaded_files:
        new_files = [f for f in uploaded_files if f.name not in st.session_state.processed_files]
        if new_files:
            if not active_api_key:
                st.warning("Please configure your Gemini API Key before indexing documents.")
            else:
                progress_bar = st.progress(0)
                status_text = st.empty()
                had_error = False

                for idx, file_obj in enumerate(new_files):
                    status_text.text(f"Extracting & chunking {file_obj.name}...")
                    
                    # 1. Extract documents
                    raw_docs = extract_documents(file_obj, filename=file_obj.name)
                    if not raw_docs:
                        st.warning(f"No extractable text found in '{file_obj.name}'.")
                        st.session_state.processed_files.add(file_obj.name)
                        continue

                    # 2. Chunk text
                    chunks = chunk_documents(raw_docs)
                    if not chunks:
                        st.warning(f"'{file_obj.name}' produced 0 usable chunks.")
                        st.session_state.processed_files.add(file_obj.name)
                        continue

                    # 3. Embed chunks (batched with exponential backoff & TLS resilience)
                    status_text.text(f"Generating embeddings for {len(chunks)} chunks from {file_obj.name}...")
                    try:
                        texts = [c["text"] for c in chunks]
                        embeddings = embedder.embed_documents(texts)

                        # 4. Store in ChromaDB
                        st.session_state.vector_store.add_chunks(chunks, embeddings)
                        st.session_state.processed_files.add(file_obj.name)
                        st.toast(f"✅ Indexed {file_obj.name} ({len(chunks)} chunks)")

                    except Exception as e:
                        had_error = True
                        error_msg = f"Failed to index '{file_obj.name}': {str(e)}"
                        st.error(f"❌ {error_msg}")
                        st.session_state["last_ingestion_error"] = error_msg
                        # Do not continue loop on fatal embedding error
                        break

                    progress_bar.progress((idx + 1) / len(new_files))

                status_text.empty()
                progress_bar.empty()
                
                # Only rerun if indexing succeeded with no errors
                if not had_error:
                    if "last_ingestion_error" in st.session_state:
                        del st.session_state["last_ingestion_error"]
                    st.rerun()

    # Display persistent ingestion error if one occurred
    if "last_ingestion_error" in st.session_state and st.session_state["last_ingestion_error"]:
        st.error(f"⚠️ **Ingestion Issue**: {st.session_state['last_ingestion_error']}")
        if "dimension" in st.session_state["last_ingestion_error"].lower():
            st.info("💡 Hint: Click 'Clear Knowledge Base' below to reset the vector database for the new embedding model.")
        if st.button("Dismiss Error Alert"):
            del st.session_state["last_ingestion_error"]
            st.rerun()

    st.markdown("---")

    # Knowledge Base Statistics & Management
    st.subheader("📊 Knowledge Base")
    doc_list = st.session_state.vector_store.list_documents()
    total_chunks = st.session_state.vector_store.count()

    col_stat1, col_stat2 = st.columns(2)
    with col_stat1:
        st.metric("Documents", len(doc_list))
    with col_stat2:
        st.metric("Total Chunks", total_chunks)

    if doc_list:
        st.markdown("##### Uploaded Documents")
        for doc in doc_list:
            pages_info = f" • {len(doc['pages'])} pages" if doc["pages"] else ""
            st.markdown(
                f"<span class='doc-pill'>{doc['file_type']}</span> **{doc['source']}**<br>"
                f"<small style='color: gray;'>{doc['chunk_count']} chunks{pages_info} • Status: {doc['status']}</small>",
                unsafe_allow_html=True,
            )
            st.markdown("<div style='margin-bottom: 6px;'></div>", unsafe_allow_html=True)

        st.markdown("---")
        if st.button("🗑️ Clear Knowledge Base", use_container_width=True):
            st.session_state.vector_store.reset_collection()
            st.session_state.processed_files.clear()
            st.session_state.messages.clear()
            st.toast("Knowledge base cleared.")
            st.rerun()
    else:
        st.info("No documents currently indexed.")


# -----------------------------------------------------------------------------
# MAIN CHAT AREA
# -----------------------------------------------------------------------------
st.title("KnowFlow AI")
st.markdown(
    "**RAG-Based Knowledge Assistant** • Ask questions grounded in your uploaded documents."
)

# Display empty state welcome guide if knowledge base is empty
if total_chunks == 0:
    st.markdown(
        """
        <div style="background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
                    border: 1px solid #bae6fd; border-radius: 12px; padding: 24px; margin-top: 16px;">
            <h3 style="color: #0369a1; margin-top: 0;">👋 Welcome to KnowFlow AI</h3>
            <p style="color: #334155; font-size: 1.05rem;">
                Your intelligent, hallucination-resistant document assistant powered by 
                <strong>Google Gemini</strong>, <strong>ChromaDB</strong>, and <strong>RAG</strong>.
            </p>
            <hr style="border-top: 1px solid #cbd5e1; margin: 16px 0;">
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px;">
                <div>
                    <strong>1. 🔑 Connect Key</strong><br>
                    <small style="color: #475569;">Ensure your Gemini API key is configured in <code>.env</code> or sidebar.</small>
                </div>
                <div>
                    <strong>2. 📄 Ingest Documents</strong><br>
                    <small style="color: #475569;">Upload PDF, DOCX, or TXT files using the sidebar uploader.</small>
                </div>
                <div>
                    <strong>3. 🎯 Ask Questions</strong><br>
                    <small style="color: #475569;">Ask questions below and receive answers strictly verified against your files.</small>
                </div>
                <div>
                    <strong>4. 🛡️ Zero Hallucinations</strong><br>
                    <small style="color: #475569;">Unanswerable questions trigger a verified fallback rather than made-up answers.</small>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Render Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # If assistant response has sources, display citation pills
        if msg.get("sources"):
            st.markdown("##### 📚 Sources:")
            citations_html = "".join(
                f"<span class='source-tag'>📄 {src}</span>" for src in msg["sources"]
            )
            st.markdown(citations_html, unsafe_allow_html=True)

        # Show expandable retrieval context if available
        if msg.get("chunks"):
            with st.expander("🔍 View Retrieved Context Evidence"):
                for idx, chunk in enumerate(msg["chunks"], start=1):
                    meta = chunk.get("metadata", {})
                    src = meta.get("source", "document")
                    page_num = meta.get("page")
                    page_str = f" • Page {page_num}" if page_num else ""
                    sim = chunk.get("similarity", 0.0)
                    dist = chunk.get("distance", 0.0)

                    st.markdown(
                        f"**Snippet #{idx}** — `{src}`{page_str} "
                        f"<span class='confidence-badge'>Sim: {sim:.2f} | Dist: {dist:.2f}</span>",
                        unsafe_allow_html=True,
                    )
                    st.caption(chunk.get("text", ""))
                    st.divider()


# User Query Input
query_input = st.chat_input("Ask a question about your uploaded documents...")

if query_input:
    # 1. Validation checks
    if not active_api_key:
        st.error("⚠️ Please configure your Gemini API Key in the sidebar or `.env` to ask questions.")
        st.stop()

    if total_chunks == 0:
        st.warning("⚠️ No documents in the knowledge base. Please upload at least one PDF, DOCX, or TXT file first.")
        st.stop()

    # 2. Append and render user message
    st.session_state.messages.append({"role": "user", "content": query_input})
    with st.chat_message("user"):
        st.markdown(query_input)

    # 3. Retrieve and Generate
    with st.chat_message("assistant"):
        with st.spinner("Searching document knowledge base..."):
            try:
                retrieval_res = retriever.retrieve(query_input)
                
                # If relevance threshold check indicates insufficient context
                if not retrieval_res.has_relevant_context:
                    answer_text = INSUFFICIENT_CONTEXT_MESSAGE
                    sources = []
                    chunks = []
                    is_fallback = True
                else:
                    # Grounded Gemini generation
                    gen_result = generator.generate_answer(
                        question=query_input,
                        retrieved_chunks=retrieval_res.chunks,
                    )
                    answer_text = gen_result.answer
                    sources = gen_result.sources
                    chunks = gen_result.retrieved_chunks
                    is_fallback = gen_result.is_fallback

                # Render response
                st.markdown(answer_text)

                if sources:
                    st.markdown("##### 📚 Sources:")
                    citations_html = "".join(
                        f"<span class='source-tag'>📄 {src}</span>" for src in sources
                    )
                    st.markdown(citations_html, unsafe_allow_html=True)

                if chunks:
                    with st.expander("🔍 View Retrieved Context Evidence"):
                        for idx, chunk in enumerate(chunks, start=1):
                            meta = chunk.get("metadata", {})
                            src = meta.get("source", "document")
                            page_num = meta.get("page")
                            page_str = f" • Page {page_num}" if page_num else ""
                            sim = chunk.get("similarity", 0.0)
                            dist = chunk.get("distance", 0.0)

                            st.markdown(
                                f"**Snippet #{idx}** — `{src}`{page_str} "
                                f"<span class='confidence-badge'>Sim: {sim:.2f} | Dist: {dist:.2f}</span>",
                                unsafe_allow_html=True,
                            )
                            st.caption(chunk.get("text", ""))
                            st.divider()

                # Store assistant response in history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer_text,
                    "sources": sources,
                    "chunks": chunks,
                    "is_fallback": is_fallback,
                })

            except Exception as e:
                error_msg = f"An unexpected error occurred during processing: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                    "sources": [],
                    "chunks": [],
                    "is_fallback": True,
                })
