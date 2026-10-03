import os
import streamlit as st
# pyrefly: ignore [missing-import]
from dotenv import load_dotenv
from rag_engine import (
    extract_text_from_file,
    chunk_documents,
    create_vector_store,
    query_rag_system,
    get_llm
)

# Load existing environment variables
load_dotenv()

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="Real-Life RAG: Document Q&A System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 10px;
    }
    .source-box {
        background-color: #F1F5F9;
        border-left: 4px solid #3B82F6;
        padding: 10px 14px;
        border-radius: 4px;
        margin-bottom: 10px;
        font-size: 0.88rem;
    }
    .badge-rag {
        background-color: #DEF7EC;
        color: #03543F;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 12px;
        display: inline-block;
        margin-bottom: 6px;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "processed_filename" not in st.session_state:
    st.session_state.processed_filename = None
if "document_chunks" not in st.session_state:
    st.session_state.document_chunks = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "doc_stats" not in st.session_state:
    st.session_state.doc_stats = {}


# =============================================================================
# BACKEND LLM INITIALIZATION (Silent & Automatic from .env)
# =============================================================================
def get_configured_llm():
    """
    Silently loads the LLM configuration from the environment (.env)
    in the backend so the user does not need to enter keys or models in the UI.
    """
    openrouter_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    google_key = os.getenv("GOOGLE_API_KEY", "") or os.getenv("GEMINI_API_KEY", "").strip()
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    groq_key = os.getenv("GROQ_API_KEY", "").strip()

    if google_key:
        return get_llm("google", google_key, model_name="gemini-2.5-flash")
    elif openrouter_key:
        return get_llm("openrouter", openrouter_key, model_name="google/gemini-2.0-flash-exp:free")
    elif openai_key:
        return get_llm("openai", openai_key, model_name="gpt-4o-mini")
    elif groq_key:
        return get_llm("groq", groq_key, model_name="llama-3.1-8b-instant")
    else:
        # Default fallback
        return get_llm("openrouter", "demo_mode", model_name="google/gemini-2.0-flash-exp:free")


# =============================================================================
# SIDEBAR (Clean & Focused on Document Management)
# =============================================================================
with st.sidebar:
    st.header("📚 Document Assistant")
    st.markdown("Upload any document (PDF, DOCX, TXT) and ask questions. The system answers strictly using your document's content.")

    st.markdown("---")
    st.subheader("⚡ System Status")
    st.success("🟢 RAG Pipeline Active")
    st.caption("• Embeddings: Local ONNX (all-MiniLM-L6-v2)\n• Vector Store: ChromaDB\n• Grounding: Strict Anti-Hallucination")

    st.markdown("---")
    st.subheader("🛠️ Document Controls")
    if st.button("🧹 Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

    if st.button("🔄 Reset Document & Index", use_container_width=True):
        st.session_state.vector_store = None
        st.session_state.processed_filename = None
        st.session_state.document_chunks = []
        st.session_state.chat_history = []
        st.session_state.doc_stats = {}
        st.rerun()

# Sensible default retrieval parameters
chunk_size = 800
chunk_overlap = 150
top_k = 3



# =============================================================================
# MAIN LAYOUT
# =============================================================================
st.markdown('<div class="main-header">Real-Life Document Q&A System Using RAG</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Upload your own documents (PDF, DOCX, TXT) and get strictly grounded answers with verifiable source citations.</div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# STEP 1: FILE UPLOADER & PROCESSING
# -----------------------------------------------------------------------------
with st.expander("📂 Step 1: Upload & Process Document", expanded=(st.session_state.vector_store is None)):
    col1, col2 = st.columns([3, 1])

    with col1:
        uploaded_file = st.file_uploader(
            "Upload your document (PDF, TXT, DOCX)",
            type=["pdf", "txt", "docx", "md"],
            help="Your file is parsed in-memory, chunked, and embedded into local Chroma vector database."
        )

    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        use_sample = st.button("📄 Or Load Built-in Demo Document", use_container_width=True)

    # Trigger Ingestion
    file_to_process = None
    filename_to_process = None

    if uploaded_file is not None:
        file_to_process = uploaded_file
        filename_to_process = uploaded_file.name
    elif use_sample and os.path.exists("sample_document.txt"):
        with open("sample_document.txt", "rb") as f:
            file_to_process = f.read()
        filename_to_process = "sample_document.txt"

    if file_to_process is not None and (st.session_state.processed_filename != filename_to_process):
        with st.spinner(f"Ingesting and indexing '{filename_to_process}'..."):
            try:
                # 1. Parse text
                raw_docs = extract_text_from_file(file_to_process, filename_to_process)
                total_chars = sum(len(d.page_content) for d in raw_docs)

                # 2. Chunk text
                chunks = chunk_documents(raw_docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

                # 3. Embed & Vector Index
                vector_store = create_vector_store(chunks)

                # 4. Save to session state
                st.session_state.vector_store = vector_store
                st.session_state.processed_filename = filename_to_process
                st.session_state.document_chunks = chunks
                st.session_state.doc_stats = {
                    "filename": filename_to_process,
                    "pages": len(raw_docs),
                    "chars": total_chars,
                    "chunks": len(chunks)
                }
                st.session_state.chat_history = []
                st.success(f" Successfully processed '{filename_to_process}' into {len(chunks)} searchable chunks!")
                st.rerun()

            except Exception as e:
                st.error(f"Error processing document: {str(e)}")

# Display Document Stats if active
if st.session_state.vector_store is not None:
    stats = st.session_state.doc_stats
    cols = st.columns(4)
    cols[0].metric("Active Document", stats.get("filename", "N/A"))
    cols[1].metric("Pages / Sections", stats.get("pages", 0))
    cols[2].metric("Total Characters", f"{stats.get('chars', 0):,}")
    cols[3].metric("Generated Chunks", stats.get("chunks", 0))

    # Chunk Inspector Expander
    with st.expander("🔍 Inspect Document Chunks (Intern Knowledge Check)"):
        st.write(f"Displaying {min(3, len(st.session_state.document_chunks))} sample chunk(s) from Chroma:")
        for idx, chunk in enumerate(st.session_state.document_chunks[:3]):
            st.markdown(f"**Chunk #{chunk.metadata.get('chunk_id', idx+1)}** (Page {chunk.metadata.get('page', 1)})")
            st.code(chunk.page_content, language="markdown")


# -----------------------------------------------------------------------------
# STEP 2: CHAT & RETRIEVAL INTERFACE
# -----------------------------------------------------------------------------
st.markdown("---")
st.subheader("💬 Step 2: Ask Questions Grounded in Your Document")

if st.session_state.vector_store is None:
    st.info("👆 Please upload a document or click 'Load Built-in Demo Document' above to activate the RAG system.")
else:
    # Render existing conversation
    for chat in st.session_state.chat_history:
        with st.chat_message(chat["role"]):
            if chat["role"] == "assistant":
                st.markdown('<span class="badge-rag">Strictly Grounded RAG</span>', unsafe_allow_html=True)
            st.markdown(chat["content"])

            # If assistant response has retrieved sources, show them in an expander
            if chat.get("sources"):
                with st.expander(f"📚 View Retrieved Sources ({len(chat['sources'])} chunks used as context)"):
                    for idx, src_doc in enumerate(chat["sources"], 1):
                        st.markdown(f"""
                        <div class="source-box">
                            <b>Source Chunk #{src_doc.get('chunk_id', idx)}</b> | File: <i>{src_doc.get('source', 'Doc')}</i> | Page: {src_doc.get('page', 1)}<br>
                            {src_doc.get('content', '')}
                        </div>
                        """, unsafe_allow_html=True)

    # Chat Input Box
    user_query = st.chat_input("Ask a question about the uploaded document...")

    if user_query:
        # Display user question
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Generate Assistant Answer
        with st.chat_message("assistant"):
            st.markdown('<span class="badge-rag">Strictly Grounded RAG</span>', unsafe_allow_html=True)
            with st.spinner("Retrieving relevant chunks and generating grounded answer..."):
                try:
                    llm = get_configured_llm()

                    answer, retrieved_docs = query_rag_system(
                        vector_store=st.session_state.vector_store,
                        llm=llm,
                        question=user_query,
                        top_k=top_k
                    )

                    st.markdown(answer)

                    # Package sources for storage and display
                    sources_data = [
                        {
                            "chunk_id": doc.metadata.get("chunk_id", i + 1),
                            "source": doc.metadata.get("source", "Document"),
                            "page": doc.metadata.get("page", 1),
                            "content": doc.page_content
                        }
                        for i, doc in enumerate(retrieved_docs)
                    ]

                    with st.expander(f"📚 View Retrieved Sources ({len(sources_data)} chunks used as context)"):
                        for idx, src in enumerate(sources_data, 1):
                            st.markdown(f"""
                            <div class="source-box">
                                <b>Source Chunk #{src['chunk_id']}</b> | File: <i>{src['source']}</i> | Page: {src['page']}<br>
                                {src['content']}
                            </div>
                            """, unsafe_allow_html=True)

                    # Save to history
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources_data
                    })

                except Exception as e:
                    err_msg = str(e)
                    st.error(f"Generation notice: {err_msg}")

