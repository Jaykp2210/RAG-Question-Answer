"""
=============================================================================
 REAL-LIFE QUESTION-ANSWERING SYSTEM USING RAG (Retrieval-Augmented Generation)
 Built for Generative AI Engineering Portfolio & Production Showcase
=============================================================================
Pipeline Steps:
  1. Document Ingestion: Load user's PDF, DOCX, or TXT file.
  2. Recursive Chunking: Split document into semantic chunks with overlap.
  3. Embedding & Indexing: Store embeddings in local Chroma Vector DB.
  4. Context Retrieval: Retrieve top-k semantically relevant chunks for user query.
  5. Grounded Generation: Prompt LLM with strict factual grounding and citations.
=============================================================================
"""

import os
import sys
# pyrefly: ignore [missing-import]
from dotenv import load_dotenv

# Ensure Windows terminal supports UTF-8 characters and emojis
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

from rag_engine import (
    extract_text_from_file,
    chunk_documents,
    create_vector_store,
    query_rag_system,
    get_llm
)

# Load environment variables
load_dotenv()

def print_banner(step_num: int, title: str):
    print("\n" + "=" * 65)
    print(f" [Step {step_num}] {title.upper()}")
    print("=" * 65)



import argparse

def main():
    parser = argparse.ArgumentParser(description="Real-Life RAG Q&A System")
    parser.add_argument("--file", "-f", type=str, default=None, help="Path to document (PDF, TXT, DOCX)")
    parser.add_argument("--query", "-q", type=str, default=None, help="Question to ask (if set, runs query and exits)")
    args = parser.parse_args()

    print("""
    +-------------------------------------------------------------+
    |           REAL-LIFE DOCUMENT Q&A SYSTEM USING RAG           |
    |          Generative AI Project - Portfolio Edition          |
    +-------------------------------------------------------------+
    """)

    # -------------------------------------------------------------------------
    # STEP 1: DOCUMENT INGESTION (User uploads/specifies their own file)
    # -------------------------------------------------------------------------
    print_banner(1, "Document Ingestion (User Upload)")
    default_doc = "sample_document.txt"

    if args.file:
        user_file_input = args.file
        print(f"--> Using document specified via command-line: {user_file_input}")
    else:
        print(f"Enter the path to your document (PDF, TXT, DOCX)")
        print(f"Or press ENTER to use the built-in demo document [{default_doc}]:")
        user_file_input = input("File path > ").strip().strip('"').strip("'")

    if not user_file_input:
        user_file_input = default_doc

    if not os.path.exists(user_file_input):
        print(f"\n[ERROR] File not found at: {user_file_input}")
        print("Please check the path and try again.")
        sys.exit(1)

    print(f"--> Ingesting document: {user_file_input} ...")
    try:
        raw_docs = extract_text_from_file(user_file_input, os.path.basename(user_file_input))
        total_chars = sum(len(d.page_content) for d in raw_docs)
        print(f"[OK] Ingestion complete: Loaded {len(raw_docs)} page/section(s), {total_chars:,} total characters.")
    except Exception as e:
        print(f"[ERROR] Failed to read document: {e}")
        sys.exit(1)

    # -------------------------------------------------------------------------
    # STEP 2: RECURSIVE TEXT CHUNKING
    # -------------------------------------------------------------------------
    print_banner(2, "Recursive Text Chunking")
    chunk_size = 800
    chunk_overlap = 150
    print(f"--> Splitting text with chunk_size={chunk_size}, chunk_overlap={chunk_overlap}...")
    chunks = chunk_documents(raw_docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    print(f"[OK] Successfully generated {len(chunks)} chunks.")
    if chunks:
        sample_preview = chunks[0].page_content.replace("\n", " ")[:120]
        print(f"    Sample Chunk #1 Preview: \"{sample_preview}...\"")

    # -------------------------------------------------------------------------
    # STEP 3: LOCAL EMBEDDING & CHROMA VECTOR STORE INDEXING
    # -------------------------------------------------------------------------
    print_banner(3, "Vector Embeddings & Chroma Indexing")
    print("--> Initializing local ONNX embedding model (all-MiniLM-L6-v2, 384 dims)...")
    print("    [Info] 100% Free, runs locally on CPU with zero external API latency.")
    vector_store = create_vector_store(chunks)
    print(f"[OK] Chroma Vector Store indexed with {len(chunks)} embedded chunks.")

    # -------------------------------------------------------------------------
    # STEP 4: LLM INITIALIZATION
    # -------------------------------------------------------------------------
    print_banner(4, "LLM Configuration & Verification")
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    google_key = os.getenv("GOOGLE_API_KEY", "") or os.getenv("GEMINI_API_KEY", "")

    provider = "openrouter"
    active_key = api_key
    model_name = "meta-llama/llama-3.2-3b-instruct:free"

    if not active_key and google_key:
        provider = "google"
        active_key = google_key
        model_name = "gemini-2.5-flash"
    elif not active_key:
        print("Notice: No valid OPENROUTER_API_KEY or GOOGLE_API_KEY found in .env.")
        print("Please choose an LLM provider:")
        print("  1. OpenRouter (Enter key or use free models)")
        print("  2. Google Gemini (Enter Google AI Studio key)")
        choice = input("Choice [1 or 2] > ").strip()
        if choice == "2":
            provider = "google"
            active_key = input("Enter GOOGLE_API_KEY > ").strip()
            model_name = "gemini-2.5-flash"
        else:
            provider = "openrouter"
            active_key = input("Enter OPENROUTER_API_KEY > ").strip()
            model_name = "meta-llama/llama-3.2-3b-instruct:free"

    print(f"--> Initializing LLM: Provider='{provider}', Model='{model_name}'...")
    try:
        llm = get_llm(provider, active_key, model_name=model_name)
        print("[OK] LLM ready.")
    except Exception as e:
        print(f"[ERROR] Could not initialize LLM: {e}")
        sys.exit(1)

    # -------------------------------------------------------------------------
    # STEP 5: INTERACTIVE QUESTION-ANSWERING LOOP
    # -------------------------------------------------------------------------
    print_banner(5, "Real-Life RAG Q&A Session")
    
    if args.query:
        print(f"[Single Query Mode]: \"{args.query}\"")
        print("\n🔍 Retrieving semantically relevant chunks from Chroma...")
        answer, retrieved_docs = query_rag_system(vector_store, llm, args.query, top_k=3)
        print("\n🤖 [Grounded Answer]:")
        print("-" * 65)
        print(answer.strip())
        print("-" * 65)
        print(f"\n📚 [Retrieved Sources ({len(retrieved_docs)} Chunks Used)]:")
        for i, doc in enumerate(retrieved_docs, 1):
            src = doc.metadata.get("source", "Document")
            page = doc.metadata.get("page", 1)
            chunk_id = doc.metadata.get("chunk_id", i)
            snippet = doc.page_content.replace("\n", " ")[:140]
            print(f"  • Source {i} -> [Chunk #{chunk_id} | {src} | Page {page}]: \"{snippet}...\"")
        print("\n[Done] Single query complete.")
        return

    print("Ask any question regarding your uploaded document.")
    print("The system retrieves only relevant chunks and answers with strict grounding.")
    print("Type 'exit' or 'quit' to terminate.\n")

    while True:
        try:
            query = input("\n[Your Question] > ").strip()
            if not query:
                continue
            if query.lower() in ["exit", "quit", "q"]:
                print("\nExiting RAG system. Thank you!")
                break

            print("\n🔍 Retrieving semantically relevant chunks from Chroma...")
            answer, retrieved_docs = query_rag_system(vector_store, llm, query, top_k=3)

            print("\n🤖 [Grounded Answer]:")
            print("-" * 65)
            print(answer.strip())
            print("-" * 65)

            print(f"\n📚 [Retrieved Sources ({len(retrieved_docs)} Chunks Used)]:")
            for i, doc in enumerate(retrieved_docs, 1):
                src = doc.metadata.get("source", "Document")
                page = doc.metadata.get("page", 1)
                chunk_id = doc.metadata.get("chunk_id", i)
                snippet = doc.page_content.replace("\n", " ")[:140]
                print(f"  • Source {i} -> [Chunk #{chunk_id} | {src} | Page {page}]: \"{snippet}...\"")

        except KeyboardInterrupt:
            print("\nSession interrupted. Exiting.")
            break
        except Exception as e:
            print(f"\n[Error during query]: {e}")
            print("Tip: If you encounter an authentication error, verify your API key in .env or try Google Gemini / OpenRouter.")

if __name__ == "__main__":
    main()
