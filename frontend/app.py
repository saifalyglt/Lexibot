# Streamlit frontend for LexiBot legal document assistant
# Updated to connect to your Hugging Face Space backend

import os
import requests
import streamlit as st
from typing import Dict, Any, List
import time

# Configure the page
st.set_page_config(
    page_title="LexiBot Legal Document Assistant", 
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "LexiBot - AI-powered legal document assistant"
    }
)

# ─── Set up constants and endpoints ────────────────────────────────────────────
# Replace this with your actual HF Space URL (the one you see in your browser
# when you open your Space). It *must* end with .hf.space, not hug​gingface.co/spaces.
API_URL           = "https://saifaligzr-lexibot.hf.space"
UPLOAD_ENDPOINT   = f"{API_URL}/embed"
SEARCH_ENDPOINT   = f"{API_URL}/summarize"     # or f"{API_URL}/search" if you want the raw search route
DOCUMENTS_ENDPOINT= f"{API_URL}/documents"

# Initialize session state
if 'uploaded_documents' not in st.session_state:
    st.session_state.uploaded_documents = []
if 'search_history' not in st.session_state:
    st.session_state.search_history = []

# ─── Helper functions ─────────────────────────────────────────────────────────

def check_api_connection() -> bool:
    """Check if the API backend is running."""
    try:
        r = requests.get(API_URL, timeout=5)
        return r.status_code == 200
    except Exception:
        return False

def upload_document(file_data, filename: str) -> Dict[str, Any]:
    """Upload a document to the backend."""
    try:
        files = {'file': (filename, file_data, 'application/octet-stream')}
        r = requests.post(UPLOAD_ENDPOINT, files=files, timeout=60)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.Timeout:
        st.error("Upload timeout - please try a smaller file or check your connection.")
        return {}
    except Exception as e:
        st.error(f"Error uploading document: {e}")
        return {}

def search_documents(query: str, top_k: int = 5) -> Dict[str, Any]:
    """Perform a legal document search (or summarization)."""
    try:
        payload = {'query': query, 'top_k': top_k}
        r = requests.post(SEARCH_ENDPOINT, json=payload, timeout=60)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.Timeout:
        st.error("Search timeout - please try again.")
        return {}
    except Exception as e:
        st.error(f"Error searching documents: {e}")
        return {}

def get_document_list() -> List[Dict[str, Any]]:
    """Get list of uploaded documents from backend."""
    try:
        r = requests.get(DOCUMENTS_ENDPOINT, timeout=10)
        r.raise_for_status()
        return r.json().get('documents', [])
    except Exception as e:
        st.error(f"Error retrieving documents: {e}")
        return []

# ─── Main UI ──────────────────────────────────────────────────────────────────

st.title("⚖️ LexiBot: Legal Document Assistant")
st.markdown("*AI-powered legal research and document analysis*")

def main():
    # Connection check
    if not check_api_connection():
        st.error("❌ Cannot connect to LexiBot API.")
        st.info("Make sure CORS is enabled in the backend and that API_URL points to your .hf.space domain.")
        return
    st.success("✅ Connected to LexiBot API")

    col1, col2 = st.columns([1, 2])

    # Left: Upload + Document Library
    with col1:
        st.header("📄 Document Management")
        uploaded_file = st.file_uploader("Choose a file (PDF or TXT)", type=["pdf", "txt"])
        if uploaded_file:
            size_kb = len(uploaded_file.getvalue()) / 1024
            st.info(f"**{uploaded_file.name}** — {size_kb:.1f} KB")
            if st.button("🔄 Upload & Process Document"):
                with st.spinner("Processing document..."):
                    prog = st.progress(0)
                    for i in range(1, 101):
                        time.sleep(0.005)
                        prog.progress(i)
                    result = upload_document(uploaded_file.getvalue(), uploaded_file.name)
                    prog.empty()
                    if result:
                        st.success("✅ Document uploaded successfully!")
                        st.json(result)
                        st.session_state.uploaded_documents.append({
                            'name': uploaded_file.name,
                            'chunks': result.get('chunks_created', 0),
                            'timestamp': time.strftime('%Y-%m-%d %H:%M')
                        })
                        st.experimental_rerun()

        st.subheader("📚 Document Library")
        docs = get_document_list()
        if docs:
            for doc in docs:
                with st.expander(f"{doc['document_id']}"):
                    st.write(f"Chunks: {doc['chunk_count']}")
        else:
            st.info("No documents uploaded yet.")

    # Right: Search & Summarize
    with col2:
        st.header("🔍 Legal Research")
        query = st.text_area("Enter your legal question:")
        top_k = st.selectbox("Results:", [3, 5, 7, 10], index=1)
        if st.button("🚀 Search & Analyze", disabled=not query.strip()):
            with st.spinner("Analyzing..."):
                results = search_documents(query.strip(), top_k)
                if results and 'summary' in results:
                    st.markdown("### 🤖 AI-Generated Summary")
                    st.markdown(results['summary'])
                    if results.get('citations'):
                        st.markdown("### 📖 Source Citations")
                        for idx, cit in enumerate(results['citations'], 1):
                            with st.expander(f"Citation {idx} - {cit['document_id']}"):
                                st.write(f"Relevance: {cit['score']:.3f}")
                                st.write(f"> {cit['text']}")
                        st.session_state.search_history.insert(0, {
                            'query': query.strip(),
                            'timestamp': time.strftime('%H:%M:%S'),
                            'results_count': len(results.get('citations', []))
                        })
                else:
                    st.error("No summary generated.")

        # Search history
        if st.session_state.search_history:
            st.subheader("🕒 Recent Searches")
            for i, h in enumerate(st.session_state.search_history[:5]):
                with st.expander(f"[{h['timestamp']}] {h['query'][:50]}"):
                    st.write(f"Query: {h['query']}")
                    st.write(f"Results: {h['results_count']}")
                    if st.button("Repeat", key=f"rp_{i}"):
                        st.experimental_set_query_params(query=h['query'])
                        st.experimental_rerun()

if __name__ == '__main__':
    main()
