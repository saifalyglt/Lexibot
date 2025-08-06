# Streamlit frontend for LexiBot legal document assistant
# Final version: persistent summaries, upload/delete with rerun, no errors

import os
import requests
import streamlit as st
from typing import Dict, Any, List
import time

# ─── Configuration ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LexiBot Legal Document Assistant",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={'About': "LexiBot - AI-powered legal document assistant"}
)

# Replace with your actual HF Space URL (must end with .hf.space)
API_URL            = "https://saifaligzr-lexibot.hf.space"
UPLOAD_ENDPOINT    = f"{API_URL}/embed"
SEARCH_ENDPOINT    = f"{API_URL}/summarize"
DOCUMENTS_ENDPOINT = f"{API_URL}/documents"

# Admin mode (secure with Streamlit Secrets: add ADMIN_PASS to your st.secrets)
ADMIN_PASS = st.secrets.get("ADMIN_PASS", "")
admin_input = st.sidebar.text_input("Admin Password", type="password")
IS_ADMIN = bool(admin_input and admin_input == ADMIN_PASS)
if IS_ADMIN:
    st.sidebar.success("Admin mode enabled")

# Initialize session state
if 'uploaded_documents' not in st.session_state:
    st.session_state.uploaded_documents = []
if 'search_history' not in st.session_state:
    st.session_state.search_history = []
if 'last_summary' not in st.session_state:
    st.session_state.last_summary = None
if 'last_citations' not in st.session_state:
    st.session_state.last_citations = []

# ─── Helper Functions ─────────────────────────────────────────────────────────

def check_api_connection() -> bool:
    try:
        r = requests.get(API_URL, timeout=5)
        return r.status_code == 200
    except Exception:
        return False

def upload_document(file_data: bytes, filename: str) -> Dict[str, Any]:
    try:
        files = {'file': (filename, file_data, 'application/octet-stream')}
        r = requests.post(UPLOAD_ENDPOINT, files=files, timeout=60)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.Timeout:
        st.error("Upload timeout—please try a smaller file or check your connection.")
        return {}
    except Exception as e:
        st.error(f"Error uploading document: {e}")
        return {}

def search_documents(query: str, top_k: int) -> Dict[str, Any]:
    try:
        payload = {'query': query, 'top_k': top_k}
        r = requests.post(SEARCH_ENDPOINT, json=payload, timeout=60)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.Timeout:
        st.error("Search timeout—please try again.")
        return {}
    except Exception as e:
        st.error(f"Error searching documents: {e}")
        return {}

def get_document_list() -> List[Dict[str, Any]]:
    try:
        r = requests.get(DOCUMENTS_ENDPOINT, timeout=10)
        r.raise_for_status()
        return r.json().get('documents', [])
    except Exception as e:
        st.error(f"Error retrieving documents: {e}")
        return []

def delete_document(document_id: str) -> bool:
    try:
        r = requests.delete(f"{DOCUMENTS_ENDPOINT}/{document_id}", timeout=10)
        r.raise_for_status()
        return True
    except Exception as e:
        st.error(f"Failed to delete {document_id}: {e}")
        return False

# ─── Main UI ──────────────────────────────────────────────────────────────────

st.title("⚖️ LexiBot: Legal Document Assistant")
st.markdown("*AI-powered legal research and document analysis*")

def main():
    # 1) Connection check
    if not check_api_connection():
        st.error("❌ Cannot connect to LexiBot API.")
        st.info("Ensure CORS is enabled in the backend and API_URL is correct.")
        return
    st.success("✅ Connected to LexiBot API")

    col1, col2 = st.columns([1, 2])

    # ─── Left Column: Upload & Document Library ────────────────────────────────
    with col1:
        st.header("📄 Document Management")

        # Upload widget
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
                        # refresh document list
                        st.rerun()

        # Document library with delete for admin
        st.subheader("📚 Document Library")
        docs = get_document_list()
        if docs:
            for doc in docs:
                row_col1, row_col2 = st.columns([4,1])
                with row_col1:
                    st.write(f"• **{doc['document_id']}** — {doc['chunk_count']} chunks")
                with row_col2:
                    if IS_ADMIN:
                        if st.button("🗑️ Delete", key=f"del_{doc['document_id']}"):
                            if delete_document(doc['document_id']):
                                st.success(f"Deleted {doc['document_id']}")
                                st.rerun()
        else:
            st.info("No documents uploaded yet.")

    # ─── Right Column: Search & Summarize ──────────────────────────────────────
    with col2:
        st.header("🔍 Legal Research")

        query = st.text_area("Enter your legal question:")
        top_k = st.selectbox("Results:", [3, 5, 7, 10], index=1)
        if st.button("🚀 Search & Analyze", disabled=not query.strip()):
            with st.spinner("Analyzing..."):
                results = search_documents(query.strip(), top_k)
                if results and 'summary' in results:
                    # store in session state so it persists
                    st.session_state.last_summary = results['summary']
                    st.session_state.last_citations = results.get('citations', [])
                else:
                    st.session_state.last_summary = None
                    st.session_state.last_citations = []
                    st.error("No summary generated.")

        # Display last results
        if st.session_state.last_summary:
            st.markdown("### 🤖 AI-Generated Summary")
            st.markdown(st.session_state.last_summary)
            if st.session_state.last_citations:
                st.markdown("### 📖 Source Citations")
                for idx, cit in enumerate(st.session_state.last_citations, 1):
                    with st.expander(f"Citation {idx} - {cit['document_id']}"):
                        st.write(f"Relevance: {cit['score']:.3f}")
                        st.write(f"> {cit['text']}")
                        # record search history
                st.session_state.search_history.insert(0, {
                    'query': query.strip(),
                    'timestamp': time.strftime('%H:%M:%S'),
                    'results_count': len(st.session_state.last_citations)
                })

        # Search history
        if st.session_state.search_history:
            st.subheader("🕒 Recent Searches")
            for i, h in enumerate(st.session_state.search_history[:5]):
                with st.expander(f"[{h['timestamp']}] {h['query'][:50]}"):
                    st.write(f"Query: {h['query']}")
                    st.write(f"Results: {h['results_count']}")
                    if st.button("Repeat", key=f"rp_{i}"):
                        # pre-fill and re-run
                        st.experimental_set_query_params(query=h['query'])
                        st.rerun()

if __name__ == '__main__':
    main()
