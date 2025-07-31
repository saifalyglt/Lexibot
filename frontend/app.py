# Streamlit frontend for LexiBot legal document assistant
# This app provides a user interface for legal document upload and query

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

# Set up constants and endpoints
API_URL = "http://localhost:8000"
UPLOAD_ENDPOINT = f"{API_URL}/embed"
SEARCH_ENDPOINT = f"{API_URL}/summarize"
DOCUMENTS_ENDPOINT = f"{API_URL}/documents"

# Initialize session state
if 'uploaded_documents' not in st.session_state:
    st.session_state.uploaded_documents = []
if 'search_history' not in st.session_state:
    st.session_state.search_history = []

# Helper functions for interacting with the FastAPI backend

def check_api_connection() -> bool:
    """Check if the API backend is running."""
    try:
        response = requests.get(API_URL, timeout=5)
        return response.status_code == 200
    except:
        return False

def upload_document(file_data, filename: str) -> Dict[str, Any]:
    """Helper function to upload document to the backend."""
    try:
        files = {'file': (filename, file_data, 'application/octet-stream')}
        response = requests.post(UPLOAD_ENDPOINT, files=files, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.Timeout:
        st.error("Upload timeout - document processing is taking longer than expected")
        return {}
    except Exception as e:
        st.error(f"Error uploading document: {e}")
        return {}

def search_documents(query: str, top_k: int = 5) -> Dict[str, Any]:
    """Helper function to perform legal document search."""
    try:
        payload = {'query': query, 'top_k': top_k}
        response = requests.post(SEARCH_ENDPOINT, json=payload, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.Timeout:
        st.error("Search timeout - please try again")
        return {}
    except Exception as e:
        st.error(f"Error searching documents: {e}")
        return {}

def get_document_list() -> List[Dict[str, Any]]:
    """Get list of uploaded documents."""
    try:
        response = requests.get(DOCUMENTS_ENDPOINT, timeout=10)
        response.raise_for_status()
        return response.json().get('documents', [])
    except Exception as e:
        st.error(f"Error getting document list: {e}")
        return []

# Main UI Layout
st.title("⚖️ LexiBot: Legal Document Assistant")
st.markdown("*AI-powered legal research and document analysis*")

# API Connection Status
api_status = check_api_connection()
if api_status:
    st.success("✅ Connected to LexiBot API")
else:
    st.error("❌ Cannot connect to LexiBot API. Please make sure the backend is running.")
    st.info("Run: `uvicorn backend.main:app --reload` to start the backend")
    st.stop()

# Create two main columns
col1, col2 = st.columns([1, 2])

# Left Column - Document Management
with col1:
    st.header("📄 Document Management")
    
    # Upload section
    st.subheader("Upload Legal Documents")
    uploaded_file = st.file_uploader(
        "Choose a file (PDF or TXT)", 
        type=["pdf", "txt"],
        help="Upload legal documents like court cases, statutes, contracts, etc."
    )
    
    if uploaded_file is not None:
        # Display file details
        file_size_kb = len(uploaded_file.getvalue()) / 1024
        st.info(f"📁 **{uploaded_file.name}**\n\n📏 Size: {file_size_kb:.1f} KB")
        
        # Upload button
        if st.button("🔄 Upload & Process Document", type="primary"):
            with st.spinner("Processing document..."):
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.01)
                    progress_bar.progress(i + 1)
                
                result = upload_document(uploaded_file.getvalue(), uploaded_file.name)
                progress_bar.empty()
                
                if result:
                    st.success("✅ Document uploaded successfully!")
                    st.json(result)
                    # Add to session state
                    st.session_state.uploaded_documents.append({
                        'name': uploaded_file.name,
                        'chunks': result.get('chunks_created', 0),
                        'timestamp': time.strftime('%Y-%m-%d %H:%M')
                    })
                    st.experimental_rerun()
    
    # Document library
    st.subheader("📚 Document Library")
    documents = get_document_list()
    
    if documents:
        for doc in documents:
            with st.expander(f"📄 {doc['document_id']}"):
                st.write(f"**Chunks:** {doc['chunk_count']}")
    else:
        st.info("No documents uploaded yet. Upload your first legal document to get started!")

# Right Column - Search and Results
with col2:
    st.header("🔍 Legal Research")
    
    # Search section
    st.subheader("Ask Your Legal Question")
    
    # Query input
    query_text = st.text_area(
        "Enter your legal question:",
        height=100,
        placeholder="e.g., What are the elements required to prove negligence?",
        help="Ask specific legal questions to get AI-powered answers with citations"
    )
    
    # Search settings
    col_search, col_results = st.columns([2, 1])
    with col_search:
        search_button = st.button("🚀 Search & Analyze", type="primary", disabled=not query_text.strip())
    with col_results:
        top_k_value = st.selectbox("Results:", [3, 5, 7, 10], index=1)
    
    # Search execution
    if search_button and query_text.strip():
        with st.spinner("Analyzing legal documents..."):
            search_results = search_documents(query_text.strip(), top_k_value)
            
            if search_results:
                # Add to search history
                st.session_state.search_history.insert(0, {
                    'query': query_text.strip(),
                    'timestamp': time.strftime('%H:%M:%S'),
                    'results_count': len(search_results.get('citations', []))
                })
                
                # Display results
                st.subheader("📋 AI Analysis & Summary")
                
                # Summary
                if 'summary' in search_results:
                    st.markdown("### 🤖 AI-Generated Summary")
                    st.markdown(search_results['summary'])
                    
                    # Citations
                    if 'citations' in search_results and search_results['citations']:
                        st.markdown("### 📖 Source Citations")
                        
                        for i, citation in enumerate(search_results['citations'], 1):
                            with st.expander(f"Citation [{i}] - {citation['document_id']} (Relevance: {citation['score']:.3f})"):
                                st.markdown(f"**Source:** {citation['document_id']}")
                                st.markdown(f"**Relevance Score:** {citation['score']:.3f}")
                                st.markdown("**Relevant Text:**")
                                st.markdown(f"> {citation['text']}")
                else:
                    st.error("No summary generated. Please check your query and try again.")
    
    # Search history
    if st.session_state.search_history:
        st.subheader("🕒 Recent Searches")
        for i, search in enumerate(st.session_state.search_history[:5]):
            with st.expander(f"[{search['timestamp']}] {search['query'][:50]}..."):
                st.write(f"**Query:** {search['query']}")
                st.write(f"**Results:** {search['results_count']} citations found")
                if st.button(f"Search Again", key=f"search_again_{i}"):
                    st.session_state.query_text = search['query']
                    st.rerun()

# Footer
st.markdown("---")
st.markdown(
    "*LexiBot is powered by AI and provides research assistance. "
    "Always verify legal information with qualified professionals.*"
)
