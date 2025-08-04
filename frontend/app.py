# Streamlit frontend for LexiBot legal document assistant
# Updated to connect to Hugging Face Space backend

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
# Replace localhost with your HF Space domain
API_URL = "https://saifaligzr-Lexibot_app.hf.space"
UPLOAD_ENDPOINT = f"{API_URL}/embed"
SEARCH_ENDPOINT = f"{API_URL}/summarize"
DOCUMENTS_ENDPOINT = f"{API_URL}/documents"

# Initialize session state
if 'uploaded_documents' not in st.session_state:
    st.session_state.uploaded_documents = []
if 'search_history' not in st.session_state:
    st.session_state.search_history = []

# Helper functions

def check_api_connection() -> bool:
    """Check if the API backend is running."""
    try:
        response = requests.get(API_URL, timeout=5)
        return response.status_code == 200
    except Exception:
        return False


def upload_document(file_data, filename: str) -> Dict[str, Any]:
    """Upload a document to the backend."""
    try:
        files = {'file': (filename, file_data, 'application/octet-stream')}
        response = requests.post(UPLOAD_ENDPOINT, files=files, timeout=60)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.Timeout:
        st.error("Upload timeout - please try a smaller file or check your connection.")
        return {}
    except Exception as e:
        st.error(f"Error uploading document: {e}")
        return {}


def search_documents(query: str, top_k: int = 5) -> Dict[str, Any]:
    """Perform a legal document search."""
    try:
