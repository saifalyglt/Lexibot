# Document embedding and FAISS index management for LexiBot
# This module handles document processing, text splitting, embedding generation, and FAISS indexing

import os
import re
import pickle
from typing import List, Dict, Any
import logging
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import PyPDF2

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentEmbedder:
    """
    Handles document processing, embedding generation, and FAISS index management
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", data_dir: str = "../data"):
        """
        Initialize the DocumentEmbedder
        
        Args:
            model_name: Name of the sentence transformer model to use
            data_dir: Directory to store index files and metadata
        """
        self.model_name = model_name
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Initialize sentence transformer model
        logger.info(f"Loading sentence transformer model: {model_name}")
        self.encoder = SentenceTransformer(model_name)
        
        # FAISS index and metadata storage
        self.index = None
        self.metadata = []  # List of dicts containing text, document_id, chunk_id
        self.document_count = 0
        
        # File paths for persistence
        self.index_path = self.data_dir / "faiss_index.bin"
        self.metadata_path = self.data_dir / "metadata.pkl"
        
        # Initialize or load existing index
        self._initialize_index()
    
    def _initialize_index(self):
        """Initialize a new FAISS index or load existing one"""
        try:
            if self.index_path.exists() and self.metadata_path.exists():
                self.load_index()
                logger.info(f"Loaded existing index with {len(self.metadata)} chunks")
            else:
                # Create new index
                embedding_dim = self.encoder.get_sentence_embedding_dimension()
                self.index = faiss.IndexFlatIP(embedding_dim)  # Inner product for cosine similarity
                logger.info(f"Created new FAISS index with dimension {embedding_dim}")
        except Exception as e:
            logger.error(f"Error initializing index: {e}")
            # Fallback: create new index
            embedding_dim = self.encoder.get_sentence_embedding_dimension()
            self.index = faiss.IndexFlatIP(embedding_dim)
            self.metadata = []
    
    def load_index(self):
        """Load existing FAISS index and metadata from disk"""
        try:
            # Load FAISS index
            self.index = faiss.read_index(str(self.index_path))
            
            # Load metadata
            with open(self.metadata_path, 'rb') as f:
                self.metadata = pickle.load(f)
            
            logger.info(f"Successfully loaded index with {len(self.metadata)} chunks")
        except Exception as e:
            logger.error(f"Error loading index: {e}")
            raise e
    
    def save_index(self):
        """Save FAISS index and metadata to disk"""
        try:
            # Save FAISS index
            faiss.write_index(self.index, str(self.index_path))
            
            # Save metadata
            with open(self.metadata_path, 'wb') as f:
                pickle.dump(self.metadata, f)
            
            logger.info(f"Successfully saved index with {len(self.metadata)} chunks")
        except Exception as e:
            logger.error(f"Error saving index: {e}")
            raise e
    
    def _extract_text_from_pdf(self, file_path: str) -> str:
        """
        Extract text content from PDF file
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Extracted text content
        """
        try:
            text = ""
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            return text
        except Exception as e:
            logger.error(f"Error extracting text from PDF {file_path}: {e}")
            raise e
    
    def _extract_text_from_txt(self, file_path: str) -> str:
        """
        Extract text content from TXT file
        
        Args:
            file_path: Path to the TXT file
            
        Returns:
            Text content
        """
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                return file.read()
        except Exception as e:
            logger.error(f"Error reading text file {file_path}: {e}")
            raise e
    
    def _split_into_paragraphs(self, text: str) -> List[str]:
        """
        Split document text into meaningful paragraphs for embedding
        
        Args:
            text: Full document text
            
        Returns:
            List of paragraph chunks
        """
        # Clean up the text
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        text = text.strip()
        
        # Split on double newlines (paragraph breaks) or other logical separators
        paragraphs = re.split(r'\n\s*\n|\n\n+', text)
        
        # Filter out very short paragraphs and clean up
        filtered_paragraphs = []
        for para in paragraphs:
            para = para.strip()
            if len(para) > 50:  # Only keep paragraphs with substantial content
                # Further split very long paragraphs
                if len(para) > 1000:
                    # Split on sentence boundaries for very long paragraphs
                    sentences = re.split(r'(?<=[.!?])\s+', para)
                    current_chunk = ""
                    for sentence in sentences:
                        if len(current_chunk + sentence) < 800:
                            current_chunk += sentence + " "
                        else:
                            if current_chunk.strip():
                                filtered_paragraphs.append(current_chunk.strip())
                            current_chunk = sentence + " "
                    if current_chunk.strip():
                        filtered_paragraphs.append(current_chunk.strip())
                else:
                    filtered_paragraphs.append(para)
        
        logger.info(f"Split document into {len(filtered_paragraphs)} paragraphs")
        return filtered_paragraphs
    
    def process_document(self, file_path: str, document_id: str) -> int:
        """
        Process a document: extract text, split into chunks, generate embeddings, and add to index
        
        Args:
            file_path: Path to the document file
            document_id: Unique identifier for the document
            
        Returns:
            Number of chunks created from the document
        """
        try:
            logger.info(f"Processing document: {document_id}")
            
            # Extract text based on file type
            if file_path.lower().endswith('.pdf'):
                text = self._extract_text_from_pdf(file_path)
            elif file_path.lower().endswith('.txt'):
                text = self._extract_text_from_txt(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_path}")
            
            if not text.strip():
                raise ValueError("No text content found in document")
            
            # Split into paragraphs
            paragraphs = self._split_into_paragraphs(text)
            
            if not paragraphs:
                raise ValueError("No valid paragraphs found after splitting")
            
            # Generate embeddings for all paragraphs
            logger.info(f"Generating embeddings for {len(paragraphs)} paragraphs")
            embeddings = self.encoder.encode(paragraphs, convert_to_numpy=True)
            
            # Normalize embeddings for cosine similarity (required for IndexFlatIP)
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            
            # Add to FAISS index
            self.index.add(embeddings.astype(np.float32))
            
            # Store metadata for each chunk
            for i, paragraph in enumerate(paragraphs):
                self.metadata.append({
                    "text": paragraph,
                    "document_id": document_id,
                    "chunk_id": f"{document_id}_chunk_{i}",
                    "chunk_index": len(self.metadata)
                })
            
            logger.info(f"Successfully processed document {document_id} with {len(paragraphs)} chunks")
            return len(paragraphs)
            
        except Exception as e:
            logger.error(f"Error processing document {document_id}: {e}")
            raise e
    
    def get_document_list(self) -> List[Dict[str, Any]]:
        """
        Get list of all processed documents
        
        Returns:
            List of document information dictionaries
        """
        documents = {}
        for item in self.metadata:
            doc_id = item["document_id"]
            if doc_id not in documents:
                documents[doc_id] = {
                    "document_id": doc_id,
                    "chunk_count": 0
                }
            documents[doc_id]["chunk_count"] += 1
        
        return list(documents.values())
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by the model"""
        return self.encoder.get_sentence_embedding_dimension()
