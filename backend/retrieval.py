# Document retrieval and Gemini API integration for LexiBot
# This module handles semantic search and AI-powered summary generation

import os
import pickle
import logging
from typing import List, Dict, Any
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentRetriever:
    """
    Handles semantic search using FAISS and summary generation using Gemini API
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", data_dir: str = "../data"):
        """
        Initialize the DocumentRetriever
        
        Args:
            model_name: Name of the sentence transformer model (must match embedder)
            data_dir: Directory containing index files and metadata
        """
        self.model_name = model_name
        self.data_dir = Path(data_dir)
        
        # Initialize sentence transformer model (for query encoding)
        logger.info(f"Loading sentence transformer model: {model_name}")
        self.encoder = SentenceTransformer(model_name)
        
        # FAISS index and metadata storage
        self.index = None
        self.metadata = []
        
        # File paths for persistence
        self.index_path = self.data_dir / "faiss_index.bin"
        self.metadata_path = self.data_dir / "metadata.pkl"
        
        # Configure Gemini API
        self._configure_gemini()
    
    def _configure_gemini(self):
        """Configure Gemini API with the API key from environment"""
        try:
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                logger.warning("GEMINI_API_KEY not found in environment variables")
                self.gemini_model = None
                return
            
            genai.configure(api_key=api_key)
            self.gemini_model = genai.GenerativeModel('gemini-1.5-flash')
            logger.info("Successfully configured Gemini API")
            
        except Exception as e:
            logger.error(f"Error configuring Gemini API: {e}")
            self.gemini_model = None
    
    def load_index(self):
        """Load FAISS index and metadata from disk"""
        try:
            if not self.index_path.exists() or not self.metadata_path.exists():
                raise FileNotFoundError("Index files not found")
            
            # Load FAISS index
            self.index = faiss.read_index(str(self.index_path))
            
            # Load metadata
            with open(self.metadata_path, 'rb') as f:
                self.metadata = pickle.load(f)
            
            logger.info(f"Successfully loaded index with {len(self.metadata)} chunks")
            
        except Exception as e:
            logger.error(f"Error loading index: {e}")
            raise e
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Perform semantic search to find relevant document chunks
        
        Args:
            query: Natural language search query
            top_k: Number of top results to return
            
        Returns:
            List of search results with text, score, and metadata
        """
        try:
            if self.index is None:
                # Try to load index if not already loaded
                self.load_index()
            
            if self.index is None or len(self.metadata) == 0:
                logger.warning("No index or metadata available for search")
                return []
            
            # Encode the query
            query_embedding = self.encoder.encode([query], convert_to_numpy=True)
            
            # Normalize for cosine similarity
            query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
            
            # Perform search
            scores, indices = self.index.search(query_embedding.astype(np.float32), min(top_k, len(self.metadata)))
            
            # Format results
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx < len(self.metadata):  # Valid index
                    result = {
                        "text": self.metadata[idx]["text"],
                        "score": float(score),
                        "document_id": self.metadata[idx]["document_id"],
                        "chunk_id": self.metadata[idx]["chunk_id"],
                        "rank": i + 1
                    }
                    results.append(result)
            
            logger.info(f"Found {len(results)} results for query: {query[:50]}...")
            return results
            
        except Exception as e:
            logger.error(f"Error during search: {e}")
            return []
    
    def generate_summary(self, query: str, search_results: List[Dict[str, Any]]) -> str:
        """
        Generate AI-powered summary using Gemini API based on search results
        
        Args:
            query: Original user query
            search_results: List of relevant document chunks from search
            
        Returns:
            Generated summary with citations
        """
        try:
            if not self.gemini_model:
                return "AI summary generation is not available. Please configure GEMINI_API_KEY."
            
            if not search_results:
                return "No relevant documents found to generate a summary."
            
            # Prepare context from search results
            context_texts = []
            for i, result in enumerate(search_results[:5]):  # Use top 5 results
                citation_num = i + 1
                context_texts.append(f"[{citation_num}] From {result['document_id']}:\n{result['text']}")
            
            context = "\n\n".join(context_texts)
            
            # Create prompt for Gemini
            prompt = f"""You are a legal research assistant. Based on the following legal documents, provide a comprehensive and accurate summary that answers the user's question. Include specific citations to the source documents.

User Question: {query}

Relevant Legal Documents:
{context}

Instructions:
1. Provide a clear, well-structured summary that directly addresses the user's question
2. Include specific citations in the format [1], [2], etc. referring to the numbered documents above
3. Focus on the most relevant legal principles, precedents, or statutory provisions
4. If there are conflicting views or interpretations, mention them
5. Keep the summary concise but comprehensive
6. Only use information from the provided documents - do not add external knowledge

Summary:"""

            # Generate response using Gemini
            response = self.gemini_model.generate_content(prompt)
            
            if response.text:
                logger.info("Successfully generated summary using Gemini API")
                return response.text
            else:
                logger.warning("Gemini API returned empty response")
                return "Unable to generate summary at this time."
                
        except Exception as e:
            logger.error(f"Error generating summary with Gemini: {e}")
            # Fallback to simple concatenation of top results
            return self._generate_fallback_summary(query, search_results)
    
    def _generate_fallback_summary(self, query: str, search_results: List[Dict[str, Any]]) -> str:
        """
        Generate a simple fallback summary when Gemini API is unavailable
        
        Args:
            query: Original user query
            search_results: List of relevant document chunks
            
        Returns:
            Basic summary without AI generation
        """
        if not search_results:
            return "No relevant documents found for your query."
        
        summary_parts = [
            f"Based on your query '{query}', I found {len(search_results)} relevant legal document sections:",
            ""
        ]
        
        for i, result in enumerate(search_results[:3]):  # Show top 3 results
            summary_parts.append(f"{i+1}. From {result['document_id']} (relevance: {result['score']:.2f}):")
            # Truncate long text for fallback summary
            text_preview = result['text'][:300] + "..." if len(result['text']) > 300 else result['text']
            summary_parts.append(f"   {text_preview}")
            summary_parts.append("")
        
        summary_parts.append("Note: This is a basic summary. For AI-generated analysis, please configure the GEMINI_API_KEY environment variable.")
        
        return "\n".join(summary_parts)
    
    def get_similar_documents(self, document_id: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Find documents similar to a given document
        
        Args:
            document_id: ID of the reference document
            top_k: Number of similar documents to return
            
        Returns:
            List of similar documents with similarity scores
        """
        try:
            # Find all chunks from the reference document
            ref_chunks = [item for item in self.metadata if item["document_id"] == document_id]
            
            if not ref_chunks:
                logger.warning(f"Document {document_id} not found in index")
                return []
            
            # Use the first chunk as representative for similarity search
            ref_text = ref_chunks[0]["text"]
            
            # Perform similarity search
            results = self.search(ref_text, top_k + len(ref_chunks))
            
            # Filter out chunks from the same document
            similar_docs = []
            seen_docs = set()
            
            for result in results:
                if result["document_id"] != document_id and result["document_id"] not in seen_docs:
                    similar_docs.append(result)
                    seen_docs.add(result["document_id"])
                    
                    if len(similar_docs) >= top_k:
                        break
            
            return similar_docs
            
        except Exception as e:
            logger.error(f"Error finding similar documents: {e}")
            return []
