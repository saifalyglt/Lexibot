# FastAPI backend for LexiBot legal document assistant
# This module provides REST API endpoints for document embedding and semantic search

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import tempfile
from typing import List, Dict, Any
import logging

from .embedding import DocumentEmbedder
from .retrieval import DocumentRetriever

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="LexiBot API",
    description="Legal document processing and semantic search API",
    version="1.0.0"
)

# Add CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
embedder = DocumentEmbedder()
retriever = DocumentRetriever()

# Pydantic models for request/response validation
class SearchQuery(BaseModel):
    query: str
    top_k: int = 5

class SearchResult(BaseModel):
    text: str
    score: float
    document_id: str
    
class SummaryResponse(BaseModel):
    summary: str
    citations: List[SearchResult]

@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup"""
    logger.info("Starting LexiBot API...")
    # Load existing FAISS index if it exists
    try:
        retriever.load_index()
        logger.info("Loaded existing FAISS index")
    except Exception as e:
        logger.info(f"No existing index found or failed to load: {e}")

@app.get("/")
async def root():
    """Root endpoint for health check"""
    return {"message": "LexiBot API is running"}

@app.post("/embed")
async def embed_document(file: UploadFile = File(...)):
    """
    Endpoint to upload and embed a legal document
    
    Args:
        file: PDF or TXT file to be processed and embedded
        
    Returns:
        Success message with document processing details
    """
    try:
        # Validate file type
        if not file.filename.lower().endswith(('.pdf', '.txt')):
            raise HTTPException(
                status_code=400, 
                detail="Only PDF and TXT files are supported"
            )
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        try:
            # Process and embed the document
            chunks_count = embedder.process_document(tmp_path, file.filename)
            
            # Save updated index
            embedder.save_index()
            
            return {
                "message": f"Document '{file.filename}' processed successfully",
                "chunks_created": chunks_count,
                "document_id": file.filename
            }
            
        finally:
            # Clean up temporary file
            os.unlink(tmp_path)
            
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")

@app.post("/search", response_model=List[SearchResult])
async def search_documents(query: SearchQuery):
    """
    Endpoint to perform semantic search on embedded documents
    
    Args:
        query: SearchQuery object containing the search query and top_k parameter
        
    Returns:
        List of search results with text snippets and relevance scores
    """
    try:
        # Perform semantic search
        results = retriever.search(query.query, top_k=query.top_k)
        
        # Format results for response
        formatted_results = [
            SearchResult(
                text=result["text"],
                score=float(result["score"]),
                document_id=result["document_id"]
            )
            for result in results
        ]
        
        return formatted_results
        
    except Exception as e:
        logger.error(f"Error during search: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error during search: {str(e)}")

@app.post("/summarize", response_model=SummaryResponse)
async def summarize_query(query: SearchQuery):
    """
    Endpoint to get AI-generated summary with citations for a legal query
    
    Args:
        query: SearchQuery object containing the legal question
        
    Returns:
        AI-generated summary with relevant citations from legal documents
    """
    try:
        # First, perform semantic search to get relevant documents
        search_results = retriever.search(query.query, top_k=query.top_k)
        
        if not search_results:
            return SummaryResponse(
                summary="No relevant legal documents found for your query.",
                citations=[]
            )
        
        # Generate summary using Gemini API
        summary = retriever.generate_summary(query.query, search_results)
        
        # Format citations
        citations = [
            SearchResult(
                text=result["text"],
                score=float(result["score"]),
                document_id=result["document_id"]
            )
            for result in search_results
        ]
        
        return SummaryResponse(
            summary=summary,
            citations=citations
        )
        
    except Exception as e:
        logger.error(f"Error generating summary: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating summary: {str(e)}")

@app.get("/documents")
async def list_documents():
    """
    Endpoint to list all processed documents in the index
    
    Returns:
        List of document IDs and metadata
    """
    try:
        documents = embedder.get_document_list()
        return {"documents": documents}
    except Exception as e:
        logger.error(f"Error listing documents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error listing documents: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
