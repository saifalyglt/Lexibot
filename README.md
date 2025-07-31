# LexiBot Legal Document Assistant

LexiBot is a comprehensive legal document assistant built with Streamlit and FastAPI. It allows users to upload legal documents (PDFs and text files), performs semantic search using AI embeddings, and generates intelligent summaries using Google's Gemini API.

## Features

- **Document Upload**: Support for PDF and TXT legal documents
- **Semantic Search**: Uses sentence transformers and FAISS for intelligent document retrieval
- **AI-Powered Summaries**: Leverages Google Gemini API for contextual legal analysis
- **Citation-Enabled**: Provides proper citations to source documents
- **Web Interface**: User-friendly Streamlit frontend
- **REST API**: FastAPI backend for programmatic access

## Project Structure

```
lexibot_app/
├── backend/
│   ├── main.py              # FastAPI application
│   ├── embedding.py         # Document processing and embedding
│   ├── retrieval.py         # Search logic and Gemini integration
│   └── requirements.txt     # Backend dependencies
├── frontend/
│   ├── app.py              # Streamlit user interface
│   └── requirements.txt    # Frontend dependencies
├── data/                   # Storage for documents and index files
└── README.md              # This file
```

## Installation

### Prerequisites

- Python 3.8 or higher
- Google Gemini API key (obtain from Google AI Studio)

### Step 1: Clone or Navigate to the Project Directory

```bash
cd lexibot_app
```

### Step 2: Install Backend Dependencies

```bash
pip install -r backend/requirements.txt
```

### Step 3: Install Frontend Dependencies

```bash
pip install -r frontend/requirements.txt
```

### Step 4: Set Up Environment Variables

Set your Google Gemini API key as an environment variable:

**On Windows (PowerShell):**
```powershell
$env:GEMINI_API_KEY="your_gemini_api_key_here"
```

**On Windows (Command Prompt):**
```cmd
set GEMINI_API_KEY=your_gemini_api_key_here
```

**On Linux/macOS:**
```bash
export GEMINI_API_KEY=your_gemini_api_key_here
```

## Running the Application

### Step 1: Start the Backend API Server

```bash
uvicorn backend.main:app --reload
```

The API will be available at `http://localhost:8000`

### Step 2: Start the Frontend (in a new terminal)

```bash
streamlit run frontend/app.py
```

The web interface will be available at `http://localhost:8501`

## Usage

### 1. Upload Legal Documents

1. Open the Streamlit interface at `http://localhost:8501`
2. Use the sidebar file uploader to select PDF or TXT files
3. Click "Upload Document" to process and index the document
4. The system will split the document into paragraphs and create embeddings

### 2. Query Legal Documents

1. Enter your legal question in the text area
2. Adjust the number of results if needed (1-10)
3. Click "Submit Query" to get an AI-generated summary with citations
4. Review the summary and cited source documents

### 3. API Access

The backend provides REST API endpoints for programmatic access:

- `POST /embed` - Upload and process documents
- `POST /search` - Perform semantic search
- `POST /summarize` - Get AI-generated summaries with citations
- `GET /documents` - List all processed documents

Example API usage:

```python
import requests

# Upload a document
with open('legal_document.pdf', 'rb') as f:
    response = requests.post('http://localhost:8000/embed', 
                           files={'file': f})

# Search and summarize
query = {"query": "What are the requirements for contract formation?", "top_k": 5}
response = requests.post('http://localhost:8000/summarize', json=query)
summary = response.json()
```

## Technical Details

### Document Processing

- **Text Extraction**: PDF text extraction using PyPDF2, direct text reading for TXT files
- **Chunking**: Documents are split into paragraphs with intelligent size management
- **Embeddings**: Uses `all-MiniLM-L6-v2` sentence transformer model
- **Storage**: FAISS index for efficient similarity search

### AI Integration

- **Search**: Semantic similarity search using cosine similarity
- **Summarization**: Google Gemini API for contextual legal analysis
- **Citations**: Automatic citation generation with source document references

### API Endpoints

- `GET /` - Health check
- `POST /embed` - Upload and process documents
- `POST /search` - Semantic search only
- `POST /summarize` - Search + AI summary generation
- `GET /documents` - List processed documents

## Configuration

### Environment Variables

- `GEMINI_API_KEY` - Required for AI summary generation

### File Storage

- Index files are stored in the `data/` directory
- FAISS index: `data/faiss_index.bin`
- Metadata: `data/metadata.pkl`

## Troubleshooting

### Common Issues

1. **"GEMINI_API_KEY not found"**
   - Ensure the environment variable is set correctly
   - Restart both backend and frontend after setting the variable

2. **"Index files not found"**
   - This is normal when starting fresh
   - Upload documents to create the initial index

3. **PDF text extraction issues**
   - Some PDFs may have extraction problems
   - Try converting to TXT format if issues persist

4. **Port conflicts**
   - Backend default: port 8000
   - Frontend default: port 8501
   - Use `--port` flag to change ports if needed

### Logs

Check console output for detailed error messages and processing logs.

## License

This project is provided as-is for educational and development purposes.

## Contributing

Feel free to submit issues and enhancement requests!
"# Lexibot" 
"# Lexibot" 
"# Lexibot" 
"# Lexibot" 
"# Lexibot" 
"# Lexibot" 
