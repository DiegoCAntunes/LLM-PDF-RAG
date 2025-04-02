# 📚 LLM-RAG with Hybrid Search & Cross-Encoder Re-ranking

A powerful local RAG (Retrieval-Augmented Generation) system that combines semantic search with keyword matching and cross-encoder re-ranking for accurate, citation-backed answers from your PDF documents.

![RAG System Diagram](https://i.imgur.com/oPX2tRA.jpeg)

## 🌟 Key Features
- **Hybrid Search** - Combines semantic meaning and keyword matching
- **Smart Caching** - Redis-backed cache for frequent queries
- **Citation Support** - Automatic `[doc-1]` style references in answers
- **Local Processing** - Runs entirely on your machine with Ollama
- **PDF & CSV Support** - Process documents or pre-cached Q&A pairs

## 🚀 Quick Start

### Prerequisites
- [Ollama](https://ollama.ai/) installed and running
- Python 3.10+ (recommended: 3.11+)
- Redis server (for caching)

```bash
# Clone the repository
git clone https://github.com/yourusername/llm-rag.git
cd llm-rag

# Set up environment
make setup

# Launch the application
make run
```

## 🛠️ Installation Details

### 1. Install System Dependencies

#### MacOS:
```bash
brew install ollama redis
```

#### Linux (Ubuntu/Debian):
```bash
curl -fsSL https://ollama.com/install.sh | sh
sudo apt install redis-server
```

### 2. Set Up Python Environment
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or .venv\Scripts\activate  # Windows

pip install -r requirements.txt
```

### 3. Download Required Models
```bash
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

## 🖥️ Usage Guide

### Launch the application:
```bash
make run
```

### Upload Documents:
- Use the sidebar to upload PDFs or CSVs
- Click "Process" to ingest the documents

### Ask Questions:
- Type your question in the main text area
- Choose search type (Hybrid recommended)
- Click "Search" to get answers with citations

## ⚙️ Configuration
Edit `config.py` to customize:

```python
class Settings(BaseSettings):
    LLM_MODEL: str = "llama3.2:3b"
    HYBRID_SEARCH_ALPHA: float = 0.7  # 70% semantic, 30% keyword
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
```

## 🧠 Supported Models

| Model Type | Recommended Options |
|------------|---------------------|
| LLM       | `llama3.2:3b`         |
