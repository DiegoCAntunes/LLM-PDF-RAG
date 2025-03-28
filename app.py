"""
Enhanced RAG System with Semantic Caching
Improved version with better structure, error handling, and performance
"""

# Standard Library Imports
import asyncio
import csv
import logging
import os
import tempfile
from functools import lru_cache
from io import StringIO
from typing import AsyncGenerator, List, Optional, Tuple

# Third-party Imports
import chromadb
import ollama
import streamlit as st
from chromadb.utils.embedding_functions.ollama_embedding_function import (
    OllamaEmbeddingFunction,
)
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_redis import RedisConfig, RedisVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from streamlit.runtime.uploaded_file_manager import UploadedFile

# Local Config
from config import settings  # Centralized configuration

# ======================
# LOGGING CONFIGURATION
# ======================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ======================
# SYSTEM PROMPT
# ======================
system_prompt = """
You are an AI assistant tasked with providing detailed answers based solely on the given context. 
Include citations like [doc-1] when referencing specific documents.

Context will be passed as "Context:"
User question will be passed as "Question:"

Guidelines:
1. Analyze context thoroughly before answering
2. Structure response with clear paragraphs and bullet points when needed
3. Cite sources using document IDs provided
4. If context is insufficient, state this clearly

Response format:
- Clear, concise language
- Proper citations for all facts [doc-X]
- Well-organized structure
- No external knowledge
"""

# ======================
# CORE FUNCTIONS
# ======================

def get_redis_store() -> RedisVectorStore:
    """
    Gets or creates a Redis vector store with improved error handling.
    
    Returns:
        RedisVectorStore: Configured Redis vector store instance
        
    Raises:
        Exception: If connection to Redis fails
    """
    try:
        # Initialize embeddings with configured model
        embeddings = OllamaEmbeddings(
            model=settings.EMBEDDING_MODEL,
        )
        
        # Create Redis vector store with proper configuration
        return RedisVectorStore(
            embeddings,
            config=RedisConfig(
                index_name="cached_contents",
                redis_url=settings.REDIS_URL,
                distance_metric="COSINE",
                metadata_schema=[
                    {"name": "answer", "type": "text"},
                    {"name": "source", "type": "text"},  # Added source tracking
                ],
            ),
        )
    except Exception as e:
        logger.error(f"Redis connection failed: {str(e)}")
        st.error("Failed to connect to Redis cache")
        raise


def create_cached_contents(uploaded_file: UploadedFile) -> list[Document]:
    """
    Creates cached QA pairs from uploaded CSV with validation and batch processing.
    
    Args:
        uploaded_file: Uploaded CSV file containing question/answer pairs
        
    Returns:
        list[Document]: List of created Document objects
        
    Raises:
        ValueError: If CSV format is invalid
        Exception: If cache creation fails
    """
    try:
        # Read and decode uploaded file
        data = uploaded_file.getvalue().decode("utf-8")
        csv_reader = csv.DictReader(StringIO(data))
        
        # Validate CSV structure
        if not all(col in csv_reader.fieldnames for col in ["question", "answer"]):
            raise ValueError("CSV must contain 'question' and 'answer' columns")
            
        # Create documents with metadata
        docs = [
            Document(
                page_content=row["question"],
                metadata={
                    "answer": row["answer"],
                    "source": row.get("source", "unknown"),
                }
            )
            for row in csv_reader
        ]
        
        # Get Redis store and process in batches
        vector_store = get_redis_store()
        
        for i in range(0, len(docs), settings.BATCH_SIZE):
            batch = docs[i:i + settings.BATCH_SIZE]
            vector_store.add_documents(batch)
            
        st.success(f"Added {len(docs)} cached items successfully!")
        return docs
        
    except Exception as e:
        logger.error(f"Cache creation failed: {str(e)}")
        st.error(f"Failed to create cache: {str(e)}")
        raise


@lru_cache(maxsize=1000)  # Cache up to 1000 queries
def query_semantic_cache(query: str, n_results: int = 1) -> Optional[list]:
    """
    Enhanced semantic cache query with logging and metrics.
    
    Args:
        query: Search query text
        n_results: Number of results to return
        
    Returns:
        Optional[list]: List of results if found, None otherwise
    """
    try:
        vector_store = get_redis_store()
        results = vector_store.similarity_search_with_score(query, k=n_results)
        
        if not results:
            logger.debug("No cache results found")
            return None
            
        # Calculate match score percentage
        best_score = (1 - abs(results[0][1])) * 100
        logger.info(f"Cache query score: {best_score:.2f}%")
        
        # Return results if above threshold
        if best_score >= settings.CACHE_THRESHOLD:
            return results
        return None
            
    except Exception as e:
        logger.error(f"Cache query failed: {str(e)}")
        return None


def process_document(uploaded_file: UploadedFile) -> list[Document]:
    """
    Processes uploaded PDF with better temp file handling and chunking.
    
    Args:
        uploaded_file: Uploaded PDF file
        
    Returns:
        list[Document]: List of processed document chunks
        
    Raises:
        Exception: If processing fails
    """
    try:
        # Create temp file safely
        with tempfile.NamedTemporaryFile("wb", suffix=".pdf", delete=False) as temp_file:
            temp_file.write(uploaded_file.read())
            temp_path = temp_file.name
            
        try:
            # Load and split document
            loader = PyMuPDFLoader(temp_path)
            docs = loader.load()
            
            # Configure text splitter with proper chunking
            text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                chunk_size=settings.CHUNK_SIZE,
                chunk_overlap=settings.CHUNK_OVERLAP,
                separators=["\n\n", "\n", ".", "?", "!", " ", ""],
            )
            return text_splitter.split_documents(docs)
            
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
                
    except Exception as e:
        logger.error(f"Document processing failed: {str(e)}")
        st.error("Failed to process document")
        raise


def get_vector_collection() -> chromadb.Collection:
    """
    Gets ChromaDB collection with persistent client and error handling.
    
    Returns:
        chromadb.Collection: Configured ChromaDB collection
        
    Raises:
        Exception: If collection access fails
    """
    try:
        # Initialize embedding function
        ollama_ef = OllamaEmbeddingFunction(
            url=settings.OLLAMA_URL,
            model_name=settings.EMBEDDING_MODEL,
        )

        # Get or create persistent collection
        chroma_client = chromadb.PersistentClient(path="./demo-rag-chroma")
        return chroma_client.get_or_create_collection(
            name="rag_app",
            embedding_function=ollama_ef,
            metadata={"hnsw:space": "cosine"},
        )
    except Exception as e:
        logger.error(f"ChromaDB collection error: {str(e)}")
        st.error("Failed to access vector database")
        raise


def add_to_vector_collection(all_splits: list[Document], file_name: str):
    """
    Adds document splits to vector collection with batch processing.
    
    Args:
        all_splits: List of document chunks
        file_name: Base name for document IDs
        
    Raises:
        Exception: If update fails
    """
    try:
        collection = get_vector_collection()
        
        # Process in configured batch sizes
        for i in range(0, len(all_splits), settings.BATCH_SIZE):
            batch = all_splits[i:i + settings.BATCH_SIZE]
            
            # Prepare batch data
            documents = [split.page_content for split in batch]
            metadatas = [split.metadata for split in batch]
            ids = [f"{file_name}_{i+idx}" for idx, _ in enumerate(batch)]
            
            # Upsert batch
            collection.upsert(
                documents=documents,
                metadatas=metadatas,
                ids=ids,
            )
            
        st.success(f"Added {len(all_splits)} chunks to vector store!")
        
    except Exception as e:
        logger.error(f"Vector collection update failed: {str(e)}")
        st.error("Failed to update vector store")
        raise


@lru_cache(maxsize=1000)  # Cache query results
def query_collection(prompt: str, n_results: int = 10) -> dict:
    """
    Cached query to vector collection.
    
    Args:
        prompt: Search query
        n_results: Number of results to return
        
    Returns:
        dict: Query results with documents, scores and metadata
    """
    try:
        collection = get_vector_collection()
        return collection.query(query_texts=[prompt], n_results=n_results)
    except Exception as e:
        logger.error(f"Collection query failed: {str(e)}")
        return {"documents": [[]], "distances": [[]], "metadatas": [[]]}


def hybrid_search(prompt: str, n_results: int = 10) -> dict:
    """
    Combines semantic and keyword search for better results.
    
    Args:
        prompt: Search query
        n_results: Number of results to return
        
    Returns:
        dict: Combined search results
    """
    try:
        # Get semantic results (extra for filtering)
        semantic_results = query_collection(prompt, n_results * 2)
        
        if not semantic_results["documents"][0]:
            return semantic_results
            
        # Setup BM25 keyword search
        documents = semantic_results["documents"][0]
        tokenized_docs = [doc.split() for doc in documents]
        bm25 = BM25Okapi(tokenized_docs)
        
        # Get keyword scores
        tokenized_query = prompt.split()
        doc_scores = bm25.get_scores(tokenized_query)
        
        # Normalize scores
        max_semantic = max(semantic_results["distances"][0]) or 1
        max_bm25 = max(doc_scores) or 1
        
        # Combine scores with configured weighting
        combined_results = []
        for i, doc in enumerate(documents):
            semantic_norm = semantic_results["distances"][0][i] / max_semantic
            bm25_norm = doc_scores[i] / max_bm25
            combined = (settings.HYBRID_SEARCH_ALPHA * semantic_norm + 
                       (1 - settings.HYBRID_SEARCH_ALPHA) * bm25_norm)
            combined_results.append({
                "document": doc,
                "score": combined,
                "metadata": semantic_results["metadatas"][0][i],
                "id": semantic_results["ids"][0][i]
            })
        
        # Return top results
        combined_results.sort(key=lambda x: x["score"], reverse=True)
        top_results = combined_results[:n_results]
        
        return {
            "documents": [[r["document"] for r in top_results]],
            "distances": [[r["score"] for r in top_results]],
            "metadatas": [[r["metadata"] for r in top_results]],
            "ids": [[r["id"] for r in top_results]]
        }
        
    except Exception as e:
        logger.error(f"Hybrid search failed: {str(e)}")
        return query_collection(prompt, n_results)  # Fallback to semantic


def re_rank_cross_encoders(documents: List[str], prompt: str) -> Tuple[str, List[int]]:
    """
    Re-ranks documents using cross-encoder for better relevance.
    
    Args:
        documents: List of document texts
        prompt: Original query
        
    Returns:
        Tuple: (concatenated relevant text, list of relevant indices)
    """
    try:
        if not documents:
            return "", []
            
        # Initialize cross-encoder model
        encoder_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        
        # Get top 3 most relevant documents
        ranks = encoder_model.rank(prompt, documents, top_k=min(3, len(documents)))
        
        # Format results with document citations
        relevant_text = ""
        relevant_text_ids = []
        
        for rank in ranks:
            relevant_text += f"[doc-{rank['corpus_id']}] {documents[rank['corpus_id']]}\n\n"
            relevant_text_ids.append(rank["corpus_id"])
            
        return relevant_text, relevant_text_ids
        
    except Exception as e:
        logger.error(f"Reranking failed: {str(e)}")
        return "\n".join(documents), list(range(len(documents)))


async def call_llm(context: str, prompt: str, doc_ids: List[str]) -> AsyncGenerator[str, None]:
    """
    Async LLM call with citation support.
    
    Args:
        context: Relevant context text
        prompt: User question
        doc_ids: List of document IDs for citation
        
    Yields:
        str: Response chunks as they're generated
    """
    try:
        # Enhance prompt with citation instructions
        citation_prompt = f"Available document IDs for citation: {', '.join(doc_ids)}"
        
        # Run Ollama in separate thread (async wrapper)
        response = await asyncio.to_thread(
            ollama.chat,
            model=settings.LLM_MODEL,
            stream=True,
            messages=[
                {"role": "system", "content": system_prompt + citation_prompt},
                {"role": "user", "content": f"Context: {context}\nQuestion: {prompt}"},
            ],
        )
        
        # Stream response chunks
        for chunk in response:
            if not chunk["done"]:
                yield chunk["message"]["content"]
                
    except Exception as e:
        logger.error(f"LLM call failed: {str(e)}")
        yield "Sorry, I encountered an error generating the response."


# ======================
# STREAMLIT UI
# ======================
def main():
    """Main Streamlit application with improved UI and workflow."""
    # Configure page
    st.set_page_config(page_title="Enhanced RAG System", layout="wide")
    
    # Sidebar for document processing
    with st.sidebar:
        st.title("📂 Document Processing")
        uploaded_file = st.file_uploader(
            "Upload PDF or CSV files",
            type=["pdf", "csv"],
            accept_multiple_files=False,
            help="PDF for documents, CSV for cache"
        )
        
        # Upload type selector
        upload_option = st.radio(
            "Upload type:",
            options=["Primary Document", "Cache Data"],
            index=0,
            help="Select whether you're uploading a document or cache data"
        )
        
        # Process button
        if st.button("⚡ Process Data", type="primary"):
            if uploaded_file:
                try:
                    # Normalize filename
                    norm_name = uploaded_file.name.translate(
                        str.maketrans({"-": "_", ".": "_", " ": "_"})
                    )
                    
                    # Handle cache vs document upload
                    if upload_option == "Cache Data":
                        if not uploaded_file.name.endswith(".csv"):
                            st.error("Cache uploads must be CSV files")
                        else:
                            with st.spinner("Processing cache..."):
                                create_cached_contents(uploaded_file)
                    else:
                        if uploaded_file.name.endswith(".csv"):
                            st.error("PDF required for document upload")
                        else:
                            with st.spinner("Processing document..."):
                                splits = process_document(uploaded_file)
                                add_to_vector_collection(splits, norm_name)
                except Exception:
                    st.error("Processing failed - check logs for details")
            else:
                st.warning("Please upload a file first")

    # Main content area
    st.title("🧠 Enhanced RAG Question Answering")
    
    # Two-column layout
    col1, col2 = st.columns([3, 1])
    with col1:
        prompt = st.text_area("Ask your question:", height=150)
    with col2:
        st.markdown("### Settings")
        n_results = st.slider("Results to retrieve", 1, 20, 5)
        search_type = st.radio("Search type", ["Hybrid", "Semantic", "Keyword"])
    
    # Search button
    if st.button("🔍 Search", type="primary") and prompt:
        with st.spinner("Searching..."):
            try:
                # Check cache first
                if cached := query_semantic_cache(prompt):
                    st.success("Found cached answer!")
                    st.write(cached[0][0].metadata["answer"].replace("\\n", "\n"))
                    st.caption(f"Source: {cached[0][0].metadata.get('source', 'unknown')}")
                    return
                
                # Perform selected search type
                if search_type == "Hybrid":
                    results = hybrid_search(prompt, n_results)
                elif search_type == "Keyword":
                    results = hybrid_search(prompt, n_results)  # Fallback
                else:
                    results = query_collection(prompt, n_results)
                
                # Handle no results
                if not results["documents"][0]:
                    st.warning("No relevant documents found")
                    return
                
                # Rerank and prepare context
                context, relevant_ids = re_rank_cross_encoders(results["documents"][0], prompt)
                doc_ids = [results["ids"][0][i] for i in relevant_ids]
                
                # Display answer
                st.subheader("Answer")
                response_area = st.empty()
                full_response = ""
                
                # Async response streaming
                async def stream_response():
                    nonlocal full_response
                    async for chunk in call_llm(context, prompt, doc_ids):
                        full_response += chunk
                        response_area.markdown(full_response)
                
                asyncio.run(stream_response())
                
                # Show details in expanders
                with st.expander("📄 Retrieved Documents"):
                    st.json(results["documents"][0])
                
                with st.expander("🔍 Most Relevant Documents"):
                    for i, idx in enumerate(relevant_ids):
                        st.markdown(f"### [doc-{i+1}] Score: {results['distances'][0][idx]:.2f}")
                        st.write(results["documents"][0][idx])
                        st.divider()
                        
            except Exception as e:
                st.error(f"Search failed: {str(e)}")
                logger.exception("Search error")


if __name__ == "__main__":
    main()