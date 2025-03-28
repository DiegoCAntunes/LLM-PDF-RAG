from pydantic_settings import BaseSettings  # Changed import
from pydantic import Field  # Optional: for field configuration

class Settings(BaseSettings):
    REDIS_URL: str = "redis://localhost:6379"
    OLLAMA_URL: str = "http://localhost:11434/api/embeddings"
    EMBEDDING_MODEL: str = "nomic-embed-text:latest"
    LLM_MODEL: str = "llama3.2:3b"
    CHUNK_SIZE: int = 400
    CHUNK_OVERLAP: int = 100
    CACHE_THRESHOLD: float = 80.0
    BATCH_SIZE: int = 100
    HYBRID_SEARCH_ALPHA: float = Field(default=0.7, ge=0, le=1)  # Example with validation
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"  # Recommended addition

settings = Settings()