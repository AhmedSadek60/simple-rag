from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Application settings, loaded from environment variables and `.env`."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1"
    ollama_timeout_seconds: float = 120.0

    embedding_model: str = "all-MiniLM-L6-v2"
    chroma_persist_directory: Path = BASE_DIR / "data" / "chroma"
    documents_directory: Path = BASE_DIR / "documents"

    top_k: int = 5
    # Chunks with a cosine distance above this are treated as not relevant.
    max_distance: float = 0.9
    chunk_size: int = 1000
    chunk_overlap: int = 150

    # Build the index at startup when it is empty (useful where the disk is ephemeral).
    ingest_on_startup: bool = True
    max_message_chars: int = 2000

    log_level: str = "INFO"
    port: int = 8000


@lru_cache
def get_settings() -> Settings:
    return Settings()
