import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import chromadb

from app.rag.document_loader import Chunk

logger = logging.getLogger(__name__)

COLLECTION_NAME = "documents"


class EmbeddingModel(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


@dataclass(frozen=True)
class SearchHit:
    text: str
    source: str
    page: int | None
    distance: float


class VectorStore:
    """ChromaDB-backed store. Swap this class to move to an external vector database."""

    def __init__(self, persist_directory: Path, embedder: EmbeddingModel) -> None:
        self._client = chromadb.PersistentClient(path=str(persist_directory))
        self._embedder = embedder

    def _collection(self):
        return self._client.get_or_create_collection(
            COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
        )

    def count(self) -> int:
        return self._collection().count()

    def rebuild(self, chunks: list[Chunk], batch_size: int = 64) -> None:
        """Replace the whole index with `chunks`."""
        try:
            self._client.delete_collection(COLLECTION_NAME)
        except Exception:  # collection did not exist yet
            logger.debug("No existing collection to delete")
        collection = self._collection()
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            collection.add(
                ids=[c.id for c in batch],
                documents=[c.text for c in batch],
                embeddings=self._embedder.embed([c.text for c in batch]),
                metadatas=[c.metadata for c in batch],
            )

    def search(self, query: str, top_k: int) -> list[SearchHit]:
        collection = self._collection()
        if collection.count() == 0:
            return []
        result = collection.query(
            query_embeddings=self._embedder.embed([query]),
            n_results=min(top_k, collection.count()),
        )
        return [
            SearchHit(text, str(meta["source"]), meta.get("page"), distance)  # type: ignore[arg-type]
            for text, meta, distance in zip(
                result["documents"][0], result["metadatas"][0], result["distances"][0]
            )
        ]
