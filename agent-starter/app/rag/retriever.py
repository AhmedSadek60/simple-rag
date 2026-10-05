import logging

from app.rag.vector_store import SearchHit, VectorStore

logger = logging.getLogger(__name__)


class Retriever:
    def __init__(self, store: VectorStore, top_k: int, max_distance: float) -> None:
        self._store = store
        self._top_k = top_k
        self._max_distance = max_distance

    def is_ready(self) -> bool:
        return self._store.count() > 0

    def retrieve(self, question: str) -> list[SearchHit]:
        hits = self._store.search(question, self._top_k)
        relevant = [hit for hit in hits if hit.distance <= self._max_distance]
        logger.info("Retrieved %d chunks (%d relevant)", len(hits), len(relevant))
        return relevant
