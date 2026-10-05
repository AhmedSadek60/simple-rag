import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from app.rag.document_loader import find_documents, load_document, split_pages
from app.rag.vector_store import VectorStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IngestionResult:
    documents: int
    chunks: int


def ingest_documents(
    store: VectorStore,
    directory: Path,
    chunk_size: int,
    chunk_overlap: int,
    on_progress: Callable[[str], None] = lambda _: None,
) -> IngestionResult:
    """Load every supported file in `directory` and rebuild the index from it."""
    paths = find_documents(directory)
    logger.info("Found %d documents in %s", len(paths), directory)
    pages = []
    for path in paths:
        on_progress(path.name)
        try:
            pages.extend(load_document(path))
        except Exception:
            logger.exception("Skipping unreadable document %s", path.name)
    chunks = split_pages(pages, chunk_size, chunk_overlap)
    if not chunks:
        return IngestionResult(documents=len(paths), chunks=0)
    store.rebuild(chunks)
    logger.info("Indexed %d chunks from %d documents", len(chunks), len(paths))
    return IngestionResult(documents=len(paths), chunks=len(chunks))
