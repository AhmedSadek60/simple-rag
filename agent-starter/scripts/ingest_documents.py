"""Rebuild the ChromaDB index from the files in the documents directory."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import get_settings  # noqa: E402
from app.core.logging import setup_logging  # noqa: E402
from app.rag.document_loader import find_documents  # noqa: E402
from app.rag.embeddings import Embedder  # noqa: E402
from app.rag.ingestion import ingest_documents  # noqa: E402
from app.rag.vector_store import VectorStore  # noqa: E402


def main() -> int:
    settings = get_settings()
    setup_logging(settings.log_level)

    found = find_documents(settings.documents_directory)
    print(f"Found {len(found)} documents in {settings.documents_directory}.\n")
    if not found:
        print("Nothing to index. Add PDF, DOCX, TXT or MD files and run again.")
        return 1

    settings.chroma_persist_directory.mkdir(parents=True, exist_ok=True)
    store = VectorStore(settings.chroma_persist_directory, Embedder(settings.embedding_model))

    print("Processing:")
    result = ingest_documents(
        store,
        settings.documents_directory,
        settings.chunk_size,
        settings.chunk_overlap,
        on_progress=print,
    )
    if result.chunks == 0:
        print("\nNo text could be extracted from the documents.")
        return 1

    print(f"\nCreated {result.chunks} chunks.")
    print("Vector database successfully created.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
