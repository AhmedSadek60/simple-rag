# Architecture

A single FastAPI service (`app/`) serving a static chat page (`static/`).

Request flow for `POST /api/chat`:

1. `app/api/routes.py` validates the message and calls `ChatService`.
2. `ChatService` (`app/services/chat_service.py`) asks `Retriever` for relevant chunks.
3. `Retriever` queries `VectorStore` (ChromaDB, cosine distance), keeps chunks within `MAX_DISTANCE`.
4. The chunks and question are formatted into `app/prompts/rag_prompt.txt` and sent to an `LLMProvider` (`OllamaProvider`).
5. The answer and de-duplicated sources (document, PDF page) are returned. With no relevant chunks the LLM is skipped.

Indexing (`scripts/ingest_documents.py`, or automatically at startup when the index is empty): `documents/` -> loaders (pypdf, python-docx, plain text) -> LangChain `RecursiveCharacterTextSplitter` -> sentence-transformers embeddings -> ChromaDB at `CHROMA_PERSIST_DIRECTORY`.

Deployment: Docker image; Railway via `railway.toml`. Ollama is an external service reached through `OLLAMA_BASE_URL`. The Railway filesystem is ephemeral, so the index is rebuilt at startup.

Extension points: implement `LLMProvider` for another LLM; replace `VectorStore` for an external vector database.
