import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.core.config import BASE_DIR, Settings, get_settings
from app.core.logging import setup_logging
from app.llm.base import LLMProvider
from app.llm.ollama_client import OllamaProvider
from app.llm.together_client import TogetherProvider
from app.rag.document_loader import find_documents
from app.rag.embeddings import Embedder
from app.rag.ingestion import ingest_documents
from app.rag.retriever import Retriever
from app.rag.vector_store import VectorStore
from app.services.chat_service import ChatService

logger = logging.getLogger(__name__)

MAX_BODY_BYTES = 32 * 1024
STATIC_DIR = BASE_DIR / "static"


def create_app(settings: Settings | None = None, chat_service: ChatService | None = None) -> FastAPI:
    """Build the app. `chat_service` can be injected (used by tests)."""
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        setup_logging(settings.log_level)
        logger.info("Starting application (llm_provider=%s)", settings.llm_provider)
        if chat_service is None:
            _wire_services(app, settings)
        yield

    app = FastAPI(title="AI Document Chat", lifespan=lifespan)
    app.state.settings = settings
    app.state.has_documents = lambda: bool(find_documents(settings.documents_directory))
    if chat_service is not None:
        app.state.chat_service = chat_service

    @app.middleware("http")
    async def limit_request_size(request: Request, call_next):
        length = request.headers.get("content-length")
        if length and length.isdigit() and int(length) > MAX_BODY_BYTES:
            return JSONResponse({"detail": "Request too large."}, status_code=413)
        return await call_next(request)

    app.include_router(router)
    if STATIC_DIR.is_dir():
        app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
    return app


def build_llm(settings: Settings) -> LLMProvider:
    if settings.llm_provider == "together":
        if settings.together_api_key is None:
            raise RuntimeError("LLM_PROVIDER=together requires TOGETHER_API_KEY to be set.")
        return TogetherProvider(
            settings.together_api_key.get_secret_value(),
            settings.together_model,
            timeout=settings.ollama_timeout_seconds,
        )
    return OllamaProvider(
        settings.ollama_base_url, settings.ollama_model, settings.ollama_timeout_seconds
    )


def _wire_services(app: FastAPI, settings: Settings) -> None:
    Path(settings.chroma_persist_directory).mkdir(parents=True, exist_ok=True)
    store = VectorStore(settings.chroma_persist_directory, Embedder(settings.embedding_model))

    if store.count() == 0 and settings.ingest_on_startup:
        logger.info("Vector store is empty; ingesting documents")
        ingest_documents(
            store, settings.documents_directory, settings.chunk_size, settings.chunk_overlap
        )

    retriever = Retriever(store, settings.top_k, settings.max_distance)
    llm = build_llm(settings)
    app.state.chat_service = ChatService(retriever, llm, ChatService.load_prompt())


app = create_app()
