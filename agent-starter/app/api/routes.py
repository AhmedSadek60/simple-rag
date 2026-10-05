import logging

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from app.llm.base import LLMError, LLMUnavailableError
from app.services.chat_service import ChatService, KnowledgeBaseNotReadyError

logger = logging.getLogger(__name__)
router = APIRouter()


class ChatRequest(BaseModel):
    # Upper bound is also enforced against settings.max_message_chars in the route.
    message: str = Field(min_length=1, max_length=10_000)


class SourceModel(BaseModel):
    document: str
    page: int | None = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceModel]


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/api/chat", response_model=ChatResponse)
def chat(body: ChatRequest, request: Request) -> ChatResponse:
    max_chars = request.app.state.settings.max_message_chars
    message = body.message.strip()
    if not message or len(message) > max_chars:
        raise HTTPException(400, f"Message must be between 1 and {max_chars} characters.")

    service: ChatService = request.app.state.chat_service
    try:
        result = service.answer(message)
    except KnowledgeBaseNotReadyError:
        has_documents = request.app.state.has_documents()
        raise HTTPException(
            503,
            "The document knowledge base has not been initialized."
            if has_documents
            else "No documents are currently available.",
        )
    except LLMUnavailableError as exc:
        raise HTTPException(503, str(exc))
    except LLMError:
        raise HTTPException(502, "The LLM could not produce an answer. Check the server logs.")
    except Exception:
        logger.exception("Unexpected error while answering a chat request")
        raise HTTPException(500, "Something went wrong. Please try again.")

    return ChatResponse(
        answer=result.answer,
        sources=[SourceModel(document=s.document, page=s.page) for s in result.sources],
    )
