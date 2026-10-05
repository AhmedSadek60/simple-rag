from fastapi.testclient import TestClient

from app.main import create_app
from app.services.chat_service import ChatResult, ChatService


class StubService(ChatService):
    def __init__(self) -> None:
        pass

    def answer(self, message: str) -> ChatResult:
        return ChatResult("ok", [])


def test_health_returns_ok() -> None:
    client = TestClient(create_app(chat_service=StubService()))
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_oversized_request_is_rejected() -> None:
    client = TestClient(create_app(chat_service=StubService()))
    response = client.post("/api/chat", json={"message": "x" * 100_000})
    assert response.status_code == 413
