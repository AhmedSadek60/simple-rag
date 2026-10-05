import httpx
import pytest
from fastapi.testclient import TestClient

from app.llm.ollama_client import LLMUnavailableError, OllamaProvider
from app.main import create_app
from app.rag.vector_store import SearchHit
from app.services.chat_service import (
    NOT_FOUND_MESSAGE,
    ChatService,
    KnowledgeBaseNotReadyError,
    Source,
)


class FakeRetriever:
    def __init__(self, hits: list[SearchHit], ready: bool = True) -> None:
        self.hits, self.ready = hits, ready

    def is_ready(self) -> bool:
        return self.ready

    def retrieve(self, question: str) -> list[SearchHit]:
        return self.hits


class FakeLLM:
    def __init__(self, answer: str = "25 days.") -> None:
        self.answer, self.prompts = answer, []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.answer


HITS = [
    SearchHit("Employees get 25 days of leave.", "handbook.pdf", 12, 0.2),
    SearchHit("Leave carries over.", "handbook.pdf", 12, 0.3),
    SearchHit("More leave rules.", "faq.txt", None, 0.4),
]
TEMPLATE = "Context:\n{context}\n\nQuestion:\n{question}"


def test_answer_includes_context_and_deduplicated_sources() -> None:
    llm = FakeLLM()
    result = ChatService(FakeRetriever(HITS), llm, TEMPLATE).answer("How much leave?")

    assert result.answer == "25 days."
    assert result.sources == [Source("handbook.pdf", 12), Source("faq.txt", None)]
    assert "Employees get 25 days of leave." in llm.prompts[0]
    assert "How much leave?" in llm.prompts[0]


def test_no_relevant_chunks_skips_the_llm() -> None:
    llm = FakeLLM()
    result = ChatService(FakeRetriever([]), llm, TEMPLATE).answer("Unrelated?")

    assert result.answer == NOT_FOUND_MESSAGE
    assert result.sources == [] and llm.prompts == []


def test_llm_not_found_reply_drops_sources() -> None:
    result = ChatService(FakeRetriever(HITS), FakeLLM(NOT_FOUND_MESSAGE), TEMPLATE).answer("?")
    assert result.answer == NOT_FOUND_MESSAGE and result.sources == []


def test_empty_knowledge_base_raises() -> None:
    service = ChatService(FakeRetriever([], ready=False), FakeLLM(), TEMPLATE)
    with pytest.raises(KnowledgeBaseNotReadyError):
        service.answer("anything")


def client_for(service: ChatService) -> TestClient:
    return TestClient(create_app(chat_service=service))


def test_chat_endpoint_response_shape() -> None:
    client = client_for(ChatService(FakeRetriever(HITS), FakeLLM(), TEMPLATE))
    response = client.post("/api/chat", json={"message": "How much leave?"})

    assert response.status_code == 200
    assert response.json() == {
        "answer": "25 days.",
        "sources": [{"document": "handbook.pdf", "page": 12}, {"document": "faq.txt", "page": None}],
    }


def test_chat_endpoint_reports_ollama_down() -> None:
    class DownLLM:
        def generate(self, prompt: str) -> str:
            raise LLMUnavailableError()

    client = client_for(ChatService(FakeRetriever(HITS), DownLLM(), TEMPLATE))
    response = client.post("/api/chat", json={"message": "hi"})

    assert response.status_code == 503
    assert "Ollama is running" in response.json()["detail"]


def test_chat_endpoint_reports_uninitialized_knowledge_base() -> None:
    client = client_for(ChatService(FakeRetriever([], ready=False), FakeLLM(), TEMPLATE))
    response = client.post("/api/chat", json={"message": "hi"})
    assert response.status_code == 503
    assert response.json()["detail"] in {
        "The document knowledge base has not been initialized.",
        "No documents are currently available.",
    }


def test_chat_endpoint_rejects_blank_message() -> None:
    client = client_for(ChatService(FakeRetriever(HITS), FakeLLM(), TEMPLATE))
    assert client.post("/api/chat", json={"message": "   "}).status_code == 400


def test_ollama_provider_sends_prompt_and_parses_response(monkeypatch) -> None:
    captured = {}

    def fake_post(url, json, timeout):
        captured.update(url=url, json=json)
        return httpx.Response(200, json={"response": " Hello "})

    monkeypatch.setattr(httpx, "post", fake_post)
    answer = OllamaProvider("http://ollama:11434/", "llama3.1").generate("hi")

    assert answer == "Hello"
    assert captured["url"] == "http://ollama:11434/api/generate"
    assert captured["json"]["model"] == "llama3.1" and captured["json"]["stream"] is False


def test_ollama_provider_maps_connection_errors(monkeypatch) -> None:
    def boom(*args, **kwargs):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(httpx, "post", boom)
    with pytest.raises(LLMUnavailableError):
        OllamaProvider("http://localhost:11434", "llama3.1").generate("hi")


def test_together_provider_sends_auth_and_parses_response(monkeypatch) -> None:
    from app.llm.together_client import TogetherProvider

    captured = {}

    def fake_post(url, json, headers, timeout):
        captured.update(url=url, json=json, headers=headers)
        return httpx.Response(200, json={"choices": [{"message": {"content": " Hi "}}]})

    monkeypatch.setattr(httpx, "post", fake_post)
    answer = TogetherProvider("secret-key", "some/model").generate("hello")

    assert answer == "Hi"
    assert captured["url"] == "https://api.together.xyz/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer secret-key"
    assert captured["json"]["messages"] == [{"role": "user", "content": "hello"}]


def test_together_provider_errors_do_not_leak_key(monkeypatch) -> None:
    from app.llm.base import LLMError
    from app.llm.together_client import TogetherProvider

    monkeypatch.setattr(
        httpx, "post", lambda *a, **k: httpx.Response(401, json={"error": "Unauthorized"})
    )
    with pytest.raises(LLMError) as info:
        TogetherProvider("secret-key", "m").generate("x")
    assert "secret-key" not in str(info.value)


def test_together_provider_maps_connection_errors(monkeypatch) -> None:
    from app.llm.together_client import TogetherProvider

    def boom(*args, **kwargs):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(httpx, "post", boom)
    with pytest.raises(LLMUnavailableError, match="Together AI"):
        TogetherProvider("k", "m").generate("x")


def test_build_llm_selects_provider_and_requires_key() -> None:
    from app.core.config import Settings
    from app.llm.together_client import TogetherProvider
    from app.main import build_llm

    assert isinstance(build_llm(Settings(llm_provider="ollama")), OllamaProvider)
    assert isinstance(
        build_llm(Settings(llm_provider="together", together_api_key="k")), TogetherProvider
    )
    with pytest.raises(RuntimeError, match="TOGETHER_API_KEY"):
        build_llm(Settings(llm_provider="together", together_api_key=None, _env_file=None))


def test_ollama_down_shows_friendly_message(monkeypatch) -> None:
    def boom(*args, **kwargs):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(httpx, "post", boom)
    service = ChatService(FakeRetriever(HITS), OllamaProvider("http://x", "m"), TEMPLATE)
    response = client_for(service).post("/api/chat", json={"message": "hi"})

    assert response.status_code == 503
    assert response.json()["detail"] == (
        "Unable to connect to the local LLM. Please make sure Ollama is running."
    )
