import logging
import httpx

from app.llm.base import LLMError, LLMUnavailableError

logger = logging.getLogger(__name__)


class OllamaProvider:
    """Talks to an Ollama server through its `/api/generate` endpoint."""

    def __init__(self, base_url: str, model: str, timeout: float = 120.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout

    def generate(self, prompt: str) -> str:
        payload = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1},
        }
        try:
            response = httpx.post(
                f"{self._base_url}/api/generate", json=payload, timeout=self._timeout
            )
        except httpx.TransportError as exc:
            logger.error("Cannot reach Ollama at %s: %s", self._base_url, exc)
            raise LLMUnavailableError() from exc

        if response.status_code != 200:
            logger.error(
                "Ollama returned HTTP %s for model %s: %s",
                response.status_code,
                self._model,
                response.text[:200],
            )
            raise LLMError(f"Ollama returned HTTP {response.status_code}")

        answer = str(response.json().get("response", "")).strip()
        if not answer:
            raise LLMError("Ollama returned an empty response")
        logger.info("LLM request succeeded (model=%s)", self._model)
        return answer
