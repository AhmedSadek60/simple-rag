import logging

import httpx

from app.llm.base import LLMError, LLMUnavailableError

logger = logging.getLogger(__name__)

TOGETHER_BASE_URL = "https://api.together.xyz/v1"


class TogetherProvider:
    """Together AI chat completions (OpenAI-compatible API)."""

    def __init__(
        self, api_key: str, model: str, base_url: str = TOGETHER_BASE_URL, timeout: float = 120.0
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    def generate(self, prompt: str) -> str:
        payload = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "stream": False,
        }
        try:
            response = httpx.post(
                f"{self._base_url}/chat/completions",
                json=payload,
                headers={"Authorization": f"Bearer {self._api_key}"},
                timeout=self._timeout,
            )
        except httpx.TransportError as exc:
            logger.error("Cannot reach Together AI: %s", exc)
            raise LLMUnavailableError("Unable to reach the Together AI API.") from exc

        if response.status_code != 200:
            # The response body is logged truncated; the API key is never logged.
            logger.error(
                "Together AI returned HTTP %s for model %s: %s",
                response.status_code,
                self._model,
                response.text[:200],
            )
            raise LLMError(f"Together AI returned HTTP {response.status_code}")

        try:
            answer = str(response.json()["choices"][0]["message"]["content"]).strip()
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise LLMError("Unexpected response format from Together AI") from exc
        if not answer:
            raise LLMError("Together AI returned an empty response")
        logger.info("LLM request succeeded (provider=together, model=%s)", self._model)
        return answer
