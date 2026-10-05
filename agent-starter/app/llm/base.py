from typing import Protocol

OLLAMA_DOWN_MESSAGE = "Unable to connect to the local LLM. Please make sure Ollama is running."


class LLMError(Exception):
    """The LLM returned an error or an unusable response."""


class LLMUnavailableError(LLMError):
    """The LLM endpoint could not be reached. `str(exc)` is safe to show to users."""

    def __init__(self, user_message: str = OLLAMA_DOWN_MESSAGE) -> None:
        super().__init__(user_message)


class LLMProvider(Protocol):
    def generate(self, prompt: str) -> str: ...
