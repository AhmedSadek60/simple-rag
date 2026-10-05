import logging
from dataclasses import dataclass
from pathlib import Path

from app.llm.base import LLMProvider
from app.rag.retriever import Retriever

logger = logging.getLogger(__name__)

NOT_FOUND_MESSAGE = "I couldn't find this information in the available documents."
PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "rag_prompt.txt"


class KnowledgeBaseNotReadyError(Exception):
    """The vector store is empty or has not been built."""


@dataclass(frozen=True)
class Source:
    document: str
    page: int | None


@dataclass(frozen=True)
class ChatResult:
    answer: str
    sources: list[Source]


class ChatService:
    def __init__(self, retriever: Retriever, llm: LLMProvider, prompt_template: str) -> None:
        self._retriever = retriever
        self._llm = llm
        self._prompt_template = prompt_template

    @classmethod
    def load_prompt(cls) -> str:
        return PROMPT_PATH.read_text(encoding="utf-8")

    def answer(self, message: str) -> ChatResult:
        if not self._retriever.is_ready():
            raise KnowledgeBaseNotReadyError()

        hits = self._retriever.retrieve(message)
        if not hits:
            return ChatResult(NOT_FOUND_MESSAGE, [])

        context = "\n\n".join(f"[{hit.source}]\n{hit.text}" for hit in hits)
        prompt = self._prompt_template.format(context=context, question=message)
        answer = self._llm.generate(prompt)

        if NOT_FOUND_MESSAGE.rstrip(".") in answer:
            return ChatResult(NOT_FOUND_MESSAGE, [])

        sources = list(dict.fromkeys(Source(hit.source, hit.page) for hit in hits))
        return ChatResult(answer, sources)
