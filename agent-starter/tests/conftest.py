import hashlib
import re

import pytest


class FakeEmbedder:
    """Deterministic bag-of-words embedder so tests need no model download."""

    DIM = 256

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            vector = [0.0] * self.DIM
            for word in re.findall(r"[a-z0-9]+", text.lower()):
                index = int(hashlib.md5(word.encode()).hexdigest(), 16) % self.DIM
                vector[index] += 1.0
            norm = sum(v * v for v in vector) ** 0.5 or 1.0
            vectors.append([v / norm for v in vector])
        return vectors


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder()
