"""Step 2 of code search: turn text into vectors (lists of numbers that capture meaning)."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class Embedder(Protocol):
    """Anything that can turn text into vectors. Tests use a fake; bloom uses Gemini."""

    model: str
    dimensions: int

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Vectors for code chunks. Shape: (len(texts), dimensions)."""

    def embed_query(self, text: str) -> np.ndarray:
        """One vector for a search question. Shape: (dimensions,)."""


def normalize(vectors: np.ndarray) -> np.ndarray:
    """Scale every vector to length 1.

    With length-1 vectors, the inner product of two vectors equals their cosine similarity:
    1.0 means "same direction" (same meaning), 0 means unrelated.
    """
    array = np.asarray(vectors, dtype=np.float32)
    lengths = np.linalg.norm(array, axis=-1, keepdims=True)
    lengths[lengths == 0] = 1.0  # an all-zero vector stays zero instead of dividing by zero
    return array / lengths
