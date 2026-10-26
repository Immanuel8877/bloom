"""A chat model that replays a script of replies, so agent tests never touch the network."""

import re
import zlib

import numpy as np
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from pydantic import Field

from bloom.retrieval.embedder import normalize


class ScriptedChatModel(FakeMessagesListChatModel):
    """Returns the prepared replies one by one. Records every conversation it is shown."""

    seen: list = Field(default_factory=list)

    def bind_tools(self, tools, **kwargs):
        return self  # the real model needs the tool list; the fake does not

    def _generate(self, messages, *args, **kwargs):
        self.seen.append(list(messages))
        return super()._generate(messages, *args, **kwargs)


class HashingEmbedder:
    """A pretend embedder for tests: it counts words, so shared words mean similar vectors.

    It is NOT semantic. 'cat' and 'kitten' are unrelated to it. But it is deterministic and
    needs no network, so the whole search pipeline can be tested.
    """

    model = "fake-hash"
    dimensions = 64

    def _vector(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dimensions, dtype=np.float32)
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            vector[zlib.crc32(word.encode()) % self.dimensions] += 1.0
        return normalize(vector)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return np.array([self._vector(text) for text in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._vector(text)
