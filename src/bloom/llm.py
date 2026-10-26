from __future__ import annotations

import warnings

import numpy as np
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from bloom.config.settings import settings
from bloom.prompts import SYSTEM_PROMPT
from bloom.retrieval.embedder import normalize

warnings.filterwarnings("ignore", message=".*uses fixed sampling defaults.*", category=UserWarning)


def build_chat_model(model: str | None = None) -> BaseChatModel:
    return ChatGoogleGenerativeAI(
        model=model or settings.gemini_model,
        api_key=settings.gemini_api_key.get_secret_value(),
        temperature=settings.llm_temperature,
        timeout=settings.llm_timeout_s,
        max_retries=2,
    )


def ask_model(chat_model: BaseChatModel, question: str) -> AIMessage:
    """Send one question (plus the system prompt) and return the model's reply message."""
    messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=question)]
    return chat_model.invoke(messages)


def format_usage(reply: AIMessage) -> str:
    """One line describing the tokens a reply used, or a note if the provider sent none."""
    usage = reply.usage_metadata
    if not usage:
        return "tokens: not reported"
    return (
        f"tokens: {usage['input_tokens']} in, {usage['output_tokens']} out, "
        f"{usage['total_tokens']} total"
    )


class GeminiEmbedder:
    """Turns text into vectors with Gemini. This class is the only place that knows that."""

    def __init__(self, model: str, dimensions: int, api_key: str, client=None) -> None:
        self.model = model
        self.dimensions = dimensions
        self._client = client or GoogleGenerativeAIEmbeddings(
            model=model, google_api_key=api_key, output_dimensionality=dimensions
        )

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Vectors for code chunks (Gemini's RETRIEVAL_DOCUMENT task)."""
        vectors = self._client.embed_documents(
            texts, task_type="RETRIEVAL_DOCUMENT", output_dimensionality=self.dimensions
        )
        return normalize(np.array(vectors))

    def embed_query(self, text: str) -> np.ndarray:
        """One vector for a question about code (Gemini's CODE_RETRIEVAL_QUERY task)."""
        vector = self._client.embed_query(
            text, task_type="CODE_RETRIEVAL_QUERY", output_dimensionality=self.dimensions
        )
        return normalize(np.array(vector))


def build_embedder() -> GeminiEmbedder:
    """Create the embedder from settings."""
    return GeminiEmbedder(
        model=settings.embedding_model,
        dimensions=settings.embedding_dimensions,
        api_key=settings.gemini_api_key.get_secret_value(),
    )
