"""Turn model-call failures into one short message the user can act on."""

from __future__ import annotations

from langchain_core.exceptions import (
    ContextOverflowError,
    ModelAuthenticationError,
    ModelConnectionError,
    ModelError,
    ModelInvalidRequestError,
    ModelNotFoundError,
    ModelPermissionDeniedError,
    ModelRateLimitError,
    ModelTimeoutError,
)

from bloom.config.settings import settings

LOG_HINT = "See .bloom/logs/bloom.log for details."

# Order matters: the first match wins. LangChain gives every provider the same error classes,
# so this table keeps working if you ever switch away from Gemini.
_MESSAGES: list[tuple[type[Exception], str]] = [
    (ModelAuthenticationError, "The API key was rejected. Check GEMINI_API_KEY in .env."),
    (ModelPermissionDeniedError, "The API key is not allowed to use this model or API."),
    (ModelNotFoundError, "The model was not found. Try --model with a current model name."),
    (ModelRateLimitError, "Rate limit reached. Wait a minute and try again."),
    (ContextOverflowError, "The conversation is too long for this model."),
    (ModelTimeoutError, "The model took too long to answer. Try again."),
    (ModelConnectionError, "Could not reach the model. Check your internet connection."),
]


def _detail(exc: Exception, limit: int = 240) -> str:
    """The first line of the error text, shortened, with the API key blanked out if present."""
    lines = str(exc).strip().splitlines()
    text = lines[0] if lines else ""
    secret = settings.gemini_api_key.get_secret_value()
    if secret:
        text = text.replace(secret, "***")
    return text if len(text) <= limit else text[: limit - 1] + "…"


def describe_error(exc: Exception) -> str:
    """Return a short, friendly message for `exc`, with the real cause when there is no advice."""
    for error_type, message in _MESSAGES:
        if isinstance(exc, error_type):
            return message
    detail = _detail(exc)
    if isinstance(exc, ModelInvalidRequestError):
        return (
            f"The model rejected the request: {detail} "
            "If this mentions the API key, check GEMINI_API_KEY in .env."
        )
    if isinstance(exc, ModelError):
        return f"The model service returned an error: {detail} {LOG_HINT}"
    return f"Could not get an answer ({type(exc).__name__}: {detail}). {LOG_HINT}"
