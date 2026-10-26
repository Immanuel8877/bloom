from langchain_core.exceptions import (
    ModelAPIError,
    ModelAuthenticationError,
    ModelInvalidRequestError,
    ModelNotFoundError,
    ModelRateLimitError,
)
from pydantic import SecretStr

from bloom.errors import describe_error


def test_known_errors_get_specific_advice():
    assert "GEMINI_API_KEY" in describe_error(ModelAuthenticationError("x"))
    assert "--model" in describe_error(ModelNotFoundError("x"))
    assert "Rate limit" in describe_error(ModelRateLimitError("x"))


def test_invalid_request_shows_the_real_cause():
    message = describe_error(ModelInvalidRequestError("400 INVALID_ARGUMENT. API key not valid."))
    assert "API key not valid" in message
    assert "GEMINI_API_KEY" in message


def test_other_model_errors_show_cause_and_point_to_the_log():
    message = describe_error(ModelAPIError("503 overloaded"))
    assert "503 overloaded" in message
    assert "bloom.log" in message


def test_unknown_errors_name_their_type_and_message():
    message = describe_error(OSError("boom"))
    assert "OSError" in message
    assert "boom" in message


def test_api_key_is_blanked_out_of_error_text(monkeypatch):
    monkeypatch.setattr("bloom.errors.settings.gemini_api_key", SecretStr("super-secret-key"))
    message = describe_error(ModelAPIError("bad call with super-secret-key inside"))
    assert "super-secret-key" not in message
    assert "***" in message
