import pytest
from pydantic import ValidationError

from bloom.config.settings import Settings


def test_defaults_when_nothing_is_set(monkeypatch):
    for name in ("GEMINI_API_KEY", "GEMINI_MODEL", "LLM_TEMPERATURE", "LLM_TIMEOUT_S"):
        monkeypatch.delenv(name, raising=False)
    settings = Settings(_env_file=None)
    assert settings.gemini_api_key.get_secret_value() == ""
    assert settings.llm_temperature == 0.2
    assert settings.llm_timeout_s == 60


def test_reads_environment_variables(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "abc123")
    monkeypatch.setenv("GEMINI_MODEL", "some-model")
    settings = Settings(_env_file=None)
    assert settings.gemini_api_key.get_secret_value() == "abc123"
    assert settings.gemini_model == "some-model"


def test_secret_is_hidden_when_printed(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "abc123")
    assert "abc123" not in repr(Settings(_env_file=None))


def test_temperature_out_of_range_is_rejected(monkeypatch):
    monkeypatch.setenv("LLM_TEMPERATURE", "9")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)
