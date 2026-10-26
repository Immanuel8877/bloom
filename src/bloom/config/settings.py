from __future__ import annotations

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # SecretStr hides the value when the object is printed or logged.
    gemini_api_key: SecretStr = SecretStr("")
    gemini_model: str = "gemini-3.8-flash"
    # 0 = most predictable answers, higher = more varied. Code work wants low.
    llm_temperature: float = Field(0.2, ge=0.0, le=2.0)
    llm_timeout_s: int = Field(60, gt=0)
    log_level: str = "INFO"
    # Code search: which embedding model, and how long each vector is.
    embedding_model: str = "gemini-embedding-001"
    embedding_dimensions: int = Field(768, gt=0)
    # The password callers must send to the HTTP service (`bloom serve`).
    bloom_api_key: SecretStr = SecretStr("")


settings = Settings()
