import pytest


@pytest.fixture(autouse=True)
def no_log_file(monkeypatch):
    """Stop CLI tests from writing a real .bloom/logs file into the project."""
    monkeypatch.setattr("bloom.cli.init_logger", lambda *args, **kwargs: None)
