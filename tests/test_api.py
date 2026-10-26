import sys

import pytest
from fastapi.testclient import TestClient

from bloom.api import create_app, run_in_subprocess

KEY = "test-key-0123456789"
HEADERS = {"X-API-Key": KEY}


async def ok_runner(task: str) -> dict:
    return {"status": "ok", "answer": f"did: {task}", "tool_calls": [{"name": "list_files"}]}


async def failing_runner(task: str) -> dict:
    return {"status": "error", "error": "model unavailable"}


async def exploding_runner(task: str) -> dict:
    raise RuntimeError("kaboom")


def fake_searcher(query: str, k: int) -> list[dict]:
    hit = {"path": "a.py", "start_line": 1, "end_line": 3, "score": 0.9, "snippet": "x = 1"}
    return [hit][:k]


def client(runner=ok_runner, searcher=fake_searcher) -> TestClient:
    return TestClient(create_app(KEY, runner=runner, searcher=searcher))


# ---- the door ------------------------------------------------------------------------------


def test_an_empty_key_is_refused_at_startup():
    with pytest.raises(ValueError):
        create_app("")


def test_health_needs_no_key():
    response = client().get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


@pytest.mark.parametrize(
    ("method", "path"), [("post", "/runs"), ("get", "/runs/abc"), ("post", "/search")]
)
def test_every_other_endpoint_needs_the_key(method, path):
    c = client()
    assert getattr(c, method)(path).status_code == 401
    wrong = getattr(c, method)(path, headers={"X-API-Key": "nope"})
    assert wrong.status_code == 401


# ---- runs ----------------------------------------------------------------------------------


def test_a_run_is_accepted_then_finishes():
    c = client()
    started = c.post("/runs", json={"task": "list the files"}, headers=HEADERS)
    assert started.status_code == 202
    assert started.json()["status"] == "queued"
    run_id = started.json()["run_id"]

    finished = c.get(f"/runs/{run_id}", headers=HEADERS).json()
    assert finished["status"] == "done"
    assert finished["answer"] == "did: list the files"
    assert finished["tool_calls"] == [{"name": "list_files"}]
    assert finished["error"] is None


def test_a_failed_run_says_why():
    c = client(runner=failing_runner)
    run_id = c.post("/runs", json={"task": "x"}, headers=HEADERS).json()["run_id"]
    run = c.get(f"/runs/{run_id}", headers=HEADERS).json()
    assert run["status"] == "failed"
    assert run["error"] == "model unavailable"


def test_a_crashing_runner_ends_as_failed_not_stuck():
    c = client(runner=exploding_runner)
    run_id = c.post("/runs", json={"task": "x"}, headers=HEADERS).json()["run_id"]
    run = c.get(f"/runs/{run_id}", headers=HEADERS).json()
    assert run["status"] == "failed"
    assert "kaboom" in run["error"]


def test_unknown_run_is_404():
    assert client().get("/runs/missing", headers=HEADERS).status_code == 404


@pytest.mark.parametrize("body", [{}, {"task": ""}, {"task": "x" * 4001}, {"task": 5}])
def test_bad_run_requests_are_rejected(body):
    assert client().post("/runs", json=body, headers=HEADERS).status_code == 422


# ---- search --------------------------------------------------------------------------------


def test_search_returns_hits():
    response = client().post("/search", json={"query": "where is x", "k": 3}, headers=HEADERS)
    assert response.status_code == 200
    assert response.json()[0]["path"] == "a.py"


@pytest.mark.parametrize("body", [{"query": ""}, {"query": "x", "k": 0}, {"query": "x", "k": 99}])
def test_bad_search_requests_are_rejected(body):
    assert client().post("/search", json=body, headers=HEADERS).status_code == 422


def test_search_problems_become_clean_errors():
    def needs_reindex(query, k):
        raise ValueError("Run: bloom index")

    def google_down(query, k):
        raise RuntimeError("boom")

    conflict = client(searcher=needs_reindex).post("/search", json={"query": "x"}, headers=HEADERS)
    assert conflict.status_code == 409
    assert "bloom index" in conflict.json()["detail"]
    upstream = client(searcher=google_down).post("/search", json={"query": "x"}, headers=HEADERS)
    assert upstream.status_code == 502


# ---- the real subprocess runner, with a harmless stand-in command -------------------------

WRITE_SUMMARY = (
    "import json, sys; "
    "json.dump({'status': 'ok', 'answer': sys.argv[2], 'tool_calls': []}, open(sys.argv[1], 'w'))"
)


def command_writing_summary(task, summary_path):
    return [sys.executable, "-c", WRITE_SUMMARY, str(summary_path), task]


async def test_subprocess_runner_reads_the_summary_file():
    result = await run_in_subprocess("hello", command_for=command_writing_summary)
    assert result == {"status": "ok", "answer": "hello", "tool_calls": []}


async def test_subprocess_runner_stops_a_run_that_takes_too_long():
    def sleeper(task, summary_path):
        return [sys.executable, "-c", "import time; time.sleep(30)"]

    result = await run_in_subprocess("x", timeout_s=0.5, command_for=sleeper)
    assert result["status"] == "error"
    assert "stopped after 0.5 seconds" in result["error"]


async def test_subprocess_runner_reports_a_crash_with_its_message():
    def crasher(task, summary_path):
        return [sys.executable, "-c", "import sys; sys.stderr.write('bad thing'); sys.exit(3)"]

    result = await run_in_subprocess("x", command_for=crasher)
    assert result["status"] == "error"
    assert "exit 3" in result["error"]
    assert "bad thing" in result["error"]
