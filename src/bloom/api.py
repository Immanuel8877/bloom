"""A small HTTP service around bloom, so other programs can use it.

    GET  /health          is the service up?                        (no key needed)
    POST /runs            start an agent run, returns a run_id
    GET  /runs/{run_id}   status and result of that run
    POST /search          find code by meaning

Every endpoint except /health needs the header  X-API-Key: <your BLOOM_API_KEY>.
Runs are read-only: the service never approves file changes or commands.
"""

from __future__ import annotations

import asyncio
import json
import secrets
import sys
import tempfile
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from bloom import __version__
from bloom.errors import describe_error
from bloom.llm import build_embedder
from bloom.search_tool import SNIPPET_LINES, load_fresh_index

RUN_TIMEOUT_S = 300
MAX_CONCURRENT_RUNS = 2


class RunRequest(BaseModel):
    task: str = Field(min_length=1, max_length=4000)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    k: int = Field(5, ge=1, le=8)


class SearchHit(BaseModel):
    path: str
    start_line: int
    end_line: int
    score: float
    snippet: str


class Run(BaseModel):
    run_id: str
    status: str  # queued -> running -> done or failed
    task: str
    created_at: str
    answer: str | None = None
    tool_calls: list[dict] = Field(default_factory=list)
    error: str | None = None


# A runner takes the task text and returns a summary dict (see bloom.runner.RunSummary).
Runner = Callable[[str], Awaitable[dict]]
# A searcher takes (query, k) and returns a list of dicts shaped like SearchHit.
Searcher = Callable[[str, int], list[dict]]


def default_command(task: str, summary_path: Path) -> list[str]:
    """The command that runs one task in a separate process: `python -m bloom run ...`."""
    # "--" means "everything after this is the task", even if the task starts with a dash.
    return [sys.executable, "-m", "bloom", "run", "--summary-json", str(summary_path), "--", task]


async def run_in_subprocess(
    task: str,
    timeout_s: float = RUN_TIMEOUT_S,
    command_for: Callable[[str, Path], list[str]] = default_command,
) -> dict:
    """Run `task` as `bloom run` in its own process and return its summary.

    A separate process keeps the slow, blocking agent away from the web server's event loop,
    and lets us kill a run that takes too long.
    """
    with tempfile.TemporaryDirectory() as folder:
        summary_path = Path(folder) / "summary.json"
        process = await asyncio.create_subprocess_exec(
            *command_for(task, summary_path),
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            _, stderr = await asyncio.wait_for(process.communicate(), timeout_s)
        except TimeoutError:
            process.kill()
            await process.wait()
            return {"status": "error", "error": f"The run was stopped after {timeout_s:g} seconds."}
        if summary_path.is_file():
            try:
                return json.loads(summary_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                pass
        tail = stderr.decode(errors="replace").strip()[-500:]
        code = process.returncode
        return {"status": "error", "error": f"The run ended without a result (exit {code}). {tail}"}


def default_searcher(query: str, k: int) -> list[dict]:
    """Search the index of the folder the service was started in."""
    embedder = build_embedder()
    hits = load_fresh_index(embedder).search(embedder.embed_query(query), k)
    return [
        {
            "path": chunk.path,
            "start_line": chunk.start_line,
            "end_line": chunk.end_line,
            "score": round(score, 4),
            "snippet": "\n".join(chunk.text.splitlines()[:SNIPPET_LINES]),
        }
        for chunk, score in hits
    ]


def create_app(
    api_key: str, runner: Runner | None = None, searcher: Searcher | None = None
) -> FastAPI:
    """Build the service. `runner` and `searcher` can be swapped out, which tests do."""
    if not api_key:
        raise ValueError("an API key is required")
    app = FastAPI(title="bloom", version=__version__)
    runs: dict[str, Run] = {}  # in memory only: gone when the service stops
    gate = asyncio.Semaphore(MAX_CONCURRENT_RUNS)  # at most this many agent runs at once
    run_task: Runner = runner or run_in_subprocess
    search: Searcher = searcher or default_searcher

    def require_key(x_api_key: str | None = Header(default=None)) -> None:
        # compare_digest takes the same time however many characters match, so the
        # response time cannot be used to guess the key one character at a time.
        if x_api_key is None or not secrets.compare_digest(x_api_key.encode(), api_key.encode()):
            raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key header.")

    async def execute_run(run_id: str) -> None:
        run = runs[run_id]
        async with gate:
            run.status = "running"
            try:
                result = await run_task(run.task)
            except Exception as exc:  # whatever happens, the run must end up "failed", not stuck
                result = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
        run.answer = result.get("answer")
        run.tool_calls = result.get("tool_calls") or []
        run.error = result.get("error")
        run.status = "done" if result.get("status") == "ok" else "failed"

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "version": __version__}

    @app.post("/runs", status_code=202, response_model=Run, dependencies=[Depends(require_key)])
    def create_run(body: RunRequest, background: BackgroundTasks) -> Run:
        run = Run(
            run_id=uuid4().hex[:12],
            status="queued",
            task=body.task,
            created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        )
        runs[run.run_id] = run
        background.add_task(execute_run, run.run_id)
        return run

    @app.get("/runs/{run_id}", response_model=Run, dependencies=[Depends(require_key)])
    def get_run(run_id: str) -> Run:
        if run_id not in runs:
            raise HTTPException(status_code=404, detail="No such run.")
        return runs[run_id]

    @app.post("/search", response_model=list[SearchHit], dependencies=[Depends(require_key)])
    def search_code(body: SearchRequest) -> list[dict]:
        try:
            return search(body.query, body.k)
        except ValueError as exc:  # our own clear messages, such as "Run: bloom index"
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=502, detail=describe_error(exc)) from exc

    return app
