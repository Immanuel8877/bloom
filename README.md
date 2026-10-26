# bloom

A terminal coding agent, built from scratch with **LangChain**, **LangGraph** and **Gemini**.

bloom reads your project, searches it by meaning, and, only with your approval, edits files and
runs commands. It works from the terminal, in scripts, and over HTTP.

I built it step by step to learn how coding agents actually work, so the code favors being
readable over being clever. Every piece is small, tested, and explained where it lives.

## What it does

- **Agent loop**: the model asks for a tool, bloom runs it, the result goes back, repeat until
  the model has an answer. Written by hand as a LangGraph graph, not a prebuilt shortcut.
- **Eight tools**: list, read, find and search files; **semantic code search**; write and edit
  files; run shell commands.
- **Permission gate**: every tool call is checked before it runs: allow, ask you, or deny.
- **Semantic search**: a FAISS index of your code, so "where are errors shown to the user?"
  finds the right file even when the words don't match. The index refreshes itself when files change.
- **Context management**: long conversations stay small. Old tool output is shrunk before each
  model call, while the saved history stays complete.
- **Headless mode and HTTP API**: `bloom run` for scripts, `bloom serve` for other programs.
- **Tested without the network**: the whole test suite runs offline with scripted fake models.

## How it works

```
                 ┌──────────────── tool calls? ────────────────┐
                 │ yes                                          │ no
 START ──► model node ──► tools node (permission gate) ──┐      ▼
              ▲                                           │     END
              └───────────────────────────────────────────┘
```

A model is a stateless function: you send it the whole conversation and it returns one new
message. That message either contains the final answer or a *request* to call a tool. The model
never runs anything itself. bloom does, then appends the result and calls the model again.

## Quick start

You need **Python 3.11+**, **[uv](https://docs.astral.sh/uv/)**, and a free Gemini API key from
[Google AI Studio](https://aistudio.google.com/apikey).

```bash
git clone https://github.com/<your-username>/bloom.git
cd bloom
uv sync
cp .env.example .env        # then open .env and paste your key after GEMINI_API_KEY=
uv run bloom --version
uv run bloom ask "Explain what a Python virtual environment is in two sentences."
```

Run bloom from the folder you want it to work on. Its file tools only see files inside that folder.

## Commands

| Command | What it does |
|---|---|
| `bloom ask "question"` | One question to the model, no tools. `--usage` shows token counts. |
| `bloom chat` | Interactive agent that remembers the conversation. `/clear` resets, `/exit` leaves. |
| `bloom agent "task"` | One task with tools. Asks before anything that changes files or runs commands. |
| `bloom run "task"` | Same, but never asks (for scripts). Changes are declined unless you pass `--allow-changes`. `--summary-json FILE` saves the result. |
| `bloom index` | Build the semantic search index for the current folder. |
| `bloom search "query"` | Find code by meaning from the terminal. `-k 3` sets how many results. |
| `bloom serve` | Start the HTTP service (needs `BLOOM_API_KEY`). |

Every command that talks to the model accepts `--model NAME` to override `GEMINI_MODEL` for one run.

```bash
uv run bloom chat
you> Where does bloom decide whether a tool call needs my approval?
you> And what happens when I say no?
```

## Safety

Every tool call goes through a gate before it runs.

| Decision | Applies to | What happens |
|---|---|---|
| **Allow** | `list_files`, `read_file`, `find_files`, `search_text`, `semantic_search` | Runs immediately. These only read. |
| **Ask** | `write_file`, `edit_file`, `run_command`, and any tool the gate doesn't know | Shows you exactly what it wants to do and waits for `y`. Enter alone means no. |
| **Deny** | `.env` and `.git` (any file tool); commands containing `sudo`, `rm -rf`, `curl`, `wget` and similar | Never runs, and you are not asked. |

Other protections: the file tools can't leave the working folder, `.env` is never indexed or
searched, tool errors come back to the model as text instead of crashing the loop, commands stop
after 30 seconds, and a step limit stops a runaway agent.

The deny-list for commands is a speed bump, not a sandbox. Read what you approve.

## Configuration

Set these in `.env` (see `.env.example`) or as environment variables.

| Variable | Default | Meaning |
|---|---|---|
| `GEMINI_API_KEY` | none | Your Google AI Studio key. Required. |
| `GEMINI_MODEL` | `gemini-3.8-flash` | Chat model. Check Google's model list; names change. |
| `LLM_TEMPERATURE` | `0.2` | Some models ignore this and pick their own. |
| `LLM_TIMEOUT_S` | `60` | Seconds to wait for the model. |
| `LOG_LEVEL` | `INFO` | Logs go to `.bloom/logs/bloom.log`, never the terminal. |
| `EMBEDDING_MODEL` | `gemini-embedding-001` | Model used for code search. |
| `EMBEDDING_DIMENSIONS` | `768` | Numbers per vector. Changing it means re-running `bloom index`. |
| `BLOOM_API_KEY` | none | Password for the HTTP service. At least 16 characters. |

## HTTP API

```bash
python3 -c "import secrets; print(secrets.token_urlsafe(32))"   # make a key, put it in .env
uv run bloom serve                                              # http://127.0.0.1:8000
```

Every endpoint except `/health` needs the header `X-API-Key: <your BLOOM_API_KEY>`.

```bash
# start a run (returns 202 and a run_id immediately)
curl -s -X POST localhost:8000/runs \
  -H "X-API-Key: $BLOOM_API_KEY" -H "content-type: application/json" \
  -d '{"task": "List the files in the project root."}'

# poll until status is "done" or "failed"
curl -s localhost:8000/runs/<run_id> -H "X-API-Key: $BLOOM_API_KEY"

# search by meaning
curl -s -X POST localhost:8000/search \
  -H "X-API-Key: $BLOOM_API_KEY" -H "content-type: application/json" \
  -d '{"query": "where are errors shown to the user", "k": 3}'
```

Interactive docs are at `/docs`. Runs are **read-only**: the service never approves file changes
or commands. Each run is a separate `bloom run` process with a 5-minute limit, and at most two
run at once. The service listens on `127.0.0.1` by default.

## Project layout

```
src/bloom/
├── cli.py           the commands
├── agent.py         the LangGraph loop and the permission check
├── tools.py         list / read / find / search / write / edit / run
├── search_tool.py   the semantic_search tool
├── toolset.py       which tools the agent gets
├── permissions.py   allow / ask / deny rules
├── context.py       token estimate and shrinking of old tool output
├── llm.py           the only file that knows the provider is Gemini
├── errors.py        model errors -> one short, useful message
├── runner.py        headless runs (bloom run)
├── api.py           the FastAPI service
├── prompts.py       the system prompt
├── log.py           file logging
├── config/settings.py   typed settings from .env
└── retrieval/
    ├── chunker.py   cut files into overlapping 40-line chunks
    ├── embedder.py  the embedder interface and vector normalizing
    └── index.py     FAISS index: build, search, refresh, save, load
tests/               unit tests, all offline
scratch/             small experiments used while learning
```

## Development

```bash
uv run pytest            # all tests; none of them call the network
uv run ruff format .
uv run ruff check .
```

Tests use a scripted fake chat model (`tests/fakes.py`) and a word-counting fake embedder, so
they are fast and need no API key.

## Design notes

- **Provider isolation.** Only `llm.py` imports Gemini. Swapping providers touches one file.
- **Errors are text.** A failing tool returns `Error: ...` to the model, which can read it and
  try again, instead of raising and ending the run.
- **The view is not the history.** Shrinking old tool output changes only what is sent to the
  model. The saved conversation is never edited, and every tool call keeps its answer.
- **Incremental indexing.** The index remembers each file's modified time. A refresh embeds only
  new or changed files and copies the stored vectors for the rest.
- **Nothing is half-updated.** The index swaps in its new state only after every embedding
  call has succeeded.

## Limitations

- Gemini's free tier has rate limits and quotas, including for embeddings. Large first-time
  indexes can hit them.
- Runs in the HTTP service are kept in memory and lost when it stops. They can't be cancelled.
- The FAISS index is flat (exact search). That suits small and medium projects, not huge ones.
- Token counts are an estimate (about 4 characters per token), not exact.
- Only text files of known types are indexed.

## Acknowledgements

Inspired by the course *Building a Coding Agent From Scratch* (the `decode` project). bloom is
an independent implementation on a different stack (LangChain and LangGraph instead of Pydantic
AI) and shares no code with it.
