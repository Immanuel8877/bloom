"""The semantic_search tool: lets the agent find code by meaning."""

from __future__ import annotations

from pathlib import Path

from langchain_core.tools import tool

from bloom.errors import describe_error
from bloom.llm import build_embedder
from bloom.retrieval.index import INDEX_DIR, CodeIndex

MAX_RESULTS = 8
SNIPPET_LINES = 12


def load_fresh_index(embedder) -> CodeIndex:
    """The index for this folder, built or updated first if it is missing or out of date."""
    root = Path.cwd()
    try:
        index = CodeIndex.load(INDEX_DIR)
    except FileNotFoundError:
        index = CodeIndex.from_folder(root, embedder)
        index.save(INDEX_DIR)
        return index
    if index.refresh(root, embedder):
        index.save(INDEX_DIR)
    return index


@tool
def semantic_search(query: str, k: int = 5) -> str:
    """Find code by MEANING, not by exact words.

    Use this when you do not know where something lives, for example "where are errors
    shown to the user" or "how is the config loaded". Use search_text instead when you
    know the exact text. Returns the best matching pieces of code, best first. Use read_file
    afterwards to see more of a file.

    Args:
        query: A description of what you are looking for, in plain words.
        k: How many results to return (at most 8).
    """
    k = max(1, min(k, MAX_RESULTS))
    try:
        embedder = build_embedder()
        hits = load_fresh_index(embedder).search(embedder.embed_query(query), k)
    except ValueError as exc:  # our own clear messages, such as "Run: bloom index"
        return f"Error: {exc}"
    except Exception as exc:  # a tool must never crash the loop
        return f"Error: {describe_error(exc)}"
    if not hits:
        return "(no results)"
    parts = []
    for rank, (chunk, score) in enumerate(hits, start=1):
        snippet = "\n".join(f"    {line}" for line in chunk.text.splitlines()[:SNIPPET_LINES])
        header = f"{rank}. {chunk.path}:{chunk.start_line}-{chunk.end_line} (score {score:.2f})"
        parts.append(f"{header}\n{snippet}")
    return "\n\n".join(parts)
