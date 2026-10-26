"""The `bloom` command-line entrypoint."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import click

from bloom import __version__
from bloom.agent import build_graph, final_answer, new_conversation, run_agent, run_turn
from bloom.config.settings import settings
from bloom.context import estimate_tokens
from bloom.errors import describe_error
from bloom.llm import ask_model, build_chat_model, build_embedder, format_usage
from bloom.log import init_logger
from bloom.retrieval.index import INDEX_DIR, CodeIndex
from bloom.runner import execute
from bloom.toolset import AGENT_TOOLS

logger = logging.getLogger(__name__)


@click.group()
@click.version_option(version=__version__, prog_name="bloom")
def cli() -> None:
    """bloom: a terminal coding agent."""
    init_logger(settings.log_level)


@cli.command()
@click.argument("question")
@click.option("--model", default=None, help="Model name to use instead of GEMINI_MODEL.")
@click.option("--usage", is_flag=True, help="Also print how many tokens the call used.")
def ask(question: str, model: str | None, usage: bool) -> None:
    """Send one QUESTION to the model and print the answer."""
    if not settings.gemini_api_key.get_secret_value():
        raise click.ClickException(
            "GEMINI_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    logger.info("ask: %d-character question", len(question))
    chat_model = build_chat_model(model)
    try:
        reply = ask_model(chat_model, question)
    except Exception as exc:  # the CLI boundary: log the traceback, show one short line
        logger.exception("model call failed")
        raise click.ClickException(describe_error(exc)) from exc
    click.echo(reply.text)
    if usage:
        click.echo(format_usage(reply), err=True)


def ask_user(tool_name: str, args: dict) -> bool:
    """Show the human what the agent wants to do and wait for a yes or no."""
    click.echo(f"\nbloom wants to run {tool_name}:", err=True)
    for key, value in args.items():
        shown = str(value) if len(str(value)) <= 400 else str(value)[:400] + "..."
        click.echo(f"  {key}: {shown}", err=True)
    return click.confirm("Allow?", default=False, err=True)


@cli.command()
@click.argument("task")
@click.option("--model", default=None, help="Model name to use instead of GEMINI_MODEL.")
def agent(task: str, model: str | None) -> None:
    """Give the agent a TASK. It may use tools before it answers."""
    if not settings.gemini_api_key.get_secret_value():
        raise click.ClickException(
            "GEMINI_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    logger.info("agent: %d-character task", len(task))
    chat_model = build_chat_model(model)
    try:
        messages = run_agent(chat_model, AGENT_TOOLS, task, approve=ask_user)
    except Exception as exc:  # the CLI boundary: log the traceback, show one short line
        logger.exception("agent run failed")
        raise click.ClickException(describe_error(exc)) from exc
    for message in messages:
        for call in getattr(message, "tool_calls", None) or []:
            click.echo(f"-> {call['name']}({call['args']})", err=True)
    click.echo(final_answer(messages))


@cli.command()
@click.option("--model", default=None, help="Model name to use instead of GEMINI_MODEL.")
def chat(model: str | None) -> None:
    """Talk to the agent. It remembers the conversation until you leave."""
    if not settings.gemini_api_key.get_secret_value():
        raise click.ClickException(
            "GEMINI_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    graph = build_graph(build_chat_model(model), AGENT_TOOLS, ask_user)
    history = new_conversation()
    click.echo("bloom chat. Type /exit to leave or /clear to start over.", err=True)
    while True:
        try:
            text = click.prompt("you", prompt_suffix="> ", default="", show_default=False).strip()
        except click.Abort:  # Ctrl+C or Ctrl+D
            break
        if not text:
            continue
        if text == "/exit":
            break
        if text == "/clear":
            history = new_conversation()
            click.echo("(conversation cleared)", err=True)
            continue
        before = len(history)
        try:
            history = run_turn(graph, history, text)
        except Exception as exc:  # keep the session alive; this turn is simply dropped
            logger.exception("chat turn failed")
            click.echo(f"Error: {describe_error(exc)}", err=True)
            continue
        for message in history[before:]:
            for call in getattr(message, "tool_calls", None) or []:
                click.echo(f"-> {call['name']}({call['args']})", err=True)
        click.echo(final_answer(history))
        click.echo(f"(about {estimate_tokens(history)} tokens of context)", err=True)


def require_api_key() -> None:
    """Stop with a clear message if there is no API key."""
    if not settings.gemini_api_key.get_secret_value():
        raise click.ClickException(
            "GEMINI_API_KEY is not set. Copy .env.example to .env and add your key."
        )


@cli.command("index")
def index_cmd() -> None:
    """Build the code search index for the current folder."""
    require_api_key()
    try:
        code_index = CodeIndex.from_folder(Path.cwd(), build_embedder())
    except ValueError as exc:  # for example "there is nothing to index"
        raise click.ClickException(str(exc)) from exc
    except Exception as exc:  # the CLI boundary: log the traceback, show one short line
        logger.exception("indexing failed")
        raise click.ClickException(describe_error(exc)) from exc
    code_index.save(INDEX_DIR)
    files = len({chunk.path for chunk in code_index.chunks})
    click.echo(f"Indexed {len(code_index.chunks)} chunks from {files} files into {INDEX_DIR}.")


@cli.command("search")
@click.argument("query")
@click.option("-k", default=5, show_default=True, help="How many results to show.")
def search_cmd(query: str, k: int) -> None:
    """Find the code that best matches the meaning of QUERY."""
    require_api_key()
    try:
        code_index = CodeIndex.load(INDEX_DIR)
    except FileNotFoundError:
        raise click.ClickException("There is no index yet. Run: bloom index") from None
    embedder = build_embedder()
    if code_index.model != embedder.model:
        raise click.ClickException(
            f"The index was built with '{code_index.model}' but EMBEDDING_MODEL is "
            f"'{embedder.model}'. Run: bloom index"
        )
    try:
        hits = code_index.search(embedder.embed_query(query), k)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    except Exception as exc:
        logger.exception("search failed")
        raise click.ClickException(describe_error(exc)) from exc
    for chunk, score in hits:
        first_line = next((line.strip() for line in chunk.text.splitlines() if line.strip()), "")
        click.echo(f"{score:.3f}  {chunk.path}:{chunk.start_line}-{chunk.end_line}")
        click.echo(f"       {first_line[:100]}")


@cli.command("run")
@click.argument("task")
@click.option("--model", default=None, help="Model name to use instead of GEMINI_MODEL.")
@click.option(
    "--allow-changes",
    is_flag=True,
    help="Approve every file change and command without asking. Denied actions stay denied.",
)
@click.option(
    "--summary-json",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Also write the result (answer, tool calls, status) to this JSON file.",
)
def run_cmd(task: str, model: str | None, allow_changes: bool, summary_json: Path | None) -> None:
    """Run TASK once with no questions asked, then print the answer. Made for scripts."""
    require_api_key()
    logger.info("run: %d-character task", len(task))
    summary = execute(task, model, allow_changes)
    if summary_json is not None:
        summary_json.write_text(json.dumps(summary.to_dict(), indent=2), encoding="utf-8")
    if summary.status != "ok":
        raise click.ClickException(summary.error or "The run failed.")
    click.echo(summary.answer)


@cli.command("serve")
@click.option("--host", default="127.0.0.1", show_default=True, help="Address to listen on.")
@click.option("--port", default=8000, show_default=True, help="Port to listen on.")
def serve_cmd(host: str, port: int) -> None:
    """Start the HTTP service for the current folder. Needs BLOOM_API_KEY."""
    require_api_key()
    key = settings.bloom_api_key.get_secret_value()
    if len(key) < 16:
        raise click.ClickException(
            "BLOOM_API_KEY must be set to a random string of at least 16 characters. "
            'Make one with: python3 -c "import secrets; print(secrets.token_urlsafe(32))"'
        )
    if host not in {"127.0.0.1", "localhost", "::1"}:
        click.echo(
            "Warning: this is not a local address. Anyone who can reach it and has the key "
            "can read your project through bloom.",
            err=True,
        )
    import uvicorn

    from bloom.api import create_app

    click.echo(f"bloom service for {Path.cwd()} on http://{host}:{port}  (docs: /docs)", err=True)
    uvicorn.run(create_app(key), host=host, port=port)
