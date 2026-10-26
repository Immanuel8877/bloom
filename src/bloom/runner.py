"""Run the agent once with nobody at the keyboard, and report what happened as plain data."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage

from bloom.agent import final_answer, run_agent
from bloom.errors import describe_error
from bloom.llm import build_chat_model
from bloom.toolset import AGENT_TOOLS


@dataclass
class RunSummary:
    """The outcome of one headless run. Plain data, so it can be saved as JSON."""

    status: str  # "ok" or "error"
    answer: str = ""
    tool_calls: list[dict] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def summarize(messages: list[BaseMessage]) -> RunSummary:
    """Turn a finished conversation into a RunSummary."""
    calls = [
        {"name": call["name"], "args": call["args"]}
        for message in messages
        if isinstance(message, AIMessage)
        for call in message.tool_calls
    ]
    return RunSummary(status="ok", answer=final_answer(messages), tool_calls=calls)


def execute(
    task: str,
    model: str | None = None,
    allow_changes: bool = False,
    chat_model: BaseChatModel | None = None,
) -> RunSummary:
    """Run `task` to completion. Never raises: a failure comes back as status "error".

    With no human to ask, actions that need approval are declined, unless `allow_changes`
    says to approve them all. Denied actions (like reading .env) stay denied either way.
    """
    approve = (lambda name, args: True) if allow_changes else None
    try:
        chat_model = chat_model or build_chat_model(model)
        messages = run_agent(chat_model, AGENT_TOOLS, task, approve=approve)
    except Exception as exc:
        return RunSummary(status="error", error=describe_error(exc))
    return summarize(messages)
