"""Context engineering: keeping the conversation small enough for the model to read.

The model re-reads the WHOLE conversation on every call. Tool results (file contents,
search hits, command output) are by far the biggest part, and old ones are rarely needed again.
"""

from __future__ import annotations

from langchain_core.messages import BaseMessage, ToolMessage

CHARS_PER_TOKEN = 4  # a rough rule of thumb for English text and code
CONTEXT_BUDGET_TOKENS = 30_000  # start shrinking above this (deliberately small, so you can see it)
KEEP_RECENT_TOOL_RESULTS = 4  # the newest tool results are always kept in full


def estimate_tokens(messages: list[BaseMessage]) -> int:
    """A cheap estimate of how many tokens these messages take. No network needed."""
    chars = 0
    for message in messages:
        chars += len(message.text)
        for call in getattr(message, "tool_calls", None) or []:
            chars += len(str(call["args"]))
    return chars // CHARS_PER_TOKEN


def shrink_old_tool_results(
    messages: list[BaseMessage], keep: int | None = None
) -> list[BaseMessage]:
    """Replace the text of all but the newest `keep` tool results with a short note.

    `keep` defaults to KEEP_RECENT_TOOL_RESULTS.

    The ToolMessage itself stays, with its tool_call_id, because every tool call the model
    made must still have an answer. Only the bulky text goes. The input list is not changed.
    """
    keep = KEEP_RECENT_TOOL_RESULTS if keep is None else keep
    tool_positions = [i for i, m in enumerate(messages) if isinstance(m, ToolMessage)]
    old = set(tool_positions[: max(len(tool_positions) - keep, 0)])
    shrunk = []
    for i, message in enumerate(messages):
        if i in old:
            note = f"[old tool output removed to save space: {len(message.text)} characters]"
            if len(message.text) > len(note):
                message = message.model_copy(update={"content": note})
        shrunk.append(message)
    return shrunk


def prepare_for_model(messages: list[BaseMessage]) -> list[BaseMessage]:
    """The view of the conversation that is actually sent to the model."""
    if estimate_tokens(messages) <= CONTEXT_BUDGET_TOKENS:
        return messages
    return shrink_old_tool_results(messages)
