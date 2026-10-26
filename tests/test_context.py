from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from bloom import context
from bloom.context import estimate_tokens, prepare_for_model, shrink_old_tool_results


def conversation(n_tools: int, size: int = 400):
    """A user question followed by n_tools rounds of (tool call, big tool result)."""
    messages = [HumanMessage(content="go")]
    for i in range(n_tools):
        call = {"name": "read_file", "args": {"path": f"f{i}"}, "id": f"c{i}"}
        messages.append(AIMessage(content="", tool_calls=[call]))
        messages.append(ToolMessage(content="x" * size, tool_call_id=f"c{i}"))
    return messages


def test_estimate_is_about_one_token_per_four_characters():
    assert estimate_tokens([HumanMessage(content="a" * 400)]) == 100


def test_estimate_counts_tool_call_arguments():
    call = {"name": "read_file", "args": {"path": "abcd" * 10}, "id": "c1"}
    assert estimate_tokens([AIMessage(content="", tool_calls=[call])]) > 0


def test_old_tool_results_shrink_and_new_ones_stay():
    messages = conversation(6)
    shrunk = shrink_old_tool_results(messages, keep=2)
    results = [m for m in shrunk if isinstance(m, ToolMessage)]
    assert all("removed to save space: 400 characters" in m.content for m in results[:4])
    assert [m.content for m in results[4:]] == ["x" * 400, "x" * 400]


def test_shrinking_keeps_every_tool_call_answered():
    shrunk = shrink_old_tool_results(conversation(6), keep=1)
    assert [m.tool_call_id for m in shrunk if isinstance(m, ToolMessage)] == [
        f"c{i}" for i in range(6)
    ]
    assert len(shrunk) == 13


def test_shrinking_does_not_change_the_original_list():
    messages = conversation(3)
    shrink_old_tool_results(messages, keep=0)
    assert all(m.content == "x" * 400 for m in messages if isinstance(m, ToolMessage))


def test_short_results_are_left_alone():
    messages = conversation(3, size=5)
    assert shrink_old_tool_results(messages, keep=0) == messages


def test_prepare_does_nothing_under_budget(monkeypatch):
    monkeypatch.setattr(context, "CONTEXT_BUDGET_TOKENS", 10_000)
    messages = conversation(6)
    assert prepare_for_model(messages) is messages


def test_prepare_shrinks_over_budget(monkeypatch):
    monkeypatch.setattr(context, "CONTEXT_BUDGET_TOKENS", 100)
    prepared = prepare_for_model(conversation(8))
    results = [m for m in prepared if isinstance(m, ToolMessage)]
    assert "removed" in results[0].content
    assert results[-1].content == "x" * 400
