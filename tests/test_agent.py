import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.errors import GraphRecursionError

from bloom.agent import final_answer, run_agent
from bloom.tools import TOOLS, list_files
from tests.fakes import ScriptedChatModel


def ask_for(name: str, args: dict, call_id: str = "call_1") -> AIMessage:
    """A model reply that asks for one tool call."""
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


def test_loop_runs_the_tool_then_answers(tmp_path, monkeypatch):
    (tmp_path / "hello.txt").write_text("hi")
    monkeypatch.chdir(tmp_path)
    model = ScriptedChatModel(
        responses=[ask_for("list_files", {"path": "."}), AIMessage(content="One file: hello.txt")]
    )
    messages = run_agent(model, [list_files], "What files are here?")

    kinds = [type(m) for m in messages]
    assert kinds == [SystemMessage, HumanMessage, AIMessage, ToolMessage, AIMessage]
    assert messages[3].content == "hello.txt"
    assert messages[3].tool_call_id == "call_1"
    assert final_answer(messages) == "One file: hello.txt"
    # the second model call was shown the tool result
    assert isinstance(model.seen[1][-1], ToolMessage)


def test_no_tool_call_means_one_model_call():
    model = ScriptedChatModel(responses=[AIMessage(content="Hello!")])
    messages = run_agent(model, [list_files], "hi")
    assert len(model.seen) == 1
    assert final_answer(messages) == "Hello!"


def test_unknown_tool_becomes_an_error_message_not_a_crash():
    model = ScriptedChatModel(
        responses=[ask_for("fly_to_moon", {}), AIMessage(content="Sorry, I cannot do that.")]
    )
    messages = run_agent(model, [list_files], "go")
    assert "no tool called 'fly_to_moon'" in messages[3].content


def test_a_crashing_tool_becomes_an_error_message_not_a_crash():
    @tool
    def explode() -> str:
        """Always fails."""
        raise RuntimeError("boom")

    model = ScriptedChatModel(
        responses=[ask_for("explode", {}), AIMessage(content="That tool failed.")]
    )
    messages = run_agent(model, [explode], "go", approve=lambda name, args: True)
    assert "RuntimeError: boom" in messages[3].content


def test_a_runaway_loop_is_stopped():
    # the model asks for a tool every single time and never answers
    model = ScriptedChatModel(responses=[ask_for("list_files", {}, f"c{i}") for i in range(40)])
    with pytest.raises(GraphRecursionError):
        run_agent(model, [list_files], "loop forever")


def test_several_rounds_of_tools_before_the_answer(tmp_path, monkeypatch):
    (tmp_path / "app.py").write_text("def main():\n    return 1\n")
    monkeypatch.chdir(tmp_path)
    model = ScriptedChatModel(
        responses=[
            ask_for("search_text", {"pattern": "def main"}, "c1"),
            ask_for("read_file", {"path": "app.py"}, "c2"),
            AIMessage(content="main returns 1."),
        ]
    )
    messages = run_agent(model, TOOLS, "What does main return?")
    tool_results = [m.content for m in messages if isinstance(m, ToolMessage)]
    assert tool_results == ["app.py:1: def main():", "1: def main():\n2:     return 1"]
    assert len(model.seen) == 3
    assert final_answer(messages) == "main returns 1."


def test_approved_write_happens(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    asked = []

    def yes(name, args):
        asked.append(name)
        return True

    model = ScriptedChatModel(
        responses=[
            ask_for("write_file", {"path": "a.txt", "content": "hi"}),
            AIMessage(content="Done."),
        ]
    )
    run_agent(model, TOOLS, "make a.txt", approve=yes)
    assert asked == ["write_file"]
    assert (tmp_path / "a.txt").read_text() == "hi"


def test_declined_write_does_not_happen(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    model = ScriptedChatModel(
        responses=[
            ask_for("write_file", {"path": "a.txt", "content": "hi"}),
            AIMessage(content="Okay, I will not."),
        ]
    )
    messages = run_agent(model, TOOLS, "make a.txt", approve=lambda name, args: False)
    assert not (tmp_path / "a.txt").exists()
    assert messages[3].content.startswith("Declined")


def test_no_approver_means_changes_are_declined(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    model = ScriptedChatModel(
        responses=[ask_for("run_command", {"command": "touch x"}), AIMessage(content="ok")]
    )
    messages = run_agent(model, TOOLS, "go")
    assert not (tmp_path / "x").exists()
    assert messages[3].content.startswith("Declined")


def test_denied_actions_never_ask_and_never_run(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("GEMINI_API_KEY=secret")
    monkeypatch.chdir(tmp_path)
    asked = []
    model = ScriptedChatModel(
        responses=[ask_for("read_file", {"path": ".env"}), AIMessage(content="I cannot.")]
    )
    messages = run_agent(model, TOOLS, "show .env", approve=lambda n, a: asked.append(n) or True)
    assert asked == []
    assert messages[3].content.startswith("Denied")
    assert "secret" not in messages[3].content


def test_old_tool_output_is_shrunk_in_what_the_model_sees(tmp_path, monkeypatch):
    from bloom import context

    (tmp_path / "big.txt").write_text("word " * 400)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(context, "CONTEXT_BUDGET_TOKENS", 50)
    monkeypatch.setattr(context, "KEEP_RECENT_TOOL_RESULTS", 1)
    model = ScriptedChatModel(
        responses=[
            ask_for("read_file", {"path": "big.txt"}, "c1"),
            ask_for("read_file", {"path": "big.txt"}, "c2"),
            AIMessage(content="done"),
        ]
    )
    messages = run_agent(model, TOOLS, "read it twice")
    # the saved history is complete...
    full = [m.content for m in messages if isinstance(m, ToolMessage)]
    assert all("word word" in text for text in full)
    # ...but on the third call the model saw the first result shrunk and the second intact
    seen_results = [m.content for m in model.seen[2] if isinstance(m, ToolMessage)]
    assert "removed to save space" in seen_results[0]
    assert "word word" in seen_results[1]
