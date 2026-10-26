from langchain_core.messages import AIMessage

from bloom.runner import RunSummary, execute, summarize
from tests.fakes import ScriptedChatModel


def ask_for(name: str, args: dict, call_id: str = "c1") -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


def test_a_finished_run_reports_answer_and_tool_calls(tmp_path, monkeypatch):
    (tmp_path / "a.txt").write_text("x")
    monkeypatch.chdir(tmp_path)
    model = ScriptedChatModel(
        responses=[ask_for("list_files", {"path": "."}), AIMessage(content="One file.")]
    )
    summary = execute("what is here?", chat_model=model)
    assert summary == RunSummary(
        status="ok",
        answer="One file.",
        tool_calls=[{"name": "list_files", "args": {"path": "."}}],
    )
    assert summary.to_dict()["error"] is None


def test_changes_are_declined_by_default(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    model = ScriptedChatModel(
        responses=[
            ask_for("write_file", {"path": "a.txt", "content": "hi"}),
            AIMessage(content="I could not write it."),
        ]
    )
    summary = execute("make a.txt", chat_model=model)
    assert summary.status == "ok"
    assert not (tmp_path / "a.txt").exists()


def test_allow_changes_approves_them(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    model = ScriptedChatModel(
        responses=[
            ask_for("write_file", {"path": "a.txt", "content": "hi"}),
            AIMessage(content="Done."),
        ]
    )
    execute("make a.txt", allow_changes=True, chat_model=model)
    assert (tmp_path / "a.txt").read_text() == "hi"


def test_allow_changes_does_not_unlock_denied_actions(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("GEMINI_API_KEY=secret")
    monkeypatch.chdir(tmp_path)
    model = ScriptedChatModel(
        responses=[ask_for("read_file", {"path": ".env"}), AIMessage(content="Refused.")]
    )
    summary = execute("show .env", allow_changes=True, chat_model=model)
    assert "secret" not in summary.answer


def test_a_failure_becomes_an_error_summary_not_an_exception():
    class Broken:
        def bind_tools(self, tools):
            raise RuntimeError("boom")

    summary = execute("anything", chat_model=Broken())
    assert summary.status == "error"
    assert "boom" in summary.error


def test_summarize_collects_tool_calls_from_every_step():
    messages = [
        ask_for("search_text", {"pattern": "x"}, "c1"),
        ask_for("read_file", {"path": "a"}, "c2"),
        AIMessage(content="done"),
    ]
    summary = summarize(messages)
    assert [call["name"] for call in summary.tool_calls] == ["search_text", "read_file"]
    assert summary.answer == "done"
