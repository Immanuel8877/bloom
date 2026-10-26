from click.testing import CliRunner
from langchain_core.exceptions import ModelRateLimitError
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import SecretStr

from bloom import cli as cli_module
from bloom.cli import cli


def with_key(monkeypatch):
    monkeypatch.setattr(cli_module.settings, "gemini_api_key", SecretStr("fake"))


def test_version():
    result = CliRunner().invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert "bloom" in result.output


def test_ask_without_key_fails_with_a_clear_message(monkeypatch):
    monkeypatch.setattr(cli_module.settings, "gemini_api_key", SecretStr(""))
    result = CliRunner().invoke(cli, ["ask", "hi"])
    assert result.exit_code != 0
    assert "GEMINI_API_KEY" in result.output


def test_ask_prints_the_answer(monkeypatch):
    with_key(monkeypatch)
    monkeypatch.setattr(cli_module, "build_chat_model", lambda model=None: object())
    monkeypatch.setattr(cli_module, "ask_model", lambda m, q: AIMessage(content="pong"))
    result = CliRunner().invoke(cli, ["ask", "ping"])
    assert result.exit_code == 0
    assert result.output.strip() == "pong"


def test_usage_flag_prints_token_counts(monkeypatch):
    with_key(monkeypatch)
    reply = AIMessage(
        content="pong",
        usage_metadata={"input_tokens": 3, "output_tokens": 1, "total_tokens": 4},
    )
    monkeypatch.setattr(cli_module, "build_chat_model", lambda model=None: object())
    monkeypatch.setattr(cli_module, "ask_model", lambda m, q: reply)
    result = CliRunner().invoke(cli, ["ask", "ping", "--usage"])
    assert result.exit_code == 0
    assert "pong" in result.output
    assert "tokens: 3 in, 1 out, 4 total" in result.output


def test_model_errors_become_short_messages(monkeypatch):
    with_key(monkeypatch)

    def boom(model, question):
        raise ModelRateLimitError("quota")

    monkeypatch.setattr(cli_module, "build_chat_model", lambda model=None: object())
    monkeypatch.setattr(cli_module, "ask_model", boom)
    result = CliRunner().invoke(cli, ["ask", "hi"])
    assert result.exit_code != 0
    assert "Rate limit" in result.output
    assert "Traceback" not in result.output


def test_agent_command_prints_tool_calls_and_the_answer(monkeypatch):
    with_key(monkeypatch)
    conversation = [
        HumanMessage(content="task"),
        AIMessage(content="", tool_calls=[{"name": "list_files", "args": {}, "id": "c1"}]),
        AIMessage(content="Two files."),
    ]
    monkeypatch.setattr(cli_module, "build_chat_model", lambda model=None: object())
    monkeypatch.setattr(cli_module, "run_agent", lambda m, t, task, approve=None: conversation)
    result = CliRunner().invoke(cli, ["agent", "what is here?"])
    assert result.exit_code == 0
    assert "-> list_files" in result.output
    assert "Two files." in result.output


def chat_setup(monkeypatch, run_turn):
    with_key(monkeypatch)
    monkeypatch.setattr(cli_module, "build_chat_model", lambda model=None: object())
    monkeypatch.setattr(cli_module, "build_graph", lambda m, t, approve: object())
    monkeypatch.setattr(cli_module, "run_turn", run_turn)


def test_chat_keeps_the_conversation_between_turns(monkeypatch):
    sizes = []

    def fake_turn(graph, history, text):
        sizes.append(len(history))
        return [*history, HumanMessage(content=text), AIMessage(content=f"re: {text}")]

    chat_setup(monkeypatch, fake_turn)
    result = CliRunner().invoke(cli, ["chat"], input="one\ntwo\n/exit\n")
    assert result.exit_code == 0
    assert sizes == [1, 3]  # system prompt, then +2 messages per turn
    assert "re: one" in result.output
    assert "re: two" in result.output


def test_chat_clear_starts_over(monkeypatch):
    sizes = []

    def fake_turn(graph, history, text):
        sizes.append(len(history))
        return [*history, HumanMessage(content=text), AIMessage(content="ok")]

    chat_setup(monkeypatch, fake_turn)
    CliRunner().invoke(cli, ["chat"], input="one\n/clear\ntwo\n/exit\n")
    assert sizes == [1, 1]


def test_chat_survives_a_failed_turn(monkeypatch):
    calls = []

    def fake_turn(graph, history, text):
        calls.append(text)
        if text == "bad":
            raise ModelRateLimitError("quota")
        return [*history, HumanMessage(content=text), AIMessage(content="fine")]

    chat_setup(monkeypatch, fake_turn)
    result = CliRunner().invoke(cli, ["chat"], input="bad\ngood\n/exit\n")
    assert result.exit_code == 0
    assert calls == ["bad", "good"]
    assert "Rate limit" in result.output
    assert "fine" in result.output


def test_index_then_search_end_to_end(tmp_path, monkeypatch):
    from tests.fakes import HashingEmbedder

    with_key(monkeypatch)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "db.py").write_text("def connect_database():\n    open the database connection\n")
    (tmp_path / "ui.py").write_text("def draw_button():\n    render a button on screen\n")
    monkeypatch.setattr(cli_module, "build_embedder", HashingEmbedder)

    built = CliRunner().invoke(cli, ["index"])
    assert built.exit_code == 0
    assert "Indexed 2 chunks from 2 files" in built.output

    found = CliRunner().invoke(cli, ["search", "database connection", "-k", "1"])
    assert found.exit_code == 0
    assert "db.py:1-2" in found.output
    assert "def connect_database():" in found.output


def test_search_without_an_index_says_what_to_do(tmp_path, monkeypatch):
    with_key(monkeypatch)
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(cli, ["search", "anything"])
    assert result.exit_code != 0
    assert "bloom index" in result.output


def test_search_refuses_an_index_from_another_model(tmp_path, monkeypatch):
    from tests.fakes import HashingEmbedder

    class OtherModel(HashingEmbedder):
        model = "some-other-model"

    with_key(monkeypatch)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "a.py").write_text("x = 1\n")
    monkeypatch.setattr(cli_module, "build_embedder", HashingEmbedder)
    CliRunner().invoke(cli, ["index"])
    monkeypatch.setattr(cli_module, "build_embedder", OtherModel)
    result = CliRunner().invoke(cli, ["search", "x"])
    assert result.exit_code != 0
    assert "Run: bloom index" in result.output


# ---- run and serve -------------------------------------------------------------------------


def test_run_prints_the_answer_and_writes_the_summary(tmp_path, monkeypatch):
    import json

    from bloom.runner import RunSummary

    with_key(monkeypatch)
    seen = {}

    def fake_execute(task, model, allow_changes):
        seen.update(task=task, allow_changes=allow_changes)
        return RunSummary(status="ok", answer="all good", tool_calls=[{"name": "list_files"}])

    monkeypatch.setattr(cli_module, "execute", fake_execute)
    out = tmp_path / "summary.json"
    result = CliRunner().invoke(
        cli, ["run", "do it", "--allow-changes", "--summary-json", str(out)]
    )
    assert result.exit_code == 0
    assert result.output.strip() == "all good"
    assert seen == {"task": "do it", "allow_changes": True}
    assert json.loads(out.read_text())["status"] == "ok"


def test_run_failure_exits_nonzero_but_still_writes_the_summary(tmp_path, monkeypatch):
    import json

    from bloom.runner import RunSummary

    with_key(monkeypatch)
    monkeypatch.setattr(
        cli_module, "execute", lambda t, m, a: RunSummary(status="error", error="Rate limit.")
    )
    out = tmp_path / "summary.json"
    result = CliRunner().invoke(cli, ["run", "x", "--summary-json", str(out)])
    assert result.exit_code != 0
    assert "Rate limit." in result.output
    assert json.loads(out.read_text())["error"] == "Rate limit."


def test_serve_refuses_a_missing_or_short_key(monkeypatch):
    with_key(monkeypatch)
    for key in ("", "short"):
        monkeypatch.setattr(cli_module.settings, "bloom_api_key", SecretStr(key))
        result = CliRunner().invoke(cli, ["serve"])
        assert result.exit_code != 0
        assert "BLOOM_API_KEY" in result.output


def test_serve_starts_the_service_on_the_local_address_by_default(monkeypatch):
    import uvicorn

    with_key(monkeypatch)
    monkeypatch.setattr(cli_module.settings, "bloom_api_key", SecretStr("x" * 20))
    started = {}
    monkeypatch.setattr(
        uvicorn, "run", lambda app, host, port: started.update(host=host, port=port)
    )
    result = CliRunner().invoke(cli, ["serve"])
    assert result.exit_code == 0
    assert started == {"host": "127.0.0.1", "port": 8000}
    assert "Warning" not in result.output


def test_serve_warns_about_a_public_address(monkeypatch):
    import uvicorn

    with_key(monkeypatch)
    monkeypatch.setattr(cli_module.settings, "bloom_api_key", SecretStr("x" * 20))
    monkeypatch.setattr(uvicorn, "run", lambda app, host, port: None)
    result = CliRunner().invoke(cli, ["serve", "--host", "0.0.0.0"])
    assert "Warning" in result.output
