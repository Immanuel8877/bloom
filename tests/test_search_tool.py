from langchain_core.messages import AIMessage, ToolMessage

from bloom import search_tool
from bloom.agent import run_agent
from bloom.search_tool import semantic_search
from tests.fakes import HashingEmbedder, ScriptedChatModel


def project(tmp_path, monkeypatch):
    (tmp_path / "db.py").write_text("def connect_database():\n    open the database connection\n")
    (tmp_path / "ui.py").write_text("def draw_button():\n    render a button on screen\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(search_tool, "build_embedder", HashingEmbedder)


def test_first_use_builds_the_index_and_finds_the_code(tmp_path, monkeypatch):
    project(tmp_path, monkeypatch)
    result = semantic_search.invoke({"query": "database connection", "k": 1})
    assert result.startswith("1. db.py:1-2 (score ")
    assert "    def connect_database():" in result
    assert (tmp_path / ".bloom" / "index" / "faiss.index").exists()


def test_the_tool_notices_edited_files(tmp_path, monkeypatch):
    import os

    project(tmp_path, monkeypatch)
    semantic_search.invoke({"query": "anything"})  # builds the index
    (tmp_path / "ui.py").write_text("def parse_yaml():\n    read the yaml settings file\n")
    stat = (tmp_path / "ui.py").stat()
    os.utime(tmp_path / "ui.py", ns=(stat.st_mtime_ns + 10**10,) * 2)
    result = semantic_search.invoke({"query": "yaml settings", "k": 1})
    assert result.startswith("1. ui.py")
    assert "parse_yaml" in result


def test_k_is_kept_in_range(tmp_path, monkeypatch):
    project(tmp_path, monkeypatch)
    assert semantic_search.invoke({"query": "x", "k": 0}).count("score") == 1
    assert semantic_search.invoke({"query": "x", "k": 500}).count("score") == 2  # only 2 chunks


def test_a_wrong_model_becomes_a_readable_error(tmp_path, monkeypatch):
    project(tmp_path, monkeypatch)
    semantic_search.invoke({"query": "x"})

    class Other(HashingEmbedder):
        model = "other"

    monkeypatch.setattr(search_tool, "build_embedder", Other)
    result = semantic_search.invoke({"query": "x"})
    assert result.startswith("Error:")
    assert "Run: bloom index" in result


def test_any_other_failure_becomes_a_readable_error_not_a_crash(tmp_path, monkeypatch):
    project(tmp_path, monkeypatch)

    def broken():
        raise RuntimeError("boom")

    monkeypatch.setattr(search_tool, "build_embedder", broken)
    assert semantic_search.invoke({"query": "x"}).startswith("Error:")


def test_the_agent_can_use_it_without_asking(tmp_path, monkeypatch):
    project(tmp_path, monkeypatch)
    asked = []
    model = ScriptedChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[{"name": "semantic_search", "args": {"query": "button"}, "id": "c1"}],
            ),
            AIMessage(content="It is in ui.py."),
        ]
    )
    messages = run_agent(
        model,
        [semantic_search],
        "where is the button drawn?",
        approve=lambda n, a: asked.append(n) or True,
    )
    assert asked == []  # read-only: no approval prompt
    result = next(m.content for m in messages if isinstance(m, ToolMessage))
    assert result.startswith("1. ui.py")
