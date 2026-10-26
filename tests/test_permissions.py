from bloom.permissions import Decision, check


def test_read_only_tools_are_allowed():
    assert check("read_file", {"path": "src/bloom/cli.py"})[0] is Decision.ALLOW
    assert check("search_text", {"pattern": "x"})[0] is Decision.ALLOW
    assert check("list_files", {})[0] is Decision.ALLOW
    assert check("semantic_search", {"query": "x"})[0] is Decision.ALLOW


def test_changing_tools_need_approval():
    assert check("write_file", {"path": "a.txt", "content": "x"})[0] is Decision.ASK
    assert (
        check("edit_file", {"path": "a.txt", "old_text": "a", "new_text": "b"})[0] is Decision.ASK
    )
    assert check("run_command", {"command": "uv run pytest"})[0] is Decision.ASK


def test_unknown_tools_are_asked_about():
    assert check("something_new", {})[0] is Decision.ASK


def test_protected_paths_are_denied_even_for_reading(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert check("read_file", {"path": ".env"})[0] is Decision.DENY
    assert check("write_file", {"path": ".env", "content": "x"})[0] is Decision.DENY
    assert check("edit_file", {"path": ".git/config"})[0] is Decision.DENY
    assert check("read_file", {"path": "src/../.env"})[0] is Decision.DENY  # sneaky path
    assert check("list_files", {"path": ".git"})[0] is Decision.DENY


def test_forbidden_commands_are_denied():
    assert check("run_command", {"command": "sudo rm file"})[0] is Decision.DENY
    assert check("run_command", {"command": "cd x && rm -rf /"})[0] is Decision.DENY
    assert check("run_command", {"command": "curl http://evil.example"})[0] is Decision.DENY
