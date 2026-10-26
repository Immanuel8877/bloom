from bloom.tools import (
    MAX_ENTRIES,
    MAX_LINES,
    MAX_MATCHES,
    edit_file,
    find_files,
    list_files,
    read_file,
    run_command,
    search_text,
    write_file,
)


def test_lists_files_and_folders(tmp_path, monkeypatch):
    (tmp_path / "b.txt").write_text("x")
    (tmp_path / "a_folder").mkdir()
    monkeypatch.chdir(tmp_path)
    assert list_files.invoke({"path": "."}) == "a_folder/\nb.txt"


def test_path_defaults_to_the_root(tmp_path, monkeypatch):
    (tmp_path / "only.txt").write_text("x")
    monkeypatch.chdir(tmp_path)
    assert list_files.invoke({}) == "only.txt"


def test_refuses_paths_outside_the_folder(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert "outside the working folder" in list_files.invoke({"path": ".."})
    assert "outside the working folder" in list_files.invoke({"path": "/etc"})


def test_missing_path_and_file_give_readable_errors(tmp_path, monkeypatch):
    (tmp_path / "f.txt").write_text("x")
    monkeypatch.chdir(tmp_path)
    assert "does not exist" in list_files.invoke({"path": "nope"})
    assert "is a file" in list_files.invoke({"path": "f.txt"})


def test_empty_folder(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert list_files.invoke({}) == "(empty folder)"


def test_long_listings_are_cut_off(tmp_path, monkeypatch):
    for i in range(MAX_ENTRIES + 5):
        (tmp_path / f"f{i:03}.txt").write_text("")
    monkeypatch.chdir(tmp_path)
    lines = list_files.invoke({}).splitlines()
    assert len(lines) == MAX_ENTRIES + 1
    assert lines[-1] == "... 5 more not shown"


def make_project(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("def main():\n    return 1\n# TODO fix\n")
    (tmp_path / "README.md").write_text("hello\n")
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".venv" / "junk.py").write_text("def main(): pass\n")
    (tmp_path / "data.bin").write_bytes(b"\xff\xfe\x00\x01")


def test_read_file_numbers_the_lines(tmp_path, monkeypatch):
    make_project(tmp_path)
    monkeypatch.chdir(tmp_path)
    assert (
        read_file.invoke({"path": "src/app.py"}) == "1: def main():\n2:     return 1\n3: # TODO fix"
    )


def test_read_file_pages_through_long_files(tmp_path, monkeypatch):
    (tmp_path / "big.txt").write_text("\n".join(f"line{i}" for i in range(1, 501)))
    monkeypatch.chdir(tmp_path)
    first = read_file.invoke({"path": "big.txt"})
    assert first.splitlines()[0] == "1: line1"
    assert f"{MAX_LINES}: line{MAX_LINES}" in first
    assert f"Call again with start_line={MAX_LINES + 1}" in first
    second = read_file.invoke({"path": "big.txt", "start_line": 491})
    assert second.splitlines()[-1] == "500: line500"
    assert "more lines" not in second


def test_read_file_errors_are_readable(tmp_path, monkeypatch):
    make_project(tmp_path)
    monkeypatch.chdir(tmp_path)
    assert "outside the working folder" in read_file.invoke({"path": "../x"})
    assert "does not exist" in read_file.invoke({"path": "nope.txt"})
    assert "is a folder" in read_file.invoke({"path": "src"})
    assert "not a text file" in read_file.invoke({"path": "data.bin"})


def test_find_files_by_name_and_by_path(tmp_path, monkeypatch):
    make_project(tmp_path)
    monkeypatch.chdir(tmp_path)
    assert find_files.invoke({"pattern": "*.py"}) == "src/app.py"  # .venv is skipped
    assert find_files.invoke({"pattern": "src/*"}) == "src/app.py"
    assert find_files.invoke({"pattern": "*.zip"}) == "(no files matched)"


def test_search_text_finds_lines_and_skips_junk(tmp_path, monkeypatch):
    make_project(tmp_path)
    monkeypatch.chdir(tmp_path)
    assert search_text.invoke({"pattern": "def main"}) == "src/app.py:1: def main():"
    assert search_text.invoke({"pattern": "TODO", "path": "src"}) == "src/app.py:3: # TODO fix"
    assert search_text.invoke({"pattern": "zzz"}) == "(no matches)"


def test_search_text_errors_and_limits(tmp_path, monkeypatch):
    (tmp_path / "many.txt").write_text("hit\n" * (MAX_MATCHES + 50))
    monkeypatch.chdir(tmp_path)
    assert "invalid regular expression" in search_text.invoke({"pattern": "("})
    assert "outside the working folder" in search_text.invoke({"pattern": "x", "path": ".."})
    lines = search_text.invoke({"pattern": "hit"}).splitlines()
    assert len(lines) == MAX_MATCHES + 1
    assert lines[-1] == f"... stopped after {MAX_MATCHES} matches"


def test_write_file_creates_and_replaces(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert write_file.invoke({"path": "new/a.txt", "content": "hi\nthere"}) == (
        "Created new/a.txt (2 lines)"
    )
    assert (tmp_path / "new" / "a.txt").read_text() == "hi\nthere"
    assert write_file.invoke({"path": "new/a.txt", "content": "x"}).startswith("Replaced")
    assert "outside the working folder" in write_file.invoke({"path": "../x", "content": "x"})


def test_edit_file_replaces_exactly_one_match(tmp_path, monkeypatch):
    (tmp_path / "a.txt").write_text("one two two")
    monkeypatch.chdir(tmp_path)
    assert "appears 2 times" in edit_file.invoke(
        {"path": "a.txt", "old_text": "two", "new_text": "2"}
    )
    assert "not found" in edit_file.invoke({"path": "a.txt", "old_text": "zzz", "new_text": "2"})
    assert edit_file.invoke({"path": "a.txt", "old_text": "one", "new_text": "1"}) == "Edited a.txt"
    assert (tmp_path / "a.txt").read_text() == "1 two two"
    assert "not an existing file" in edit_file.invoke(
        {"path": "nope.txt", "old_text": "a", "new_text": "b"}
    )


def test_run_command_returns_exit_code_and_output(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert run_command.invoke({"command": "echo hello"}) == "exit code 0\nhello"
    assert run_command.invoke({"command": "exit 3"}).startswith("exit code 3")


def test_run_command_times_out(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("bloom.tools.COMMAND_TIMEOUT_S", 1)
    assert "stopped after 1 seconds" in run_command.invoke({"command": "sleep 5"})


def test_secret_env_file_is_never_searched(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("GEMINI_API_KEY=super-secret")
    (tmp_path / "a.txt").write_text("nothing")
    monkeypatch.chdir(tmp_path)
    assert search_text.invoke({"pattern": "super-secret"}) == "(no matches)"
    assert find_files.invoke({"pattern": ".env"}) == "(no files matched)"
