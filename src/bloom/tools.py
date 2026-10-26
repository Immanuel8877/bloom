"""Tools the model can ask bloom to run. Each tool is a plain function plus @tool."""

from __future__ import annotations

import fnmatch
import os
import re
import subprocess
from collections.abc import Iterator
from pathlib import Path

from langchain_core.tools import tool

MAX_ENTRIES = 200
MAX_LINES = 200
MAX_MATCHES = 100
MAX_LINE_CHARS = 300
MAX_OUTPUT_CHARS = 4000
COMMAND_TIMEOUT_S = 30
# Folders that are noise for a coding agent: not your code, and often huge.
IGNORED_DIRS = {
    ".git",
    ".venv",
    ".bloom",
    "__pycache__",
    "node_modules",
    ".pytest_cache",
    ".ruff_cache",
}

# Files that are never searched or listed by content: they hold secrets.
IGNORED_FILES = {".env"}


def resolve_inside(path: str) -> Path:
    """Turn `path` into a real path, refusing anything outside the current folder."""
    root = Path.cwd().resolve()
    target = (root / path).resolve()
    if not target.is_relative_to(root):
        raise ValueError(f"'{path}' is outside the working folder")
    return target


@tool
def list_files(path: str = ".") -> str:
    """List the files and folders inside a folder of the project.

    Args:
        path: Folder to list, relative to the project root. Use "." for the root.
    """
    try:
        folder = resolve_inside(path)
    except ValueError as exc:
        return f"Error: {exc}"
    if not folder.exists():
        return f"Error: '{path}' does not exist"
    if not folder.is_dir():
        return f"Error: '{path}' is a file, not a folder"

    entries = sorted(folder.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
    names = [f"{p.name}/" if p.is_dir() else p.name for p in entries]
    if not names:
        return "(empty folder)"
    shown = names[:MAX_ENTRIES]
    if len(names) > MAX_ENTRIES:
        shown.append(f"... {len(names) - MAX_ENTRIES} more not shown")
    return "\n".join(shown)


def iter_files(folder: Path) -> Iterator[Path]:
    """Yield every file under `folder`, in a stable order, skipping IGNORED_DIRS."""
    for current, dirs, files in os.walk(folder):
        dirs[:] = sorted(d for d in dirs if d not in IGNORED_DIRS)
        for name in sorted(f for f in files if f not in IGNORED_FILES):
            yield Path(current) / name


def relative(path: Path) -> str:
    """A path as the model sees it: relative to the project root."""
    return path.relative_to(Path.cwd().resolve()).as_posix()


@tool
def read_file(path: str, start_line: int = 1, max_lines: int = MAX_LINES) -> str:
    """Read a text file and return its lines, each starting with its line number.

    Args:
        path: File to read, relative to the project root.
        start_line: First line to show (1 is the top of the file).
        max_lines: How many lines to show at most.
    """
    try:
        file = resolve_inside(path)
    except ValueError as exc:
        return f"Error: {exc}"
    if not file.exists():
        return f"Error: '{path}' does not exist"
    if not file.is_file():
        return f"Error: '{path}' is a folder, not a file. Use list_files."
    try:
        lines = file.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        return f"Error: '{path}' is not a text file"

    start = max(start_line, 1)
    count = max(1, min(max_lines, MAX_LINES))
    chunk = lines[start - 1 : start - 1 + count]
    if not chunk:
        return f"(no lines: the file has {len(lines)} lines)"
    text = "\n".join(f"{n}: {line}" for n, line in enumerate(chunk, start=start))
    last = start + len(chunk) - 1
    if last < len(lines):
        text += f"\n... {len(lines) - last} more lines. Call again with start_line={last + 1}."
    return text


@tool
def find_files(pattern: str) -> str:
    """Find files by name pattern, such as "*.py" or "tests/*.py".

    A pattern without a "/" is matched against file names anywhere in the project.
    A pattern with a "/" is matched against the whole path from the project root.

    Args:
        pattern: A glob pattern. "*" matches any characters.
    """
    root = Path.cwd().resolve()
    found = []
    for file in iter_files(root):
        rel = relative(file)
        target = rel if "/" in pattern else file.name
        if fnmatch.fnmatch(target, pattern):
            found.append(rel)
    if not found:
        return "(no files matched)"
    shown = found[:MAX_ENTRIES]
    if len(found) > MAX_ENTRIES:
        shown.append(f"... {len(found) - MAX_ENTRIES} more not shown")
    return "\n".join(shown)


@tool
def search_text(pattern: str, path: str = ".") -> str:
    """Search inside files for a regular expression. Returns path:line: text for each match.

    Args:
        pattern: The regular expression to look for, for example "def main" or "TODO".
        path: Folder (or single file) to search, relative to the project root.
    """
    try:
        start = resolve_inside(path)
        regex = re.compile(pattern)
    except ValueError as exc:
        return f"Error: {exc}"
    except re.error as exc:
        return f"Error: invalid regular expression: {exc}"
    if not start.exists():
        return f"Error: '{path}' does not exist"

    files = [start] if start.is_file() else iter_files(start)
    matches = []
    for file in files:
        try:
            lines = file.read_text(encoding="utf-8").splitlines()
        except (UnicodeDecodeError, OSError):
            continue  # binary or unreadable files are skipped
        for number, line in enumerate(lines, start=1):
            if regex.search(line):
                matches.append(f"{relative(file)}:{number}: {line.strip()[:MAX_LINE_CHARS]}")
                if len(matches) > MAX_MATCHES:
                    break
        if len(matches) > MAX_MATCHES:
            break
    if not matches:
        return "(no matches)"
    if len(matches) > MAX_MATCHES:
        matches = [*matches[:MAX_MATCHES], f"... stopped after {MAX_MATCHES} matches"]
    return "\n".join(matches)


@tool
def write_file(path: str, content: str) -> str:
    """Create a file, or replace it completely. Missing folders are created.

    Args:
        path: File to write, relative to the project root.
        content: The full new text of the file.
    """
    try:
        file = resolve_inside(path)
    except ValueError as exc:
        return f"Error: {exc}"
    if file.is_dir():
        return f"Error: '{path}' is a folder"
    existed = file.exists()
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(content, encoding="utf-8")
    verb = "Replaced" if existed else "Created"
    return f"{verb} {path} ({len(content.splitlines())} lines)"


@tool
def edit_file(path: str, old_text: str, new_text: str) -> str:
    """Change part of an existing file by replacing one exact piece of text.

    The old_text must appear exactly once in the file. Include enough surrounding
    lines to make it unique.

    Args:
        path: File to edit, relative to the project root.
        old_text: The exact text to replace.
        new_text: What to put there instead.
    """
    try:
        file = resolve_inside(path)
    except ValueError as exc:
        return f"Error: {exc}"
    if not file.is_file():
        return f"Error: '{path}' is not an existing file"
    try:
        text = file.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return f"Error: '{path}' is not a text file"
    count = text.count(old_text)
    if count == 0:
        return "Error: old_text was not found. Read the file again and copy the text exactly."
    if count > 1:
        return (
            f"Error: old_text appears {count} times. Add more surrounding lines to make it unique."
        )
    file.write_text(text.replace(old_text, new_text), encoding="utf-8")
    return f"Edited {path}"


@tool
def run_command(command: str) -> str:
    """Run a shell command in the project folder and return its output.

    Use it for things like running tests. Commands that take longer than 30 seconds are stopped.

    Args:
        command: The shell command, for example "uv run pytest".
    """
    try:
        done = subprocess.run(
            command,
            shell=True,
            cwd=Path.cwd(),
            capture_output=True,
            text=True,
            timeout=COMMAND_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        return f"Error: the command was stopped after {COMMAND_TIMEOUT_S} seconds"
    output = (done.stdout + done.stderr).strip() or "(no output)"
    if len(output) > MAX_OUTPUT_CHARS:
        output = output[:MAX_OUTPUT_CHARS] + "\n... output cut off"
    return f"exit code {done.returncode}\n{output}"


TOOLS = [list_files, read_file, find_files, search_text, write_file, edit_file, run_command]
