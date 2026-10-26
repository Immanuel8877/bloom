"""Step 1 of code search: cut files into small, overlapping pieces (chunks)."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from bloom.tools import iter_files

CHUNK_LINES = 40
OVERLAP_LINES = 5
MAX_FILE_BYTES = 200_000
# Only index files people actually read. Lock files and binaries would just be noise.
INDEXED_SUFFIXES = {
    ".py", ".md", ".txt", ".toml", ".json", ".yaml", ".yml", ".js", ".ts", ".tsx",
    ".html", ".css", ".sh", ".sql", ".go", ".rs", ".java", ".c", ".h", ".cpp",
}  # fmt: skip


@dataclass(frozen=True)
class Chunk:
    """A piece of one file: lines start_line..end_line (1-based, inclusive)."""

    path: str
    start_line: int
    end_line: int
    text: str

    @property
    def embedding_text(self) -> str:
        """What gets embedded: the text plus its file path, which says a lot about meaning."""
        return f"{self.path}\n{self.text}"


def chunk_text(
    path: str, text: str, size: int = CHUNK_LINES, overlap: int = OVERLAP_LINES
) -> list[Chunk]:
    """Cut `text` into chunks of `size` lines. Neighbours share `overlap` lines.

    The overlap means a function that straddles a boundary still appears whole in one chunk.
    """
    if not 0 <= overlap < size:
        raise ValueError("overlap must be at least 0 and smaller than size")
    lines = text.splitlines()
    chunks: list[Chunk] = []
    start = 0
    while start < len(lines):
        end = min(start + size, len(lines))
        body = "\n".join(lines[start:end])
        if body.strip():  # skip chunks that are only blank lines
            chunks.append(Chunk(path, start + 1, end, body))
        if end == len(lines):
            break
        start = end - overlap
    return chunks


def indexable_files(root: Path) -> Iterator[Path]:
    """The files worth indexing: a known type, not too big, outside the noise folders."""
    for file in iter_files(root):
        if file.suffix not in INDEXED_SUFFIXES:
            continue
        try:
            if file.stat().st_size <= MAX_FILE_BYTES:
                yield file
        except OSError:
            continue


def snapshot(root: Path) -> dict[str, int]:
    """Each indexable file's path and last-modified time. Cheap: no file is read."""
    found: dict[str, int] = {}
    for file in indexable_files(root):
        try:
            found[file.relative_to(root).as_posix()] = file.stat().st_mtime_ns
        except OSError:
            continue  # deleted while we were looking
    return found


def file_chunks(root: Path, relative_path: str) -> list[Chunk]:
    """The chunks of one file. Unreadable or binary files give no chunks."""
    try:
        text = (root / relative_path).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []
    return chunk_text(relative_path, text)


def collect_chunks(root: Path) -> list[Chunk]:
    """Chunk every indexable file under `root`."""
    chunks: list[Chunk] = []
    for relative_path in snapshot(root):
        chunks.extend(file_chunks(root, relative_path))
    return chunks
