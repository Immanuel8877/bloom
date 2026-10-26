import numpy as np
import pytest

from bloom.llm import GeminiEmbedder
from bloom.retrieval.chunker import Chunk, chunk_text, collect_chunks
from bloom.retrieval.embedder import normalize
from bloom.retrieval.index import CodeIndex
from tests.fakes import HashingEmbedder


def numbered(n: int) -> str:
    return "\n".join(f"line{i}" for i in range(1, n + 1))


# ---- chunker -------------------------------------------------------------------------------


def test_short_file_is_one_chunk():
    chunks = chunk_text("a.py", numbered(10), size=40, overlap=5)
    assert chunks == [Chunk("a.py", 1, 10, numbered(10))]


def test_long_file_is_cut_with_overlap():
    chunks = chunk_text("a.py", numbered(100), size=40, overlap=5)
    assert [(c.start_line, c.end_line) for c in chunks] == [(1, 40), (36, 75), (71, 100)]


def test_chunks_cover_every_line():
    chunks = chunk_text("a.py", numbered(97), size=30, overlap=4)
    covered = set()
    for chunk in chunks:
        covered.update(range(chunk.start_line, chunk.end_line + 1))
    assert covered == set(range(1, 98))


def test_blank_chunks_are_skipped_and_empty_text_gives_nothing():
    assert chunk_text("a.py", "") == []
    assert chunk_text("a.py", "\n\n\n") == []


def test_bad_overlap_is_rejected():
    with pytest.raises(ValueError):
        chunk_text("a.py", "x", size=10, overlap=10)


def test_embedding_text_includes_the_path():
    assert Chunk("src/a.py", 1, 1, "x = 1").embedding_text == "src/a.py\nx = 1"


def test_collect_chunks_skips_noise(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("print('hi')\n")
    (tmp_path / "README.md").write_text("# hello\n")
    (tmp_path / "uv.lock").write_text("lots of lock text\n")
    (tmp_path / "pic.png").write_bytes(b"\x89PNG\xff\xfe")
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".venv" / "junk.py").write_text("x = 1\n")
    (tmp_path / ".env").write_text("GEMINI_API_KEY=secret\n")
    paths = sorted(c.path for c in collect_chunks(tmp_path.resolve()))
    assert paths == ["README.md", "src/app.py"]


# ---- vectors -------------------------------------------------------------------------------


def test_normalize_makes_unit_vectors_and_handles_zero():
    result = normalize(np.array([[3.0, 4.0], [0.0, 0.0]]))
    assert np.allclose(np.linalg.norm(result[0]), 1.0)
    assert np.allclose(result[1], [0.0, 0.0])
    assert result.dtype == np.float32


def test_normalize_accepts_a_single_vector():
    assert np.allclose(normalize(np.array([0.0, 5.0])), [0.0, 1.0])


# ---- index ---------------------------------------------------------------------------------


def sample_chunks():
    return [
        Chunk("db.py", 1, 5, "def connect_database(): open the database connection pool"),
        Chunk("ui.py", 1, 5, "def draw_button(): render a button on the screen"),
        Chunk("mail.py", 1, 5, "def send_email(): send an email message to a user"),
    ]


def test_search_finds_the_chunk_sharing_the_most_words():
    embedder = HashingEmbedder()
    index = CodeIndex.build(sample_chunks(), embedder)
    hits = index.search(embedder.embed_query("database connection"), k=3)
    assert hits[0][0].path == "db.py"
    assert hits[0][1] > hits[1][1]  # best first
    assert len(hits) == 3


def test_search_never_returns_more_than_exist():
    embedder = HashingEmbedder()
    index = CodeIndex.build(sample_chunks(), embedder)
    assert len(index.search(embedder.embed_query("email"), k=50)) == 3


def test_index_survives_save_and_load(tmp_path):
    embedder = HashingEmbedder()
    CodeIndex.build(sample_chunks(), embedder).save(tmp_path / "idx")
    loaded = CodeIndex.load(tmp_path / "idx")
    assert loaded.model == "fake-hash"
    assert loaded.chunks == sample_chunks()
    assert loaded.search(embedder.embed_query("send an email"), k=1)[0][0].path == "mail.py"


def test_load_without_an_index_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        CodeIndex.load(tmp_path / "missing")


def test_building_an_empty_index_is_an_error():
    with pytest.raises(ValueError):
        CodeIndex.build([], HashingEmbedder())


# ---- the Gemini embedder (with a stub client: no network) ---------------------------------


class StubClient:
    def __init__(self):
        self.calls = []

    def embed_documents(self, texts, **kwargs):
        self.calls.append(("documents", kwargs))
        return [[3.0, 4.0] for _ in texts]

    def embed_query(self, text, **kwargs):
        self.calls.append(("query", kwargs))
        return [0.0, 2.0]


def test_gemini_embedder_uses_the_right_task_types_and_normalizes():
    client = StubClient()
    embedder = GeminiEmbedder("m", 2, "key", client=client)
    docs = embedder.embed_documents(["a", "b"])
    query = embedder.embed_query("q")
    assert client.calls == [
        ("documents", {"task_type": "RETRIEVAL_DOCUMENT", "output_dimensionality": 2}),
        ("query", {"task_type": "CODE_RETRIEVAL_QUERY", "output_dimensionality": 2}),
    ]
    assert np.allclose(np.linalg.norm(docs, axis=1), 1.0)
    assert np.allclose(query, [0.0, 1.0])


# ---- keeping the index fresh ---------------------------------------------------------------


class CountingEmbedder(HashingEmbedder):
    """Remembers every text it was asked to embed, so tests can see what was re-embedded."""

    def __init__(self):
        self.embedded: list[str] = []

    def embed_documents(self, texts):
        self.embedded.extend(texts)
        return super().embed_documents(texts)


def touch(path, seconds_later: int = 10):
    """Move a file's modified time forward, so a change is visible even on a coarse clock."""
    import os

    stat = path.stat()
    later = stat.st_mtime_ns + seconds_later * 1_000_000_000
    os.utime(path, ns=(later, later))


def make_project(tmp_path):
    (tmp_path / "db.py").write_text("def connect_database():\n    open the database connection\n")
    (tmp_path / "ui.py").write_text("def draw_button():\n    render a button on screen\n")
    (tmp_path / "mail.py").write_text("def send_email():\n    send an email message\n")
    return tmp_path.resolve()


def test_from_folder_records_every_file(tmp_path):
    root = make_project(tmp_path)
    index = CodeIndex.from_folder(root, HashingEmbedder())
    assert sorted(index.files) == ["db.py", "mail.py", "ui.py"]
    assert len(index.chunks) == 3


def test_refresh_does_nothing_when_nothing_changed(tmp_path):
    root = make_project(tmp_path)
    embedder = CountingEmbedder()
    index = CodeIndex.from_folder(root, embedder)
    embedder.embedded.clear()
    assert index.refresh(root, embedder) is False
    assert embedder.embedded == []


def test_refresh_only_re_embeds_changed_and_new_files(tmp_path):
    root = make_project(tmp_path)
    embedder = CountingEmbedder()
    index = CodeIndex.from_folder(root, embedder)
    embedder.embedded.clear()

    (root / "db.py").write_text("def open_pool():\n    pool of postgres connections\n")
    touch(root / "db.py")
    (root / "new.py").write_text("def parse_config():\n    read the yaml settings\n")

    assert index.refresh(root, embedder) is True
    assert sorted(text.split("\n")[0] for text in embedder.embedded) == ["db.py", "new.py"]
    assert sorted(index.files) == ["db.py", "mail.py", "new.py", "ui.py"]
    assert len(index.chunks) == 4
    hits = index.search(embedder.embed_query("postgres connections pool"), k=1)
    assert hits[0][0].path == "db.py"
    assert "postgres" in hits[0][0].text  # the new text, not the old


def test_refresh_drops_deleted_files(tmp_path):
    root = make_project(tmp_path)
    embedder = CountingEmbedder()
    index = CodeIndex.from_folder(root, embedder)
    embedder.embedded.clear()
    (root / "ui.py").unlink()
    assert index.refresh(root, embedder) is True
    assert embedder.embedded == []  # nothing new to embed
    assert "ui.py" not in index.files
    assert [c.path for c in index.chunks] == ["db.py", "mail.py"]
    assert all(chunk.path != "ui.py" for chunk, _ in index.search(embedder.embed_query("button")))


def test_refresh_keeps_working_after_save_and_load(tmp_path):
    root = make_project(tmp_path)
    embedder = CountingEmbedder()
    saved_at = root / ".bloom" / "index"  # the real location, which indexing skips
    CodeIndex.from_folder(root, embedder).save(saved_at)
    loaded = CodeIndex.load(saved_at)
    embedder.embedded.clear()
    assert loaded.refresh(root, embedder) is False  # the saved mtimes still match


def test_refresh_refuses_a_different_model(tmp_path):
    root = make_project(tmp_path)
    index = CodeIndex.from_folder(root, HashingEmbedder())

    class Other(HashingEmbedder):
        model = "other"

    with pytest.raises(ValueError, match="Run: bloom index"):
        index.refresh(root, Other())


def test_search_rejects_a_query_of_the_wrong_size(tmp_path):
    root = make_project(tmp_path)
    index = CodeIndex.from_folder(root, HashingEmbedder())
    with pytest.raises(ValueError, match="different vector sizes"):
        index.search(np.ones(10, dtype=np.float32))
