"""Step 3 of code search: store the vectors in FAISS and find the closest ones."""

from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np

from bloom.retrieval.chunker import Chunk, file_chunks, snapshot
from bloom.retrieval.embedder import Embedder

INDEX_DIR = Path(".bloom") / "index"
INDEX_FILE = "faiss.index"
CHUNKS_FILE = "chunks.json"


class CodeIndex:
    """A FAISS index plus the list of chunks it was built from.

    FAISS only stores vectors and returns row numbers. Row i of the index is chunks[i].
    `files` remembers each file's modified time when it was indexed, so we can tell later
    which files changed.
    """

    def __init__(
        self,
        index: faiss.Index,
        chunks: list[Chunk],
        model: str,
        files: dict[str, int] | None = None,
    ) -> None:
        self.index = index
        self.chunks = chunks
        self.model = model
        self.files = files or {}

    @classmethod
    def build(
        cls, chunks: list[Chunk], embedder: Embedder, files: dict[str, int] | None = None
    ) -> CodeIndex:
        """Embed every chunk and put the vectors into a flat inner-product index."""
        if not chunks:
            raise ValueError("there is nothing to index")
        vectors = embedder.embed_documents([chunk.embedding_text for chunk in chunks])
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(np.ascontiguousarray(vectors, dtype=np.float32))
        return cls(index, chunks, embedder.model, files)

    @classmethod
    def from_folder(cls, root: Path, embedder: Embedder) -> CodeIndex:
        """Index every indexable file under `root` from scratch."""
        files = snapshot(root)
        chunks = [chunk for path in files for chunk in file_chunks(root, path)]
        return cls.build(chunks, embedder, files)

    def refresh(self, root: Path, embedder: Embedder) -> bool:
        """Bring the index up to date with the files on disk. True if anything changed.

        Only new and modified files are embedded again. Vectors of unchanged files are
        copied out of the old index, so a small edit costs one small embedding call.
        """
        if embedder.model != self.model:
            raise ValueError(
                f"The index was built with '{self.model}' but the embedding model is "
                f"'{embedder.model}'. Run: bloom index"
            )
        current = snapshot(root)
        changed = {path for path, mtime in current.items() if self.files.get(path) != mtime}
        gone = set(self.files) - set(current)
        if not changed and not gone:
            return False

        stale = changed | gone
        keep = [i for i, chunk in enumerate(self.chunks) if chunk.path not in stale]
        kept_vectors = self.index.reconstruct_n(0, self.index.ntotal)[keep]
        kept_chunks = [self.chunks[i] for i in keep]

        new_chunks = [chunk for path in sorted(changed) for chunk in file_chunks(root, path)]
        vectors = kept_vectors
        if new_chunks:
            new_vectors = embedder.embed_documents([c.embedding_text for c in new_chunks])
            if new_vectors.shape[1] != self.index.d:
                raise ValueError(
                    "The embedding size changed (EMBEDDING_DIMENSIONS). Run: bloom index"
                )
            vectors = np.vstack([kept_vectors, new_vectors])

        index = faiss.IndexFlatIP(self.index.d)
        index.add(np.ascontiguousarray(vectors, dtype=np.float32))
        # Only now, with everything embedded successfully, replace the old state.
        self.index, self.chunks, self.files = index, kept_chunks + new_chunks, current
        return True

    def search(self, query_vector: np.ndarray, k: int = 5) -> list[tuple[Chunk, float]]:
        """The `k` chunks closest in meaning to the query, best first, with their scores."""
        query = np.ascontiguousarray(query_vector, dtype=np.float32).reshape(1, -1)
        if query.shape[1] != self.index.d:
            raise ValueError(
                "The index and the embedding model use different vector sizes. Run: bloom index"
            )
        if not self.chunks:
            return []
        scores, rows = self.index.search(query, min(k, len(self.chunks)))
        return [
            (self.chunks[row], float(score))
            for score, row in zip(scores[0], rows[0], strict=True)
            if row != -1  # FAISS uses -1 for "no result"
        ]

    def save(self, folder: Path = INDEX_DIR) -> None:
        """Write the index and its chunk list to `folder`."""
        folder.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(folder / INDEX_FILE))
        payload = {
            "model": self.model,
            "files": self.files,
            "chunks": [
                {"path": c.path, "start_line": c.start_line, "end_line": c.end_line, "text": c.text}
                for c in self.chunks
            ],
        }
        (folder / CHUNKS_FILE).write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, folder: Path = INDEX_DIR) -> CodeIndex:
        """Read an index back from `folder`. Raises FileNotFoundError if there is none."""
        for name in (INDEX_FILE, CHUNKS_FILE):
            if not (folder / name).is_file():  # FAISS itself would raise a confusing RuntimeError
                raise FileNotFoundError(folder / name)
        index = faiss.read_index(str(folder / INDEX_FILE))
        payload = json.loads((folder / CHUNKS_FILE).read_text(encoding="utf-8"))
        chunks = [Chunk(**item) for item in payload["chunks"]]
        return cls(index, chunks, payload["model"], payload.get("files", {}))
