"""Pretrained MiniLM embeddings and cosine-similarity ranking.

Artifacts (git-ignored) live in one directory:
    embeddings.npy   float32 [N, dim], L2-normalized, row i <-> recall_ids[i]
    recall_ids.json  official recall NIDs in row order
    metadata.json    model id + exact revision, dim, corpus fingerprint, versions
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from retrieval import MINILM_MODEL_ID
from retrieval.corpus import CorpusRecord, corpus_fingerprint

EMBEDDINGS_FILE = "embeddings.npy"
IDS_FILE = "recall_ids.json"
METADATA_FILE = "metadata.json"


class StaleIndexError(RuntimeError):
    """Embeddings do not match the current corpus (re-run `python -m retrieval.embed`)."""


class ModelUnavailableError(RuntimeError):
    """The MiniLM model could not be loaded (e.g. offline with an empty HF cache)."""


class Encoder(Protocol):
    def encode(self, texts: list[str]) -> np.ndarray:
        """Return float32 [len(texts), dim] L2-normalized embeddings."""
        ...


class MiniLMEncoder:
    """Lazily loads all-MiniLM-L6-v2 at a pinned revision (CPU is sufficient)."""

    RETRY_AFTER_SECONDS = 60.0

    def __init__(self, revision: str | None = None, model_id: str = MINILM_MODEL_ID) -> None:
        self.model_id = model_id
        self.revision = revision
        self._model = None
        self._lock = threading.Lock()
        self._failure: tuple[float, str] | None = None

    def _load(self):
        with self._lock:
            if self._model is None and self._failure is not None:
                failed_at, message = self._failure
                if time.monotonic() - failed_at < self.RETRY_AFTER_SECONDS:
                    raise ModelUnavailableError(message)  # fail fast, don't block on retries
            if self._model is None:
                try:
                    from sentence_transformers import SentenceTransformer

                    self._model = SentenceTransformer(
                        self.model_id, revision=self.revision, device="cpu"
                    )
                except (ImportError, OSError, ValueError) as error:
                    message = f"Could not load {self.model_id}@{self.revision}: {error}"
                    self._failure = (time.monotonic(), message)
                    raise ModelUnavailableError(message) from error
        return self._model

    @property
    def dimension(self) -> int:
        return int(self._load().get_embedding_dimension())

    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        vectors = self._load().encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)


@dataclass(frozen=True)
class EmbeddingIndex:
    """Precomputed notice embeddings (row i = recall_ids[i]) plus provenance metadata.

    The metadata records which encoder produced the rows and a fingerprint of the corpus
    text, so an index built from different text or a different model is refused at load
    time instead of silently returning wrong neighbours."""
    recall_ids: list[int]
    matrix: np.ndarray  # float32 [N, dim], unit rows
    metadata: dict

    @property
    def dimension(self) -> int:
        return int(self.matrix.shape[1])

    def check_matches(self, corpus: list[CorpusRecord]) -> None:
        """Fail loudly if row order, ids or texts differ from the corpus being served."""
        ids = [record.recall_id for record in corpus]
        if ids != self.recall_ids:
            raise StaleIndexError("Embedding recall_ids do not match the corpus ids/order")
        if self.metadata.get("corpus_fingerprint") != corpus_fingerprint(corpus):
            raise StaleIndexError("Corpus text changed since the embeddings were computed")

    def cosine_scores(self, query_vector: np.ndarray) -> np.ndarray:
        """Cosine similarity in [-1, 1] = dot product of unit vectors."""
        query = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        if query.shape[0] != self.dimension:
            raise ValueError(f"Query dim {query.shape[0]} != index dim {self.dimension}")
        norm = float(np.linalg.norm(query))
        if norm == 0.0:
            raise ValueError("Query embedding has zero norm")
        return (self.matrix @ (query / norm)).astype(np.float64)


def save_embeddings(
    directory: Path, recall_ids: list[int], matrix: np.ndarray, metadata: dict
) -> None:
    """Write the index atomically; metadata is written last, so a crash mid-write leaves an
    index that fails to load rather than a partial one that loads."""
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] != len(recall_ids):
        raise ValueError(f"Matrix shape {matrix.shape} does not match {len(recall_ids)} ids")
    if len(set(recall_ids)) != len(recall_ids):
        raise ValueError("Duplicate recall ids in embedding index")

    directory.mkdir(parents=True, exist_ok=True)
    _atomic(directory / EMBEDDINGS_FILE, lambda tmp: np.save(tmp, matrix, allow_pickle=False))
    _atomic(directory / IDS_FILE, lambda tmp: tmp.write_text(json.dumps(recall_ids), "utf-8"))
    # metadata last: its presence marks a complete set
    _atomic(
        directory / METADATA_FILE,
        lambda tmp: tmp.write_text(json.dumps(metadata, indent=2) + "\n", "utf-8"),
    )


def load_embeddings(directory: Path) -> EmbeddingIndex:
    """Load and validate an index (shapes, counts, finite unit rows). Every failure raises
    StaleIndexError/FileNotFoundError, which the API turns into a per-model 503."""
    paths = [directory / name for name in (EMBEDDINGS_FILE, IDS_FILE, METADATA_FILE)]
    missing = [p.name for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Embedding artifacts missing in {directory}: {missing}")
    try:
        matrix = np.load(paths[0], allow_pickle=False)
        recall_ids = json.loads(paths[1].read_text("utf-8"))
        metadata = json.loads(paths[2].read_text("utf-8"))
    except (OSError, ValueError, EOFError) as error:  # truncated .npy, malformed JSON
        raise StaleIndexError(f"Embedding artifacts are unreadable: {error}") from error
    if matrix.ndim != 2 or matrix.shape[0] != len(recall_ids):
        raise StaleIndexError(f"Embedding shape {matrix.shape} != {len(recall_ids)} ids")
    if metadata.get("count", len(recall_ids)) != len(recall_ids):
        raise StaleIndexError("Embedding count does not match metadata")
    if matrix.shape[1] != metadata.get("dimension"):
        raise StaleIndexError("Embedding dimension does not match metadata")
    matrix = matrix.astype(np.float32)
    if not np.isfinite(matrix).all():
        raise StaleIndexError("Embedding matrix contains NaN or infinite values")
    if not np.allclose(np.linalg.norm(matrix, axis=1), 1.0, atol=1e-3):
        raise StaleIndexError("Embedding rows are not unit-normalized")
    return EmbeddingIndex(recall_ids=recall_ids, matrix=matrix, metadata=metadata)


def _atomic(path: Path, write) -> None:
    tmp = path.with_name(path.name + ".tmp" + path.suffix)  # keep suffix for np.save
    try:
        write(tmp)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
