"""Build a SearchEngine from the on-disk database and embedding artifacts."""

from __future__ import annotations

from pathlib import Path

from retrieval import MINILM_MODEL_ID
from retrieval.corpus import load_corpus
from retrieval.search import SearchEngine
from retrieval.semantic import Encoder, MiniLMEncoder, StaleIndexError, load_embeddings


def build_engine(
    db_path: Path, embeddings_dir: Path, encoder: Encoder | None = None
) -> SearchEngine:
    """BM25 always; MiniLM only when valid embeddings exist (otherwise a clear reason).

    `encoder` is injectable for tests; the real encoder must match the model id and the
    exact revision recorded with the embeddings, so queries and documents share a space.
    """
    corpus = load_corpus(db_path)
    try:
        embeddings = load_embeddings(embeddings_dir)
        embeddings.check_matches(corpus)
        if encoder is None:
            if embeddings.metadata.get("model_id") != MINILM_MODEL_ID:
                raise StaleIndexError(
                    f"Embeddings were built with {embeddings.metadata.get('model_id')!r}, "
                    f"expected {MINILM_MODEL_ID!r}"
                )
            if not embeddings.metadata.get("model_revision"):
                raise StaleIndexError("Embedding metadata has no pinned model_revision")
    except (FileNotFoundError, StaleIndexError) as error:
        return SearchEngine(
            corpus,
            embeddings_error=f"MiniLM search unavailable: {error}. "
            "Run `python -m retrieval.embed` to (re)build the embeddings.",
        )
    if encoder is None:
        encoder = MiniLMEncoder(revision=embeddings.metadata["model_revision"])
    return SearchEngine(corpus, embeddings=embeddings, encoder=encoder)
