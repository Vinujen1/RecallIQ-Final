"""Build a SearchEngine from the on-disk database and embedding artifacts."""

from __future__ import annotations

from pathlib import Path

from retrieval import MINILM_MODEL_ID
from retrieval.corpus import load_corpus
from retrieval.search import SearchEngine
from retrieval.semantic import Encoder, MiniLMEncoder, StaleIndexError, load_embeddings


def _finetuned_entry(corpus, checkpoint: Path | None, embeddings_dir: Path | None,
                     encoder: Encoder | None):
    """(index, encoder, error) for one fine-tuned model; never raises."""
    from retrieval.finetuned import CheckpointEncoder, check_compatible
    from retrieval.semantic import ModelUnavailableError

    if checkpoint is None or embeddings_dir is None:
        return None, None, None
    try:
        index = load_embeddings(embeddings_dir)
        index.check_matches(corpus)
        check_compatible(index, checkpoint)
    except (FileNotFoundError, StaleIndexError, ValueError, KeyError) as error:
        return None, None, (f"unavailable: {error}. Train the model and build its index with "
                            "`python -m retrieval.finetuned`.")
    if encoder is None:
        # Load eagerly, right after the weights-hash check, so the network serving queries
        # is the one that built the index (a lazy load could pick up a later `best/`).
        try:
            encoder = CheckpointEncoder(checkpoint).load()
        except ModelUnavailableError as error:
            return None, None, f"unavailable: {error}"
    return index, encoder, None


def build_engine(
    db_path: Path, embeddings_dir: Path, encoder: Encoder | None = None,
    finetuned: dict[str, tuple[Path | None, Path | None]] | None = None,
    finetuned_encoders: dict[str, Encoder] | None = None,
) -> SearchEngine:
    """BM25 always; MiniLM only when valid embeddings exist (otherwise a clear reason).

    `encoder` is injectable for tests; the real encoder must match the model id and the
    exact revision recorded with the embeddings, so queries and documents share a space.
    """
    corpus = load_corpus(db_path)
    dense = {}
    for name, (checkpoint, index_dir) in (finetuned or {}).items():
        idx, enc, err = _finetuned_entry(corpus, checkpoint, index_dir,
                                         (finetuned_encoders or {}).get(name))
        dense[name] = (idx, enc, f"{name} {err}" if err else None)
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
            dense=dense,
        )
    if encoder is None:
        encoder = MiniLMEncoder(revision=embeddings.metadata["model_revision"])
    return SearchEngine(corpus, embeddings=embeddings, encoder=encoder, dense=dense)
