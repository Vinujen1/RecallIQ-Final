"""One ranking interface for every compared system, all over the SAME 1,000-record index.

    bm25          Baseline 0 - Okapi BM25 (no training)
    minilm        Model A    - pretrained all-MiniLM-L6-v2 (stored Phase 2A embeddings)
    <label>=<dir> Model B/C  - a fine-tuned checkpoint (training/checkpoints/best); the
                               index is re-encoded with that checkpoint's encoder

A ranker maps (query, top_k) to recall ids, best first. Ties break by smaller recall id,
as in retrieval.search.rank, so all systems are deterministic.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np

from retrieval.search import SearchEngine, rank

Ranker = Callable[[str, int], list[int]]


def parse_checkpoint(spec: str) -> tuple[str, Path]:
    """'model_b=ml/artifacts/models/model-b-v1/checkpoints/best' -> (label, path)."""
    label, sep, path = spec.partition("=")
    if not sep or not label or not path:
        raise ValueError(f"checkpoint must be LABEL=PATH, got {spec!r}")
    if label in ("bm25", "minilm"):
        raise ValueError(f"checkpoint label {label!r} is reserved")
    return label, Path(path)


class CheckpointRanker:
    """Dense ranker for a fine-tuned MeanPoolEncoder checkpoint."""

    def __init__(self, directory: Path, engine: SearchEngine) -> None:
        from training.model import ENCODER_META, MeanPoolEncoder

        if not (directory / ENCODER_META).is_file():
            raise FileNotFoundError(f"{directory} is not a RecallIQ encoder checkpoint")
        self.directory = directory
        self.meta = json.loads((directory / ENCODER_META).read_text("utf-8"))
        self.encoder = MeanPoolEncoder.load(directory)
        self.recall_ids = engine.recall_ids
        self.matrix = self.encoder.encode([r.search_text for r in engine.corpus])

    def __call__(self, query: str, top_k: int) -> list[int]:
        scores = self.matrix @ self.encoder.encode([query])[0]
        return [self.recall_ids[i] for i in rank(np.asarray(scores), self.recall_ids, top_k)]


def _engine_ranker(engine: SearchEngine, model: str) -> Ranker:
    return lambda query, top_k: [h.recall_id for h in engine.search(query, model=model, top_k=top_k)]


def build_rankers(engine: SearchEngine, models: list[str],
                  checkpoints: list[tuple[str, Path]]) -> dict[str, Ranker]:
    rankers: dict[str, Ranker] = {model: _engine_ranker(engine, model) for model in models}
    for label, directory in checkpoints:
        rankers[label] = CheckpointRanker(directory, engine)
    return rankers
