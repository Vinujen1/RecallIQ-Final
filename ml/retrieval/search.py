"""Shared search interface over the recall corpus: model = "bm25" | "minilm"."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from retrieval.bm25 import BM25Index
from retrieval.corpus import CorpusRecord
from retrieval.semantic import EmbeddingIndex, Encoder, ModelUnavailableError

ModelName = Literal["bm25", "minilm"]
ScoreType = Literal["bm25", "cosine_similarity"]

SUPPORTED_MODELS: tuple[str, ...] = ("bm25", "minilm")
SCORE_TYPES: dict[str, ScoreType] = {"bm25": "bm25", "minilm": "cosine_similarity"}
SCORE_DESCRIPTIONS: dict[str, str] = {
    "bm25": (
        "Okapi BM25 lexical score: unbounded, >= 0, higher = more query-term overlap. "
        "Only comparable between results of the same query; not comparable with cosine "
        "similarity and not a probability of relevance."
    ),
    "minilm": (
        "Cosine similarity between the query and recall embeddings from pretrained "
        "all-MiniLM-L6-v2, range [-1, 1]. Not a calibrated relevance probability and not "
        "comparable with BM25 scores."
    ),
}
MAX_QUERY_LENGTH = 500
MAX_TOP_K = 50


class SearchUnavailableError(RuntimeError):
    """The requested model cannot be served (e.g. embeddings not built)."""


@dataclass(frozen=True)
class SearchHit:
    rank: int
    recall_id: int
    title: str
    category: str | None
    issue: str | None
    url: str
    score: float


def validate_query(query: str) -> str:
    if not isinstance(query, str):
        raise ValueError("Query must be a string")
    cleaned = " ".join(query.split())
    if not cleaned:
        raise ValueError("Query must not be empty")
    if len(cleaned) > MAX_QUERY_LENGTH:
        raise ValueError(f"Query must be at most {MAX_QUERY_LENGTH} characters")
    return cleaned


def rank(scores: np.ndarray, recall_ids: list[int], top_k: int) -> list[int]:
    """Row indices sorted by score desc, ties broken by smaller recall id (deterministic)."""
    order = np.lexsort((np.asarray(recall_ids), -scores))
    return order[:top_k].tolist()


class SearchEngine:
    def __init__(
        self,
        corpus: list[CorpusRecord],
        embeddings: EmbeddingIndex | None = None,
        encoder: Encoder | None = None,
        embeddings_error: str | None = None,
    ) -> None:
        if not corpus:
            raise ValueError("Corpus is empty")
        self.corpus = corpus
        self.recall_ids = [record.recall_id for record in corpus]
        self.bm25 = BM25Index([record.search_text for record in corpus])
        if embeddings is not None:
            embeddings.check_matches(corpus)
        self.embeddings = embeddings
        self.encoder = encoder
        self.embeddings_error = embeddings_error

    def search(self, query: str, model: str = "bm25", top_k: int = 10) -> list[SearchHit]:
        query = validate_query(query)
        if model not in SUPPORTED_MODELS:
            raise ValueError(f"Unknown model {model!r}; expected one of {SUPPORTED_MODELS}")
        if not 1 <= top_k <= MAX_TOP_K:
            raise ValueError(f"top_k must be between 1 and {MAX_TOP_K}")

        if model == "bm25":
            scores = self.bm25.scores(query)
            candidates = [i for i in rank(scores, self.recall_ids, top_k) if scores[i] > 0.0]
        else:
            scores = self._cosine_scores(query)
            candidates = rank(scores, self.recall_ids, top_k)

        return [
            SearchHit(
                rank=position,
                recall_id=self.corpus[i].recall_id,
                title=self.corpus[i].title,
                category=self.corpus[i].category,
                issue=self.corpus[i].issue,
                url=self.corpus[i].url,
                score=float(scores[i]),
            )
            for position, i in enumerate(candidates, start=1)
        ]

    def _cosine_scores(self, query: str) -> np.ndarray:
        if self.embeddings is None or self.encoder is None:
            raise SearchUnavailableError(
                self.embeddings_error
                or "MiniLM embeddings are not available; run `python -m retrieval.embed`."
            )
        try:
            vector = self.encoder.encode([query])[0]
        except ModelUnavailableError as error:
            raise SearchUnavailableError(str(error)) from error
        try:
            return self.embeddings.cosine_scores(vector)
        except ValueError as error:  # dimension / zero-norm mismatch is a server-side fault
            raise SearchUnavailableError(f"MiniLM index/model mismatch: {error}") from error
