"""Shared search interface over the recall corpus.

model = "bm25" | "minilm" (pretrained, Model A) | "minilm_finetuned" (Model B) |
"minilm_hard_negative" (Model C). Dense models rank by cosine similarity against an
embedding index built by the same network; BM25 scores are never mixed with cosines.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from retrieval.bm25 import BM25Index
from retrieval.corpus import CorpusRecord
from retrieval.semantic import EmbeddingIndex, Encoder, ModelUnavailableError

ModelName = Literal["bm25", "minilm", "minilm_finetuned", "minilm_hard_negative"]
ScoreType = Literal["bm25", "cosine_similarity"]

DENSE_MODELS: tuple[str, ...] = ("minilm", "minilm_finetuned", "minilm_hard_negative")
SUPPORTED_MODELS: tuple[str, ...] = ("bm25", *DENSE_MODELS)
SCORE_TYPES: dict[str, ScoreType] = {"bm25": "bm25",
                                     **{m: "cosine_similarity" for m in DENSE_MODELS}}
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
    "minilm_finetuned": (
        "Cosine similarity from MiniLM fine-tuned on RecallIQ training recalls (Model B, "
        "in-batch negatives), range [-1, 1]. Not a calibrated probability; comparable only "
        "between results of this model for the same query."
    ),
    "minilm_hard_negative": (
        "Cosine similarity from MiniLM fine-tuned with hard negatives (Model C), range "
        "[-1, 1]. Not a calibrated probability; comparable only between results of this "
        "model for the same query."
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
        dense: dict[str, tuple[EmbeddingIndex | None, Encoder | None, str | None]] | None = None,
    ) -> None:
        """`embeddings`/`encoder`/`embeddings_error` configure "minilm" (unchanged API);
        `dense` adds other dense models: name -> (index, encoder, error_if_unavailable)."""
        if not corpus:
            raise ValueError("Corpus is empty")
        self.corpus = corpus
        self.recall_ids = [record.recall_id for record in corpus]
        self._by_id = {record.recall_id: record for record in corpus}
        self.bm25 = BM25Index([record.search_text for record in corpus])
        self.dense: dict[str, tuple[EmbeddingIndex | None, Encoder | None, str | None]] = {
            "minilm": (embeddings, encoder, embeddings_error)}
        for name, entry in (dense or {}).items():
            if name not in DENSE_MODELS or name == "minilm":
                raise ValueError(f"Unknown dense model {name!r}")
            self.dense[name] = entry
        for index, _, _ in self.dense.values():
            if index is not None:
                index.check_matches(corpus)
        self.embeddings = embeddings
        self.encoder = encoder
        self.embeddings_error = embeddings_error

    def record(self, recall_id: int) -> CorpusRecord:
        return self._by_id[recall_id]

    def available_models(self) -> dict[str, str | None]:
        """model -> None if servable, else the reason it is not."""
        out: dict[str, str | None] = {"bm25": None}
        for name in DENSE_MODELS:
            index, encoder, error = self.dense.get(name, (None, None, None))
            out[name] = None if index is not None and encoder is not None else (
                error or f"{name} is not configured")
        return out

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
            scores = self._cosine_scores(query, model)
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

    def _cosine_scores(self, query: str, model: str = "minilm") -> np.ndarray:
        index, encoder, error = self.dense.get(model, (None, None, None))
        if index is None or encoder is None:
            if model == "minilm":
                raise SearchUnavailableError(
                    error or "MiniLM embeddings are not available; run `python -m retrieval.embed`.")
            raise SearchUnavailableError(
                error or f"{model} is not available: no checkpoint/embedding index configured "
                "(build it with `python -m retrieval.finetuned`).")
        try:
            vector = encoder.encode([query])[0]
        except ModelUnavailableError as exc:
            raise SearchUnavailableError(str(exc)) from exc
        try:
            return index.cosine_scores(vector)
        except (ValueError, RuntimeError) as exc:  # dimension / zero-norm: server-side fault
            raise SearchUnavailableError(f"{model} index/model mismatch: {exc}") from exc
