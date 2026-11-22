"""Ranking metrics with multiple (optionally graded) relevant documents per query.

Definitions (k = cutoff, R = set of relevant ids, grades g >= 1):
  Recall@k = |R ∩ top_k| / |R|                     (fraction of ALL relevant docs found)
  MRR@k    = 1 / rank of the first relevant doc in top_k, else 0
  nDCG@k   = DCG@k / IDCG@k,  DCG = sum_i (2^g_i - 1) / log2(i + 1)  over ranks i = 1..k
             IDCG = DCG of the ideal ordering of all judged relevant docs (top k)
Unjudged documents count as non-relevant. Binary judgments use g = 1.
Note: with |R| > k, Recall@k cannot reach 1 (max k/|R|); compare per-query values with care.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from statistics import mean


def _check(ranked: Sequence[int], relevant: Mapping[int, int], k: int) -> None:
    if k < 1:
        raise ValueError("k must be >= 1")
    if not relevant:
        raise ValueError("A query needs at least one relevant document")
    if len(set(ranked)) != len(ranked):
        raise ValueError("Ranked list contains duplicate document ids")
    if any(isinstance(g, bool) or not isinstance(g, int) or g < 1 for g in relevant.values()):
        raise ValueError("Relevance grades must be integers >= 1")


def recall_at_k(ranked: Sequence[int], relevant: Mapping[int, int], k: int = 5) -> float:
    _check(ranked, relevant, k)
    found = sum(1 for doc in ranked[:k] if doc in relevant)
    return found / len(relevant)


def mrr_at_k(ranked: Sequence[int], relevant: Mapping[int, int], k: int = 10) -> float:
    _check(ranked, relevant, k)
    for position, doc in enumerate(ranked[:k], start=1):
        if doc in relevant:
            return 1.0 / position
    return 0.0


def _dcg(grades: Sequence[int]) -> float:
    return sum((2**g - 1) / math.log2(i + 1) for i, g in enumerate(grades, start=1))


def ndcg_at_k(ranked: Sequence[int], relevant: Mapping[int, int], k: int = 10) -> float:
    _check(ranked, relevant, k)
    gains = [relevant.get(doc, 0) for doc in ranked[:k]]
    ideal = sorted(relevant.values(), reverse=True)[:k]
    return _dcg(gains) / _dcg(ideal)


METRICS = {
    "recall@5": lambda ranked, rel: recall_at_k(ranked, rel, 5),
    "mrr@10": lambda ranked, rel: mrr_at_k(ranked, rel, 10),
    "ndcg@10": lambda ranked, rel: ndcg_at_k(ranked, rel, 10),
}


def evaluate_run(
    run: Mapping[str, Sequence[int]], qrels: Mapping[str, Mapping[int, int]]
) -> dict:
    """Per-query and mean metrics. Every judged query must have a ranking (maybe empty)."""
    missing = sorted(set(qrels) - set(run))
    if missing:
        raise ValueError(f"No ranking for queries: {missing[:5]}")
    per_query = {
        qid: {name: fn(run[qid], qrels[qid]) for name, fn in METRICS.items()}
        for qid in sorted(qrels)
    }
    means = {
        name: (mean(q[name] for q in per_query.values()) if per_query else None)
        for name in METRICS
    }
    return {"num_queries": len(per_query), "mean": means, "per_query": per_query}
