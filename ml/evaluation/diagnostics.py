"""Proposal metrics beyond ranking quality, with explicit numerators and denominators.

False-match rate (FMR@1)
    numerator   = queries whose top-1 result is NOT a judged-relevant recall (an empty
                  ranking also counts: the user is shown no correct recall first)
    denominator = all evaluated queries
    Unjudged documents count as non-relevant (pooling), so FMR@1 is an upper bound when
    judgments are incomplete. FMR@1 = 1 - Precision@1.

Identifier-verification precision (IVP@k)
    For every query that names at least one identifier (keyword-tagged code or valid UPC,
    see verification.identifiers.has_identifier) and every result in its top k:
    numerator   = results the verifier marks CONFIRMED_MATCH that are judged relevant
    denominator = results the verifier marks CONFIRMED_MATCH
    Also reported: how many such queries exist and how many results were CONFIRMED_MISMATCH
    (and of those, how many were nevertheless judged relevant = wrongly rejected).
    When the denominator is 0 the value is None with a reason - never 0 or 1.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from verification.identifiers import Verdict, has_identifier, verify

FMR_DEFINITION = ("share of evaluated queries whose rank-1 result is not judged relevant "
                  "(unjudged = non-relevant); denominator = all evaluated queries")


def false_match_rate(run: Mapping[str, Sequence[int]], qrels: Mapping[str, Mapping[int, int]]) -> dict:
    qrels = {q: {d: g for d, g in rel.items() if g > 0} for q, rel in qrels.items()}  # grade 0 = not relevant
    misses = sorted(q for q in qrels if not run.get(q) or run[q][0] not in qrels[q])
    total = len(qrels)
    return {"value": len(misses) / total if total else None, "numerator": len(misses),
            "denominator": total, "false_match_query_ids": misses, "definition": FMR_DEFINITION}


def identifier_precision(run: Mapping[str, Sequence[int]], queries: Mapping[str, str],
                         qrels: Mapping[str, Mapping[int, int]], evidence: Mapping[int, str],
                         k: int = 5) -> dict:
    with_ids = sorted(q for q in qrels if has_identifier(queries[q]))
    confirmed = correct = mismatched = wrongly_rejected = 0
    for q in with_ids:
        for rid in run.get(q, [])[:k]:
            verdict = verify(queries[q], evidence.get(rid, "")).verdict
            if verdict is Verdict.MATCH:
                confirmed += 1
                correct += rid in qrels[q]
            elif verdict is Verdict.MISMATCH:
                mismatched += 1
                wrongly_rejected += rid in qrels[q]
    out = {"k": k, "queries_with_identifiers": len(with_ids), "confirmed_matches": confirmed,
           "confirmed_matches_relevant": correct, "confirmed_mismatches": mismatched,
           "confirmed_mismatches_relevant": wrongly_rejected}
    if confirmed == 0:
        out["value"] = None
        out["reason"] = ("no queries with identifiers" if not with_ids else
                         "no CONFIRMED_MATCH verdicts in the top-k results (the indexed recall "
                         "evidence rarely lists identifiers)")
    else:
        out["value"] = correct / confirmed
    return out


def error_rows(run: Mapping[str, Sequence[int]], queries: Mapping[str, str],
               qrels: Mapping[str, Mapping[int, int]], titles: Mapping[int, str],
               depth: int = 10) -> list[dict]:
    """Per query: first relevant rank (None = not in top `depth`) and the top-3 titles."""
    rows = []
    for q in sorted(qrels):
        ranked = list(run.get(q, []))[:depth]
        first = next((i for i, rid in enumerate(ranked, start=1) if rid in qrels[q]), None)
        rows.append({"query_id": q, "query": queries[q], "first_relevant_rank": first,
                     "relevant": [{"recall_id": r, "title": titles.get(r, "")} for r in qrels[q]],
                     "top3": [{"recall_id": r, "title": titles.get(r, ""),
                               "relevant": r in qrels[q]} for r in ranked[:3]]})
    return rows


def rank_changes(rows_a: list[dict], rows_b: list[dict]) -> dict:
    """Queries whose first relevant rank improved / worsened from system a to system b."""
    def rank(row):  # not found in top depth -> worse than any found rank
        return row["first_relevant_rank"] or 10**6
    by_q = {r["query_id"]: r for r in rows_a}
    better, worse = [], []
    for row in rows_b:
        before = by_q[row["query_id"]]
        if rank(row) < rank(before):
            better.append({"query_id": row["query_id"], "query": row["query"],
                           "from": before["first_relevant_rank"], "to": row["first_relevant_rank"]})
        elif rank(row) > rank(before):
            worse.append({"query_id": row["query_id"], "query": row["query"],
                          "from": before["first_relevant_rank"], "to": row["first_relevant_rank"]})
    return {"improved": better, "worsened": worse,
            "unchanged": len(rows_b) - len(better) - len(worse)}


def paired_bootstrap(per_query_a: Mapping[str, Mapping[str, float]],
                     per_query_b: Mapping[str, Mapping[str, float]], metric: str = "mrr@10",
                     resamples: int = 10_000, seed: int = 4110) -> dict:
    """Mean per-query difference (b - a) with a 95 % percentile bootstrap CI over queries.

    Same queries for both systems (paired). The CI reflects query sampling only, not
    training-seed variance; "ci_excludes_zero" is not a claim of practical importance."""
    import random

    queries = sorted(per_query_a)
    if sorted(per_query_b) != queries:
        raise ValueError("paired bootstrap needs the same queries for both systems")
    diffs = [per_query_b[q][metric] - per_query_a[q][metric] for q in queries]
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(resamples))
    low, high = means[int(0.025 * resamples)], means[int(0.975 * resamples) - 1]
    return {"metric": metric, "mean_difference": sum(diffs) / n, "ci95": [low, high],
            "ci_excludes_zero": low > 0 or high < 0, "better": sum(d > 0 for d in diffs),
            "worse": sum(d < 0 for d in diffs), "ties": sum(d == 0 for d in diffs),
            "num_queries": n, "resamples": resamples, "seed": seed}
