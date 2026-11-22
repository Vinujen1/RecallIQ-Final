"""False-match rate, identifier-verification precision, error rows (hand-computed cases)."""

import pytest

from evaluation.diagnostics import error_rows, false_match_rate, identifier_precision, rank_changes
from evaluation.rankers import parse_checkpoint

QRELS = {"q1": {10: 1}, "q2": {20: 2, 21: 1}, "q3": {30: 1}, "q4": {40: 1}}
RUN = {"q1": [10, 11], "q2": [99, 21], "q3": [], "q4": [40]}


def test_false_match_rate_counts_wrong_and_empty_top1():
    fmr = false_match_rate(RUN, QRELS)
    assert fmr["numerator"] == 2 and fmr["denominator"] == 4 and fmr["value"] == 0.5
    assert fmr["false_match_query_ids"] == ["q2", "q3"]


def test_identifier_precision_none_when_no_identifier_queries():
    texts = {q: "is my cheese safe" for q in QRELS}
    ivp = identifier_precision(RUN, texts, QRELS, {})
    assert ivp["value"] is None and ivp["queries_with_identifiers"] == 0
    assert "no queries with identifiers" in ivp["reason"]


def test_identifier_precision_hand_computed():
    qrels = {"a": {1: 1}, "b": {3: 1}}
    run = {"a": [1, 2], "b": [4, 3]}
    texts = {"a": "heater model XJ-500", "b": "charger model HC45"}
    evidence = {1: "heater model XJ-500", 2: "heater model XJ-500 (other recall)",
                3: "charger model HC-45", 4: "charger model HC46"}
    ivp = identifier_precision(run, texts, qrels, evidence, k=5)
    # confirmed matches: a->1 (relevant), a->2 (not relevant), b->3 (relevant) => 2/3
    assert ivp["confirmed_matches"] == 3 and ivp["confirmed_matches_relevant"] == 2
    assert ivp["value"] == pytest.approx(2 / 3)
    assert ivp["confirmed_mismatches"] == 1 and ivp["confirmed_mismatches_relevant"] == 0


def test_identifier_precision_none_when_no_confirmed_match():
    ivp = identifier_precision({"a": [1]}, {"a": "model XJ-500"}, {"a": {1: 1}}, {1: "a heater"})
    assert ivp["value"] is None and ivp["queries_with_identifiers"] == 1


def test_error_rows_and_rank_changes():
    titles = {10: "A", 11: "B", 20: "C", 21: "D", 99: "Z", 30: "E", 40: "F"}
    texts = {q: q for q in QRELS}
    rows = error_rows(RUN, texts, QRELS, titles)
    by_q = {r["query_id"]: r for r in rows}
    assert by_q["q1"]["first_relevant_rank"] == 1 and by_q["q2"]["first_relevant_rank"] == 2
    assert by_q["q3"]["first_relevant_rank"] is None
    better = error_rows({**RUN, "q2": [20], "q4": [41, 40]}, texts, QRELS, titles)
    change = rank_changes(rows, better)
    assert [c["query_id"] for c in change["improved"]] == ["q2"]
    assert [c["query_id"] for c in change["worsened"]] == ["q4"]
    assert change["unchanged"] == 2


def test_parse_checkpoint():
    assert parse_checkpoint("model_b=a/b")[0] == "model_b"
    for bad in ("model_b", "=x", "bm25=x"):
        with pytest.raises(ValueError):
            parse_checkpoint(bad)


def test_paired_bootstrap_counts_and_ci():
    from evaluation.diagnostics import paired_bootstrap

    a = {f"q{i}": {"mrr@10": 0.5} for i in range(20)}
    b = {f"q{i}": {"mrr@10": 1.0 if i < 15 else 0.5} for i in range(20)}
    out = paired_bootstrap(a, b, resamples=2000)
    assert out["better"] == 15 and out["worse"] == 0 and out["ties"] == 5
    assert out["mean_difference"] == pytest.approx(0.375)
    assert out["ci_excludes_zero"] and out["ci95"][0] <= 0.375 <= out["ci95"][1]
    assert paired_bootstrap(a, a, resamples=200)["ci_excludes_zero"] is False
    with pytest.raises(ValueError):
        paired_bootstrap(a, {"x": {"mrr@10": 1.0}})
