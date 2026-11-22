import math

import pytest

from evaluation.metrics import evaluate_run, mrr_at_k, ndcg_at_k, recall_at_k


def test_recall_at_5_counts_all_relevant_documents():
    # 3 relevant docs, only #2 in the top 5 -> 1/3
    assert recall_at_k([1, 2, 3, 4, 5, 6], {2: 1, 6: 1, 9: 1}, 5) == pytest.approx(1 / 3)
    assert recall_at_k([2, 6, 9], {2: 1, 6: 1, 9: 1}, 5) == 1.0
    assert recall_at_k([], {1: 1}, 5) == 0.0


def test_mrr_uses_first_relevant_rank_within_cutoff():
    assert mrr_at_k([7, 8, 3, 4], {3: 1, 4: 1}, 10) == pytest.approx(1 / 3)
    assert mrr_at_k([7, 8, 9], {3: 1}, 10) == 0.0
    ranked = list(range(100, 110)) + [3]  # relevant doc at rank 11
    assert mrr_at_k(ranked, {3: 1}, 10) == 0.0


def test_ndcg_binary_hand_computed():
    # DCG = 0 + 1/log2(3) + 1/log2(4); IDCG = 1 + 1/log2(3)
    expected = (1 / math.log2(3) + 1 / math.log2(4)) / (1 + 1 / math.log2(3))
    assert ndcg_at_k([9, 1, 2], {1: 1, 2: 1}, 10) == pytest.approx(expected)
    assert ndcg_at_k([9, 1, 2], {1: 1, 2: 1}, 10) == pytest.approx(0.693426, abs=1e-6)


def test_ndcg_graded_hand_computed():
    # gains 2^g - 1: grade 1 -> 1, grade 3 -> 7
    expected = (1 + 7 / math.log2(3)) / (7 + 1 / math.log2(3))
    assert ndcg_at_k([10, 20], {10: 1, 20: 3}, 10) == pytest.approx(expected)
    assert ndcg_at_k([20, 10], {10: 1, 20: 3}, 10) == pytest.approx(1.0)


def test_ndcg_rewards_putting_higher_grades_first():
    graded = {1: 3, 2: 1}
    assert ndcg_at_k([1, 2], graded, 10) == pytest.approx(1.0)
    assert ndcg_at_k([2, 1], graded, 10) < ndcg_at_k([1, 2], graded, 10)


def test_recall_at_5_is_capped_when_more_than_5_relevant():
    relevant = {i: 1 for i in range(1, 9)}  # 8 relevant
    assert recall_at_k([1, 2, 3, 4, 5], relevant, 5) == pytest.approx(5 / 8)


def test_boolean_grades_are_rejected():
    with pytest.raises(ValueError, match="grades"):
        ndcg_at_k([1], {1: True}, 10)


def test_ndcg_ideal_is_truncated_at_k():
    relevant = {i: 1 for i in range(1, 16)}  # 15 relevant, k = 10
    assert ndcg_at_k(list(range(1, 11)), relevant, 10) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("ranked", "relevant", "k", "message"),
    [
        ([1], {1: 1}, 0, "k must"),
        ([1], {}, 5, "at least one relevant"),
        ([1, 1], {1: 1}, 5, "duplicate"),
        ([1], {1: 0}, 5, "grades"),
    ],
)
def test_invalid_metric_inputs(ranked, relevant, k, message):
    for fn in (recall_at_k, mrr_at_k, ndcg_at_k):
        with pytest.raises(ValueError, match=message):
            fn(ranked, relevant, k)


def test_evaluate_run_averages_per_query():
    result = evaluate_run({"q1": [1, 2], "q2": [5, 6]}, {"q1": {1: 1}, "q2": {9: 1}})
    assert result["num_queries"] == 2
    assert result["per_query"]["q1"]["mrr@10"] == 1.0
    assert result["per_query"]["q2"]["mrr@10"] == 0.0
    assert result["mean"]["mrr@10"] == pytest.approx(0.5)
    assert result["mean"]["recall@5"] == pytest.approx(0.5)


def test_evaluate_run_requires_a_ranking_per_judged_query():
    with pytest.raises(ValueError, match="No ranking"):
        evaluate_run({}, {"q1": {1: 1}})
