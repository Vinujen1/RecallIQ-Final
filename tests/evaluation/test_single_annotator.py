"""Single-annotator mode: one person writes AND finalizes queries, with no second reviewer.

These judgments are opt-in for evaluation, never reported as independently reviewed, and
always broken out in the report. Queries/labels below are synthetic test fixtures.
"""

import json

import pytest

from conftest import FakeEncoder, build_fake_embeddings
from evaluation.annotations import (
    Annotation,
    Judgment,
    finalize_single_annotator,
    main as annotations_main,
    read_jsonl,
    validate,
    write_jsonl,
)
from evaluation.evaluate import EvaluationError, main as evaluate_main, run_evaluation
from evaluation.metrics import evaluate_run
from retrieval.loader import build_engine

SPLIT_OF = {101: "validation", 102: "validation", 201: "test"}


def _raw(**overrides):
    base = {"query_id": "VD-0001", "query": "is my cheese safe", "split": "validation",
            "relevant": [{"recall_id": 101, "grade": 3}], "reviewer_status": "single_annotator",
            "annotator": "VD", "reviewer": ""}
    return {**base, **overrides}


# --- validation ------------------------------------------------------------------------

def test_single_annotator_status_is_valid_but_not_counted_as_reviewed():
    report = validate([_raw()], SPLIT_OF)
    assert report.ok, report.errors
    assert report.reviewed() == []  # never counted as independently reviewed
    assert report.evaluable("validation") == []  # excluded by default
    assert [a.query_id for a in report.evaluable("validation", include_single_annotator=True)] == ["VD-0001"]


@pytest.mark.parametrize(("overrides", "message"), [
    ({"reviewer": "VD"}, "must leave reviewer empty"),      # self-"review" is not a review
    ({"reviewer": "CD"}, "must leave reviewer empty"),      # a real second check is `reviewed`
    ({"annotator": ""}, "need the annotator's initials"),
])
def test_single_annotator_rules(overrides, message):
    report = validate([_raw(**overrides)], SPLIT_OF)
    assert any(message in e for e in report.errors), report.errors


@pytest.mark.parametrize("reviewer", ["VD", "vd", " V D "])
def test_self_review_cannot_be_labelled_independent(reviewer):
    report = validate([_raw(reviewer_status="reviewed", reviewer=reviewer)], SPLIT_OF)
    assert any("different person" in e for e in report.errors)
    assert report.reviewed() == []


def test_existing_statuses_unchanged():
    raws = [_raw(query_id="A-1", reviewer_status="reviewed", annotator="AB", reviewer="CD"),
            _raw(query_id="A-2", query="q2", reviewer_status="draft", annotator="AB"),
            _raw(query_id="A-3", query="q3", reviewer_status="rejected", annotator="AB")]
    report = validate(raws, SPLIT_OF)
    assert report.ok, report.errors
    assert [a.query_id for a in report.reviewed("validation")] == ["A-1"]
    assert [a.query_id for a in report.evaluable("validation", True)] == ["A-1"]  # drafts/rejected stay out


# --- finalize --------------------------------------------------------------------------

def test_finalize_marks_only_own_drafts_and_never_sets_a_reviewer():
    raws = [_raw(query_id="VD-0001", reviewer_status="draft"),
            _raw(query_id="VD-0002", query="q2", reviewer_status="rejected"),
            _raw(query_id="AB-0001", query="q3", reviewer_status="draft", annotator="AB")]
    assert finalize_single_annotator(raws, "vd") == ["VD-0001"]
    assert raws[0]["reviewer_status"] == "single_annotator" and raws[0]["reviewer"] == ""
    assert raws[1]["reviewer_status"] == "rejected"   # non-drafts untouched
    assert raws[2]["reviewer_status"] == "draft"      # someone else's draft untouched
    assert validate(raws, SPLIT_OF).ok


def test_finalize_with_ids_refuses_bad_requests_without_changing_anything():
    raws = [_raw(query_id="VD-0001", reviewer_status="draft"),
            _raw(query_id="AB-0001", query="q2", reviewer_status="draft", annotator="AB")]
    for ids, message in (({"VD-9999"}, "not in the file"), ({"AB-0001"}, "written by AB"),
                         ({"VD-0001", "AB-0001"}, "written by AB")):
        with pytest.raises(ValueError, match=message):
            finalize_single_annotator(raws, "VD", ids)
        assert [r["reviewer_status"] for r in raws] == ["draft", "draft"]
    assert finalize_single_annotator(raws, "VD", {"VD-0001"}) == ["VD-0001"]
    with pytest.raises(ValueError, match="not draft"):
        finalize_single_annotator(raws, "VD", {"VD-0001"})


def test_finalize_cli_rewrites_file_preserving_other_fields(eval_env, capsys):
    split_of = {int(k): v for k, v in eval_env["manifest"]["assignments"].items()}
    val = [i for i, s in sorted(split_of.items()) if s == "validation"]
    path = eval_env["tmp"] / "vd.jsonl"
    fp = eval_env["manifest"]["assignment_fingerprint"]
    write_jsonl([Annotation("VD-0001", "first query", "validation", [Judgment(val[0], 3)],
                            "draft", "VD", notes="kept", created_at="2026-11-17T00:00:00+00:00",
                            split_fingerprint=fp)], path)
    before = read_jsonl(path)[0]
    common = ["--db", str(eval_env["db"]), "--manifest", str(eval_env["manifest_path"])]
    assert annotations_main(["finalize", str(path), "--annotator", "VD", *common]) == 0
    after = read_jsonl(path)[0]
    assert after == {**before, "reviewer_status": "single_annotator", "reviewer": ""}
    assert "NOT independently reviewed" in capsys.readouterr().out
    assert annotations_main(["finalize", str(path), "--annotator", "VD", *common]) == 0  # no-op
    assert annotations_main(["finalize", str(path), "--annotator", "VD", "--ids", "VD-0002",
                             *common]) == 1


# --- evaluation ------------------------------------------------------------------------

@pytest.fixture
def ctx(eval_env):
    split_of = {int(k): v for k, v in eval_env["manifest"]["assignments"].items()}
    emb = build_fake_embeddings(eval_env["db"], eval_env["tmp"] / "emb")
    engine = build_engine(eval_env["db"], emb, encoder=FakeEncoder())
    titles = {r.recall_id: r.title for r in engine.corpus}
    ids = lambda split: [i for i, s in sorted(split_of.items()) if s == split]  # noqa: E731
    return {**eval_env, "split_of": split_of, "engine": engine, "titles": titles, "ids": ids}


def _write(ctx, annotations):
    path = ctx["tmp"] / "queries.jsonl"
    write_jsonl(annotations, path)
    return path


def _run(ctx, path, split="validation", **kwargs):
    return run_evaluation(path, split, ["bm25"], ctx["engine"], ctx["split_of"], ctx["titles"],
                          **kwargs)


def _mixed(ctx):
    val = ctx["ids"]("validation")
    return _write(ctx, [
        Annotation("AB-1", "Zq widget", "validation", [Judgment(val[0], 2)], "reviewed", "AB", "CD"),
        Annotation("VD-1", "Zq1 widget1", "validation", [Judgment(val[1], 3)], "single_annotator", "VD"),
        Annotation("VD-2", "draft query", "validation", [Judgment(val[2])], "draft", "VD"),
    ])


def test_single_annotator_excluded_by_default(ctx):
    report = _run(ctx, _mixed(ctx))
    comp = report["annotation_composition"]
    assert report["num_queries"] == 1 and set(report["results"]["bm25"]["rankings"]) == {"AB-1"}
    assert comp["review_mode"] == "independent" and comp["single_annotator_queries"] == 0
    assert comp["excluded_in_split"] == {"draft": 1, "single_annotator": 1}
    assert not any("LIMITATION" in w for w in report["warnings"])


def test_mixed_report_distinguishes_review_modes(ctx):
    report = _run(ctx, _mixed(ctx), include_single_annotator=True)
    comp = report["annotation_composition"]
    assert report["num_queries"] == 2  # draft still excluded
    assert comp["review_mode"] == "mixed"
    assert (comp["independently_reviewed_queries"], comp["single_annotator_queries"],
            comp["total_queries_evaluated"]) == (1, 1, 2)
    assert comp["queries_by_status"] == {"reviewed": ["AB-1"], "single_annotator": ["VD-1"]}
    assert comp["reviewers"] == ["CD"] and comp["annotators"] == ["AB", "VD"]
    assert comp["excluded_in_split"] == {"draft": 1}
    assert any("LIMITATION" in w and "1 of 2" in w for w in report["warnings"])
    result = report["results"]["bm25"]
    by_status = result["mean_by_review_status"]
    assert by_status["single_annotator"]["num_queries"] == 1
    val = ctx["ids"]("validation")
    assert by_status["single_annotator"]["mean"] == evaluate_run(
        {"VD-1": result["rankings"]["VD-1"]}, {"VD-1": {val[1]: 3}})["mean"]


def test_single_annotator_only_evaluation(ctx):
    val = ctx["ids"]("validation")
    path = _write(ctx, [Annotation("VD-1", "Zq widget", "validation", [Judgment(val[0], 3)],
                                   "single_annotator", "VD")])
    with pytest.raises(EvaluationError, match="--include-single-annotator"):
        _run(ctx, path)  # opt-in only; error tells the user why nothing ran
    report = _run(ctx, path, include_single_annotator=True)
    comp = report["annotation_composition"]
    assert comp["review_mode"] == "single_annotator" and comp["independently_reviewed_queries"] == 0
    assert comp["reviewers"] == []
    assert any("LIMITATION" in w for w in report["warnings"])


def test_test_split_still_sealed_with_single_annotator(ctx):
    path = _write(ctx, [Annotation("VD-1", "q", "test", [Judgment(ctx["ids"]("test")[0])],
                                   "single_annotator", "VD")])
    with pytest.raises(EvaluationError, match="confirm-final-test"):
        _run(ctx, path, "test", include_single_annotator=True)
    report = _run(ctx, path, "test", include_single_annotator=True, confirm_final_test=True)
    assert report["final_test"] is True and report["annotation_composition"]["single_annotator_queries"] == 1


def test_train_split_never_evaluated(ctx):
    with pytest.raises(EvaluationError, match="train"):
        _run(ctx, _write(ctx, []), "train", include_single_annotator=True)


def test_cli_flag_and_report_composition(ctx, capsys):
    path = _mixed(ctx)
    out = ctx["tmp"] / "report.json"
    common = ["--annotations", str(path), "--db", str(ctx["db"]), "--manifest",
              str(ctx["manifest_path"]), "--out", str(out), "--models", "bm25",
              "--embeddings", str(ctx["tmp"] / "none")]
    assert evaluate_main(common) == 0
    assert json.loads(out.read_text("utf-8"))["annotation_composition"]["single_annotator_queries"] == 0
    assert evaluate_main([*common, "--include-single-annotator"]) == 0
    report = json.loads(out.read_text("utf-8"))
    assert report["annotation_composition"]["review_mode"] == "mixed"
    printed = capsys.readouterr().out
    assert "single-annotator=1" in printed and "LIMITATION" in printed
