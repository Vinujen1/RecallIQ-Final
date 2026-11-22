import json

import pytest

from evaluation.annotations import (
    Annotation,
    Judgment,
    main,
    read_any,
    read_csv,
    read_jsonl,
    validate,
    write_csv,
    write_jsonl,
)

SPLIT_OF = {101: "validation", 102: "validation", 201: "test", 301: "train"}
TITLES = {101: "Brand A cheddar cheese recalled due to Listeria",
          102: "Brand B yogurt recalled", 201: "Toy blocks", 301: "Phone charger"}


def _raw(**overrides):
    base = {"query_id": "AB-0001", "query": "is my cheese safe to eat", "split": "validation",
            "relevant": [{"recall_id": 101, "grade": 2}, {"recall_id": 102}],
            "reviewer_status": "reviewed", "annotator": "AB", "reviewer": "CD"}
    return {**base, **overrides}


def test_valid_annotation_is_accepted():
    report = validate([_raw()], SPLIT_OF, TITLES)
    assert report.ok, report.errors
    assert report.valid[0].qrels == {101: 2, 102: 1}  # missing grade defaults to 1
    assert [a.query_id for a in report.reviewed("validation")] == ["AB-0001"]


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"relevant": [{"recall_id": 101}, {"recall_id": 101, "grade": 2}]}, "duplicate relevance label"),
        ({"relevant": [{"recall_id": 999}]}, "not in the 1,000-record corpus"),
        ({"relevant": [{"recall_id": 201}]}, "anchor recall 201 belongs to the test split"),
        ({"relevant": [{"recall_id": 101}, {"recall_id": 201, "grade": 3}]},
         "anchor recall 201 belongs to the test split"),  # highest grade is the anchor
        ({"anchor_recall_id": 999}, "anchor_recall_id must be one of"),
        ({"relevant": [{"recall_id": 101, "grade": 5}]}, "grade"),
        ({"relevant": [{"recall_id": "abc"}]}, "not an integer"),
        ({"relevant": []}, "at least one relevant"),
        ({"query": "   "}, "empty"),
        ({"query": "x" * 501}, "longer than"),
        ({"split": "dev"}, "split must be"),
        ({"reviewer_status": "approved"}, "reviewer_status"),
        ({"reviewer": ""}, "need a reviewer"),
        ({"query_id": "bad id!"}, "query_id"),
    ],
)
def test_invalid_annotations_are_rejected(overrides, message):
    report = validate([_raw(**overrides)], SPLIT_OF, TITLES)
    assert not report.ok
    assert any(message in e for e in report.errors), report.errors
    assert report.valid == []


def test_duplicate_query_ids_rejected():
    report = validate([_raw(), _raw(query="different text")], SPLIT_OF, TITLES)
    assert any("duplicate query_id" in e for e in report.errors)


def test_drafts_are_valid_but_not_used_for_evaluation():
    report = validate([_raw(reviewer_status="draft", reviewer="")], SPLIT_OF, TITLES)
    assert report.ok
    assert report.reviewed() == []


@pytest.mark.parametrize("reviewer", ["AB", "ab", " a b "])
def test_self_review_is_an_error(reviewer):
    report = validate([_raw(reviewer=reviewer)], SPLIT_OF, TITLES)
    assert any("different person" in e for e in report.errors)


def test_warnings_for_title_copy_and_repeated_text():
    raws = [_raw(),
            _raw(query_id="AB-0002", query="Brand A cheddar cheese recalled due to Listeria",
                 relevant=[{"recall_id": 101}]),
            _raw(query_id="AB-0003", query="IS MY CHEESE SAFE TO EAT")]
    report = validate(raws, SPLIT_OF, TITLES)
    assert report.ok
    text = " ".join(report.warnings)
    assert "copies the title" in text and "same query text" in text


def test_split_fingerprint_must_match_current_manifest():
    stale = validate([_raw(split_fingerprint="old")], SPLIT_OF, TITLES, "current")
    assert any("different split assignment" in e for e in stale.errors)
    current = validate([_raw(split_fingerprint="current")], SPLIT_OF, TITLES, "current")
    assert current.ok and not current.warnings
    unstamped = validate([_raw()], SPLIT_OF, TITLES, "current")
    assert unstamped.ok and any("no split_fingerprint" in w for w in unstamped.warnings)
    assert unstamped.valid[0].split_fingerprint == "current"  # stamped on import


def test_relevant_recalls_from_other_splits_are_allowed_with_in_split_anchor():
    raw = _raw(relevant=[{"recall_id": 101}, {"recall_id": 201}, {"recall_id": 301}])
    report = validate([raw], SPLIT_OF, TITLES)
    assert report.ok, report.errors
    assert report.valid[0].anchor_recall_id == 101
    explicit = validate([_raw(relevant=[{"recall_id": 201}, {"recall_id": 102}],
                              anchor_recall_id=102)], SPLIT_OF, TITLES)
    assert explicit.ok and explicit.valid[0].anchor_recall_id == 102


def test_csv_import_groups_rows_per_query(tmp_path):
    path = tmp_path / "a.csv"
    path.write_text(
        "query_id,query,split,recall_id,grade,reviewer_status,annotator,reviewer\n"
        "AB-0001,is my cheese safe,validation,101,2,reviewed,AB,CD\n"
        "AB-0001,,,102,,,,\n"
        "AB-0002,toy recall,test,201,1,draft,AB,\n",
        encoding="utf-8",
    )
    report = validate(read_any(path), SPLIT_OF, TITLES)
    assert report.ok, report.errors
    first = report.valid[0]
    assert first.qrels == {101: 2, 102: 1} and first.reviewer_status == "reviewed"


def test_csv_duplicate_and_conflicting_rows_rejected(tmp_path):
    path = tmp_path / "dup.csv"
    path.write_text(
        "query_id,query,split,recall_id,grade,reviewer_status,annotator,reviewer\n"
        "AB-0001,is my cheese safe,validation,101,1,reviewed,AB,CD\n"
        "AB-0001,is my cheese safe,validation,101,2,reviewed,AB,CD\n"
        "AB-0002,toy,test,201,1,draft,AB,\n"
        "AB-0002,toy,validation,201,1,draft,AB,\n",
        encoding="utf-8",
    )
    report = validate(read_csv(path), SPLIT_OF, TITLES)
    assert any("duplicate relevance label for recall 101" in e for e in report.errors)
    assert any("split differs" in e for e in report.errors)


def test_csv_missing_columns_rejected(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("query_id,query\nq1,x\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing CSV columns"):
        read_csv(path)


def test_invalid_jsonl_line_reports_line_number(tmp_path):
    path = tmp_path / "a.jsonl"
    path.write_text(json.dumps(_raw()) + "\n{not json\n", encoding="utf-8")
    with pytest.raises(ValueError, match=":2:"):
        read_jsonl(path)


def test_jsonl_and_csv_round_trip(tmp_path):
    annotation = Annotation("AB-0001", "is my cheese safe", "validation",
                            [Judgment(101, 2), Judgment(102, 1)], "reviewed", "AB", "CD")
    write_jsonl([annotation], tmp_path / "a.jsonl")
    write_csv([annotation], tmp_path / "a.csv")
    for path in (tmp_path / "a.jsonl", tmp_path / "a.csv"):
        report = validate(read_any(path), SPLIT_OF, TITLES)
        assert report.ok and report.valid[0].qrels == {101: 2, 102: 1}


def test_import_cli_writes_only_when_valid(eval_env, tmp_path, capsys):
    split_of = {int(k): v for k, v in eval_env["manifest"]["assignments"].items()}
    val_id = next(i for i, s in split_of.items() if s == "validation")
    test_id = next(i for i, s in split_of.items() if s == "test")
    good, bad, out = tmp_path / "good.csv", tmp_path / "bad.csv", tmp_path / "queries.jsonl"
    header = "query_id,query,split,recall_id,grade,reviewer_status,annotator,reviewer\n"
    good.write_text(header + f"AB-0001,my query,validation,{val_id},1,reviewed,AB,CD\n", "utf-8")
    bad.write_text(header + f"AB-0001,my query,validation,{test_id},1,reviewed,AB,CD\n", "utf-8")
    common = ["--db", str(eval_env["db"]), "--manifest", str(eval_env["manifest_path"])]

    assert main(["import", str(bad), "--out", str(out), *common]) == 1
    assert not out.exists()
    assert main(["import", str(good), "--out", str(out), *common]) == 0
    assert len(read_jsonl(out)) == 1
    assert main(["validate", str(out), *common]) == 0
    assert main(["validate", str(tmp_path / "missing.jsonl"), *common]) == 2
    assert "Cannot read annotations" in capsys.readouterr().out


def test_header_only_template_validates_with_zero_queries(eval_env, capsys):
    from evaluation import REPO_ROOT

    template = REPO_ROOT / "data" / "annotations" / "template.csv"
    code = main(["validate", str(template), "--db", str(eval_env["db"]),
                 "--manifest", str(eval_env["manifest_path"])])
    assert code == 0
    assert "valid queries: 0" in capsys.readouterr().out
