import json

import pytest

from conftest import FakeEncoder, build_fake_embeddings
from evaluation.annotate import (
    derive_split,
    next_query_id,
    parse_relevance,
    run_session,
)
from evaluation.annotations import Annotation, Judgment, read_jsonl, validate, write_jsonl
from evaluation.evaluate import EvaluationError, main as evaluate_main, run_evaluation
from evaluation.metrics import evaluate_run
from retrieval.loader import build_engine


@pytest.fixture
def ctx(eval_env):
    split_of = {int(k): v for k, v in eval_env["manifest"]["assignments"].items()}
    emb = build_fake_embeddings(eval_env["db"], eval_env["tmp"] / "emb")
    engine = build_engine(eval_env["db"], emb, encoder=FakeEncoder())
    titles = {r.recall_id: r.title for r in engine.corpus}
    return {**eval_env, "split_of": split_of, "engine": engine, "titles": titles, "emb": emb}


def _ids(ctx, split):
    return [i for i, s in sorted(ctx["split_of"].items()) if s == split]


def _write(ctx, annotations):
    path = ctx["tmp"] / "queries.jsonl"
    write_jsonl(annotations, path)
    return path


def test_evaluation_uses_only_reviewed_queries_in_split(ctx):
    val = _ids(ctx, "validation")
    path = _write(ctx, [
        Annotation("T-1", "Zq widget", "validation", [Judgment(val[0], 2)], "reviewed", "T", "R"),
        Annotation("T-2", "another query", "validation", [Judgment(val[1])], "draft", "T"),
    ])
    report = run_evaluation(path, "validation", ["bm25", "minilm"], ctx["engine"],
                            ctx["split_of"], ctx["titles"])
    assert report["num_queries"] == 1 and report["corpus_size"] == len(ctx["engine"].corpus)
    for model in ("bm25", "minilm"):
        result = report["results"][model]
        ranking = result["rankings"]["T-1"]
        expected = evaluate_run({"T-1": ranking}, {"T-1": {val[0]: 2}})
        assert result["mean"] == expected["mean"]  # metrics computed from the real ranking
    assert any("noise" in w for w in report["warnings"])


def test_cross_split_judgments_are_counted_and_broken_out(ctx):
    val, train = _ids(ctx, "validation"), _ids(ctx, "train")
    path = _write(ctx, [Annotation("T-1", "Zq widget", "validation",
                                   [Judgment(val[0], 2), Judgment(train[0])], "reviewed", "T", "R")])
    report = run_evaluation(path, "validation", ["bm25"], ctx["engine"], ctx["split_of"],
                            ctx["titles"])
    assert report["relevant_judgments_outside_query_split"] == 1
    result = report["results"]["bm25"]
    ranking = result["rankings"]["T-1"]
    assert result["mean"] == evaluate_run({"T-1": ranking}, {"T-1": {val[0]: 2, train[0]: 1}})["mean"]
    assert result["mean_in_split_judgments_only"] == evaluate_run(
        {"T-1": ranking}, {"T-1": {val[0]: 2}})["mean"]


def test_test_split_requires_explicit_confirmation(ctx):
    path = _write(ctx, [Annotation("T-1", "q", "test", [Judgment(_ids(ctx, "test")[0])],
                                   "reviewed", "T", "R")])
    with pytest.raises(EvaluationError, match="confirm-final-test"):
        run_evaluation(path, "test", ["bm25"], ctx["engine"], ctx["split_of"], ctx["titles"])
    report = run_evaluation(path, "test", ["bm25"], ctx["engine"], ctx["split_of"],
                            ctx["titles"], confirm_final_test=True)
    assert report["final_test"] is True


def test_train_split_is_not_evaluated(ctx):
    path = _write(ctx, [])
    with pytest.raises(EvaluationError, match="train"):
        run_evaluation(path, "train", ["bm25"], ctx["engine"], ctx["split_of"], ctx["titles"])


def test_no_reviewed_queries_means_no_metrics(ctx):
    path = _write(ctx, [Annotation("T-1", "q", "validation",
                                   [Judgment(_ids(ctx, "validation")[0])], "draft", "T")])
    with pytest.raises(EvaluationError, match="no metrics computed"):
        run_evaluation(path, "validation", ["bm25"], ctx["engine"], ctx["split_of"], ctx["titles"])


def test_invalid_annotations_block_evaluation(ctx):
    path = ctx["tmp"] / "bad.jsonl"
    path.write_text(json.dumps({"query_id": "T-1", "query": "q", "split": "validation",
                                "relevant": [{"recall_id": 123456}], "reviewer_status": "reviewed",
                                "annotator": "T", "reviewer": "R"}) + "\n", "utf-8")
    with pytest.raises(EvaluationError, match="annotation errors"):
        run_evaluation(path, "validation", ["bm25"], ctx["engine"], ctx["split_of"], ctx["titles"])


def test_evaluate_cli_writes_report_and_degrades_without_embeddings(ctx, capsys):
    path = _write(ctx, [Annotation("T-1", "Zq0 widget0", "validation",
                                   [Judgment(_ids(ctx, "validation")[0])], "reviewed", "T", "R")])
    out = ctx["tmp"] / "report.json"
    common = ["--annotations", str(path), "--db", str(ctx["db"]),
              "--manifest", str(ctx["manifest_path"]), "--out", str(out)]
    code = evaluate_main([*common, "--embeddings", str(ctx["tmp"] / "none")])
    assert code == 3  # incomplete comparison is not a success
    report = json.loads(out.read_text("utf-8"))
    assert "mean" in report["results"]["bm25"]
    assert "error" in report["results"]["minilm"]  # no embeddings -> reported, not faked
    assert report["provenance"]["split_seed"] == 4110
    assert report["provenance"]["split_assignment_fingerprint"]
    assert evaluate_main([*common, "--embeddings", str(ctx["tmp"] / "none"),
                          "--models", "bm25"]) == 0


def test_final_test_runs_are_logged(ctx, tmp_path):
    from evaluation.evaluate import _log_test_run

    log = tmp_path / "test_runs.jsonl"
    report = {"provenance": {"x": 1}, "num_queries": 3, "results": {"bm25": {"mean": {}}}}
    assert _log_test_run(report, log) == 0
    assert _log_test_run(report, log) == 1  # second run is detectable
    assert len(log.read_text("utf-8").splitlines()) == 2


def test_evaluate_cli_without_annotations_fails_cleanly(ctx, capsys):
    code = evaluate_main(["--annotations", str(ctx["tmp"] / "missing.jsonl"),
                          "--db", str(ctx["db"]), "--manifest", str(ctx["manifest_path"])])
    assert code == 2
    assert "Evaluation not run" in capsys.readouterr().out


# --- real corpus plumbing check (skipped until DB, manifest and embeddings exist) --------

def _real_ready() -> bool:
    from evaluation import DEFAULT_DB, DEFAULT_MANIFEST
    from retrieval.embed import DEFAULT_OUT

    return DEFAULT_DB.is_file() and DEFAULT_MANIFEST.is_file() and (DEFAULT_OUT / "metadata.json").is_file()


@pytest.mark.skipif(not _real_ready(), reason="real DB / split manifest / embeddings not built")
def test_metrics_run_on_real_corpus_with_reviewed_annotations(tmp_path):
    """PLUMBING CHECK ONLY: synthetic 'queries' (record titles) exercise the pipeline on the
    real 1,000-record index. These are not consumer queries and the numbers mean nothing."""
    from evaluation import DEFAULT_DB, DEFAULT_MANIFEST
    from evaluation.annotations import corpus_context
    from retrieval.embed import DEFAULT_OUT

    split_of, titles, manifest = corpus_context(DEFAULT_DB, DEFAULT_MANIFEST)
    val = [i for i, s in sorted(split_of.items()) if s == "validation"][:5]
    train = [i for i, s in sorted(split_of.items()) if s == "train"][:1]
    annotations = [
        Annotation(f"PLUMB-{n}", titles[rid], "validation",
                   [Judgment(rid, 2)] + ([Judgment(train[0])] if n == 0 else []),
                   "reviewed", "TEST", "FIXTURE", split_fingerprint=manifest["assignment_fingerprint"])
        for n, rid in enumerate(val)
    ]
    path = tmp_path / "plumbing.jsonl"
    write_jsonl(annotations, path)
    engine = build_engine(DEFAULT_DB, DEFAULT_OUT)
    report = run_evaluation(path, "validation", ["bm25", "minilm"], engine, split_of, titles,
                            assignment_fingerprint=manifest["assignment_fingerprint"])

    assert report["corpus_size"] == 1000 and report["num_queries"] == 5
    assert report["relevant_judgments_outside_query_split"] == 1
    for model in ("bm25", "minilm"):
        result = report["results"][model]
        assert "error" not in result
        recomputed = evaluate_run(result["rankings"], {a.query_id: a.qrels for a in annotations})
        assert result["mean"] == recomputed["mean"]
        assert all(0.0 <= v <= 1.0 for v in result["mean"].values())
        assert all(len(r) <= 10 for r in result["rankings"].values())


# --- annotation tool -------------------------------------------------------------------

def test_parse_relevance():
    judgments = parse_relevance("101:3 102, 103;104:2")
    assert [(j.recall_id, j.grade) for j in judgments] == [(101, 3), (102, 1), (103, 1), (104, 2)]
    for bad in ("", "abc", "101:9", "101 101"):
        with pytest.raises(ValueError):
            parse_relevance(bad)


def test_derive_split_uses_anchor_and_allows_cross_split_relevance():
    split_of = {1: "train", 2: "test"}
    assert derive_split([Judgment(1)], split_of) == ("train", 1)
    assert derive_split([Judgment(1), Judgment(2)], split_of) == ("train", 1)  # first listed
    assert derive_split([Judgment(1), Judgment(2, 3)], split_of) == ("test", 2)  # top grade
    with pytest.raises(ValueError, match="Not in the corpus"):
        derive_split([Judgment(3)], split_of)


def test_next_query_id():
    assert next_query_id({"AB-0001", "AB-0002"}, "AB") == "AB-0003"
    assert next_query_id(set(), "CD") == "CD-0001"


def test_interactive_session_saves_valid_draft(ctx):
    val, test = _ids(ctx, "validation"), _ids(ctx, "test")
    answers = iter([
        "widget for kids",          # query
        "/find widget",             # extra search
        f"/show {val[0]}",          # read a record
        "not-an-id",                # rejected input
        f"{test[0]} {test[0]}",     # duplicate -> rejected
        f"{val[0]}:2 {test[0]}",    # accepted: anchor = val[0] (grade 2), test doc also relevant
        "",                         # finish
    ])
    output: list[str] = []
    out = ctx["tmp"] / "session.jsonl"
    fingerprint = ctx["manifest"]["assignment_fingerprint"]
    saved = run_session(ctx["engine"], ctx["split_of"], out, "AB", split_fingerprint=fingerprint,
                        read=lambda _: next(answers), write=output.append)
    assert saved == 1
    text = "\n".join(output)
    assert "Cannot read" in text and "listed twice" in text
    candidate_lines = [line for line in output if line.startswith("  #")]
    assert candidate_lines and not any("[validation" in l or "[test" in l or "[train" in l
                                       for l in candidate_lines)  # splits hidden while judging
    raws = read_jsonl(out)
    assert raws[0]["query_id"] == "AB-0001" and raws[0]["reviewer_status"] == "draft"
    assert raws[0]["split"] == "validation" and raws[0]["anchor_recall_id"] == val[0]
    assert raws[0]["split_fingerprint"] == fingerprint
    report = validate(raws, ctx["split_of"], ctx["titles"], fingerprint)
    assert report.ok and not report.warnings
    assert report.valid[0].qrels == {val[0]: 2, test[0]: 1}
