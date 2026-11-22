import json

import pytest

from evaluation.split_review import (
    COLUMNS,
    apply,
    collect_pairs,
    effective_links,
    export,
    main,
    pair_id,
    read_review,
    validate_review,
    write_review,
)
from evaluation.splits import (
    MANUAL_LINKS,
    assignment_fingerprint,
    load_manifest,
    main as splits_main,
    split_lookup,
    write_manifest,
)
from retrieval.corpus import corpus_fingerprint, load_corpus

PAIR = "P-9005-9006"  # synthetic cheddar pair: Jaccard 0.5, different brands (ambiguous)


@pytest.fixture
def review_env(eval_env, tmp_path):
    """Manifest where the ambiguous cheddar pair is forced into different splits."""
    manifest = eval_env["manifest"]
    manifest["assignments"]["9005"], manifest["assignments"]["9006"] = "train", "test"
    for p in manifest["ambiguous_pairs"]:
        if sorted(p["recall_ids"]) == [9005, 9006]:
            p["splits"] = ["train", "test"]
    manifest["assignment_fingerprint"] = assignment_fingerprint(split_lookup(manifest))
    write_manifest(manifest, eval_env["manifest_path"])
    proposals = tmp_path / "proposals.json"
    proposals.write_text(json.dumps({"proposals": {"9005-9006": {
        "decision": "DISTINCT", "flag": "check_source", "rationale": "AI text from titles"}}}),
        "utf-8")
    review = tmp_path / "review.csv"
    export(manifest, eval_env["db"], review, None, proposals)
    fingerprint = corpus_fingerprint(load_corpus(eval_env["db"]))
    return {**eval_env, "review": review, "fingerprint": fingerprint, "proposals": proposals,
            "annotations": tmp_path / "annotations",
            "ids": {r.recall_id for r in load_corpus(eval_env["db"])}}


def _OPTS(env):
    return {"require_semantic_audit": False, "annotations_dir": env["annotations"],
            "proposals_path": env["proposals"]}


def _decide(env, decision, reviewer="Jane Doe", rationale="Checked both recall pages myself.",
            reviewed_at="2026-11-16", fingerprint=None):
    rows = read_review(env["review"])
    for row in rows:
        if row["pair_id"] == PAIR:
            row.update(decision=decision, reviewer=reviewer, rationale=rationale,
                       reviewed_at=reviewed_at, corpus_fingerprint=fingerprint or env["fingerprint"])
    write_review(rows, env["review"])
    return rows


def test_export_contains_evidence_and_no_decisions(review_env):
    rows = read_review(review_env["review"])
    row = next(r for r in rows if r["pair_id"] == PAIR)
    assert {row["split_a"], row["split_b"]} == {"train", "test"}
    assert row["title_a"] and row["url_a"].startswith("https://") and row["category_a"]
    assert float(row["jaccard"]) == pytest.approx(0.5)
    assert row["assistant_proposal"] == "DISTINCT" and row["decision"] == ""
    assert set(COLUMNS) <= set(row)
    meta = json.loads(review_env["review"].with_suffix(".meta.json").read_text("utf-8"))
    assert meta["corpus_fingerprint"] == review_env["fingerprint"] and meta["decided"] == 0
    assert "NOT human verified" in meta["assistant_proposals"]["status"]


def test_manual_links_outside_corpus_are_skipped(review_env):
    pairs = collect_pairs(review_env["manifest"])
    assert PAIR in pairs
    assert not any(k == pair_id(a, b) for a, b, _ in MANUAL_LINKS for k in pairs)


def test_reexport_preserves_human_decisions(review_env):
    _decide(review_env, "DISTINCT")
    export(review_env["manifest"], review_env["db"], review_env["review"], None, None)
    row = next(r for r in read_review(review_env["review"]) if r["pair_id"] == PAIR)
    assert row["decision"] == "DISTINCT" and row["reviewer"] == "Jane Doe"


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"reviewer": "Claude"}, "human team member"),
        ({"reviewer": "AI assistant"}, "human team member"),
        ({"reviewer": ""}, "reviewer is required"),
        ({"rationale": "ok"}, "rationale of at least"),
        ({"rationale": "AI text from titles"}, "copies the assistant"),
        ({"reviewed_at": "yesterday"}, "reviewed_at"),
        ({"fingerprint": "stale"}, "corpus_fingerprint"),
        ({"decision": "MAYBE"}, "decision must be one of"),
    ],
)
def test_invalid_human_decisions_are_rejected(review_env, overrides, message):
    decision = overrides.pop("decision", "DISTINCT")
    rows = _decide(review_env, decision, **overrides)
    _, errors, _ = validate_review(rows, review_env["ids"], review_env["fingerprint"], False)
    assert any(message in e for e in errors), errors


def test_conflicting_duplicate_rows_rejected(review_env):
    rows = _decide(review_env, "DISTINCT")
    duplicate = dict(next(r for r in rows if r["pair_id"] == PAIR), decision="SAME_PRODUCT")
    _, errors, _ = validate_review(rows + [duplicate], review_env["ids"],
                                   review_env["fingerprint"], False)
    assert any("conflicting" in e for e in errors)


def test_apply_refuses_without_complete_human_review(review_env):
    with pytest.raises(ValueError, match="no decision yet"):
        apply(review_env["review"], review_env["db"], review_env["manifest_path"], None,
              **_OPTS(review_env))
    assert load_manifest(review_env["manifest_path"]).get("frozen") is False


def test_same_product_is_colocated_and_manifest_frozen(review_env, capsys):
    _decide(review_env, "SAME_PRODUCT")
    manifest, messages = apply(review_env["review"], review_env["db"],
                               review_env["manifest_path"], None, **_OPTS(review_env))
    assignments = manifest["assignments"]
    assert assignments["9005"] == assignments["9006"]
    assert manifest["frozen"] is True and "FROZEN" in " ".join(messages)
    assert manifest["review"]["reviewers"] == ["Jane Doe"]
    assert manifest["review"]["decisions"] == {"SAME_PRODUCT": 1}
    assert manifest["checks"]["unreviewed_crossing_pairs"] == 0

    # frozen manifests cannot be overwritten by the plain splits command, even with --force
    common = ["--db", str(review_env["db"]), "--out", str(review_env["manifest_path"]),
              "--embeddings", str(review_env["tmp"] / "none")]
    assert splits_main([*common, "--force"]) == 1
    assert "FROZEN" in capsys.readouterr().out
    with pytest.raises(RuntimeError, match="already frozen"):
        apply(review_env["review"], review_env["db"], review_env["manifest_path"], None,
              **_OPTS(review_env))
    again, messages = apply(review_env["review"], review_env["db"], review_env["manifest_path"],
                            None, refreeze=True, **_OPTS(review_env))
    assert again["frozen"] is True
    assert any("backed up" in m for m in messages)
    assert list(review_env["manifest_path"].parent.glob("*.frozen-*.json"))


def test_related_event_policy(review_env):
    _decide(review_env, "RELATED_EVENT")
    colocated, _ = apply(review_env["review"], review_env["db"], review_env["manifest_path"],
                         None, policy="colocate", **_OPTS(review_env))
    assert colocated["assignments"]["9005"] == colocated["assignments"]["9006"]
    assert colocated["review"]["related_event_policy"] == "colocate"


def test_distinct_adds_no_constraint_and_can_drop_a_manual_link():
    a, b, _ = MANUAL_LINKS[0]
    decided = [
        {"a": a, "b": b, "decision": "DISTINCT", "reviewer": "JD", "rationale": "x" * 12},
        {"a": 1, "b": 2, "decision": "SAME_PRODUCT", "reviewer": "JD", "rationale": "y" * 12},
        {"a": 3, "b": 4, "decision": "RELATED_EVENT", "reviewer": "JD", "rationale": "z" * 12},
    ]
    links, dropped = effective_links(decided, "colocate")
    linked = {pair_id(x, y) for x, y, _ in links}
    assert (a, b) in dropped and pair_id(a, b) not in linked
    assert pair_id(1, 2) in linked and pair_id(3, 4) in linked
    allow_links, _ = effective_links(decided, "allow-cross-split")
    assert pair_id(3, 4) not in {pair_id(x, y) for x, y, _ in allow_links}
    with pytest.raises(ValueError):
        effective_links(decided, "merge-everything")


def test_cli_status_and_apply_exit_codes(review_env, capsys):
    common = ["--review", str(review_env["review"]), "--db", str(review_env["db"]),
              "--manifest", str(review_env["manifest_path"]),
              "--embeddings", str(review_env["tmp"] / "none"),
              "--proposals", str(review_env["proposals"])]
    # Isolated from data/annotations/: real annotation files must never change the outcome.
    isolated = ["--annotations-dir", str(review_env["annotations"])]
    assert main(["status", *common]) == 0
    assert "undecided: 1" in capsys.readouterr().out
    assert main(["apply", *common, *isolated]) == 2  # incomplete review -> nothing changed
    _decide(review_env, "DISTINCT")
    assert main(["apply", *common, *isolated]) == 4  # no semantic audit -> not frozen
    assert main(["apply", *common, *isolated, "--skip-semantic-audit"]) == 0
    assert load_manifest(review_env["manifest_path"])["frozen"] is True


def test_cli_apply_refuses_when_the_given_annotations_dir_has_records(review_env, capsys):
    """The --annotations-dir guard still protects existing labels (isolation, not bypass)."""
    annotations = review_env["annotations"]
    annotations.mkdir(parents=True, exist_ok=True)
    (annotations / "someone.jsonl").write_text('{"query_id": "q1"}\n', "utf-8")
    _decide(review_env, "DISTINCT")
    args = ["apply", "--review", str(review_env["review"]), "--db", str(review_env["db"]),
            "--manifest", str(review_env["manifest_path"]),
            "--embeddings", str(review_env["tmp"] / "none"),
            "--proposals", str(review_env["proposals"]),
            "--annotations-dir", str(annotations), "--skip-semantic-audit"]
    assert main(args) == 2
    assert "annotation records exist" in capsys.readouterr().out
    assert load_manifest(review_env["manifest_path"]).get("frozen") is not True


# --- safeguards added after code review ------------------------------------------------

def test_no_freeze_without_semantic_audit(review_env):
    _decide(review_env, "DISTINCT")
    opts = {**_OPTS(review_env), "require_semantic_audit": True}
    manifest, messages = apply(review_env["review"], review_env["db"],
                               review_env["manifest_path"], None, **opts)
    assert manifest["frozen"] is False
    assert manifest["checks"]["semantic_audit_available"] is False
    assert "semantic leakage audit did not run" in " ".join(messages)


def test_refreeze_never_leaves_an_unfrozen_manifest(review_env):
    _decide(review_env, "DISTINCT")
    frozen, _ = apply(review_env["review"], review_env["db"], review_env["manifest_path"],
                      None, **_OPTS(review_env))
    before = review_env["manifest_path"].read_bytes()
    opts = {**_OPTS(review_env), "require_semantic_audit": True}
    with pytest.raises(RuntimeError, match="Refusing to replace the frozen manifest"):
        apply(review_env["review"], review_env["db"], review_env["manifest_path"], None,
              refreeze=True, **opts)
    assert review_env["manifest_path"].read_bytes() == before


def test_assignment_change_is_blocked_when_annotations_exist(review_env):
    annotations = review_env["annotations"]
    annotations.mkdir()
    (annotations / "ab.jsonl").write_text('{"query_id": "AB-0001"}\n', "utf-8")
    _decide(review_env, "SAME_PRODUCT")  # forces 9005/9006 together -> assignment changes
    with pytest.raises(RuntimeError, match="annotation records exist"):
        apply(review_env["review"], review_env["db"], review_env["manifest_path"], None,
              **_OPTS(review_env))
    manifest, _ = apply(review_env["review"], review_env["db"], review_env["manifest_path"],
                        None, allow_annotation_invalidation=True, **_OPTS(review_env))
    assert manifest["frozen"] is True


def test_reexport_keeps_partial_human_rows(review_env):
    rows = read_review(review_env["review"])
    rows[0]["reviewer"] = "Jane Doe"  # started, no decision yet
    write_review(rows, review_env["review"])
    export(review_env["manifest"], review_env["db"], review_env["review"], None, None)
    assert any(r["reviewer"] == "Jane Doe" for r in read_review(review_env["review"]))


def test_duplicate_pair_rows_are_rejected(review_env):
    rows = read_review(review_env["review"])
    write_review(rows + [rows[0]], review_env["review"])
    with pytest.raises(ValueError, match="duplicate pair_id"):
        read_review(review_env["review"])


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"rationale": "AI text from titles, honestly"}, "copies the assistant"),
        ({"reviewed_at": "11/15/2026"}, "YYYY-MM-DD"),
        ({"reviewed_at": "2999-01-01"}, "future"),
        ({"reviewed_at": "2026-02-30"}, "real calendar date"),
    ],
)
def test_near_copied_rationale_and_bad_dates_rejected(review_env, overrides, message):
    rows = _decide(review_env, "DISTINCT", **overrides)
    proposals = json.loads(review_env["proposals"].read_text("utf-8"))["proposals"]
    _, errors, _ = validate_review(rows, review_env["ids"], review_env["fingerprint"], False,
                                   proposals)
    assert any(message in e for e in errors), errors


def test_real_names_resembling_ai_tokens_are_accepted(review_env):
    rows = _decide(review_env, "DISTINCT", reviewer="Ai Nguyen")
    decided, errors, _ = validate_review(rows, review_env["ids"], review_env["fingerprint"], False)
    assert not errors and decided[0]["reviewer"] == "Ai Nguyen"


def test_hand_edited_manifest_is_detected(review_env):
    from evaluation.annotations import corpus_context

    manifest = load_manifest(review_env["manifest_path"])
    manifest["assignments"]["9005"] = "validation"  # edit without updating the fingerprint
    review_env["manifest_path"].write_text(json.dumps(manifest), "utf-8")
    with pytest.raises(RuntimeError, match="edited by hand"):
        corpus_context(review_env["db"], review_env["manifest_path"])


# --- export --all-candidates -----------------------------------------------------------

def _all_candidates_env(review_env, threshold=-1.0):
    """Fake embeddings + a threshold of -1 so EVERY pair is a semantic candidate; the
    ambiguous cheddar pair is put back into ONE split so only all-candidates lists it."""
    from conftest import build_fake_embeddings

    manifest = review_env["manifest"]
    manifest["assignments"]["9006"] = manifest["assignments"]["9005"]
    manifest["semantic_audit"] = {"available": True, "threshold": threshold, "top_pairs": []}
    emb = build_fake_embeddings(review_env["db"], review_env["tmp"] / "emb")
    return manifest, emb


def _merged_pairs(manifest):
    return {pair_id(a, b) for g in manifest["groups"] for a in g["recall_ids"]
            for b in g["recall_ids"] if a < b}


def test_all_candidates_lists_non_crossing_pairs_but_skips_merged_groups(review_env):
    manifest, emb = _all_candidates_env(review_env)
    out = review_env["tmp"] / "all.csv"
    export(manifest, review_env["db"], out, emb, None)  # default scope: crossing only
    assert PAIR not in {r["pair_id"] for r in read_review(out)}

    meta = export(manifest, review_env["db"], out, emb, None, all_candidates=True)
    keys = {r["pair_id"] for r in read_review(out)}
    merged = _merged_pairs(manifest)
    assert PAIR in keys                       # same split now, still a candidate
    assert merged and not (keys & merged)     # already grouped -> never needs review
    assert meta["scope"] == "all_candidates" and meta["pairs"] == len(keys)


def test_all_candidates_never_overwrites_reviewed_rows(review_env):
    rows = _decide(review_env, "DISTINCT", reviewer="Jane Doe", reviewed_at="2026-11-15")
    before = next(r for r in rows if r["pair_id"] == PAIR)
    manifest, emb = _all_candidates_env(review_env)
    export(manifest, review_env["db"], review_env["review"], emb, None, all_candidates=True)
    after = {r["pair_id"]: r for r in read_review(review_env["review"])}
    assert all(after[PAIR][c] == before[c] for c in
               ("decision", "reviewer", "rationale", "reviewed_at", "corpus_fingerprint"))
    new = [r for k, r in after.items() if k != PAIR]
    assert new and all(not r["decision"] and not r["reviewer"] for r in new)


def test_all_candidates_keeps_reviewed_rows_that_left_the_candidate_set(review_env):
    _decide(review_env, "DISTINCT")
    manifest, emb = _all_candidates_env(review_env, threshold=2.0)  # no semantic pairs
    manifest["ambiguous_pairs"] = []                                # no lexical pairs
    export(manifest, review_env["db"], review_env["review"], emb, None, all_candidates=True)
    row = next(r for r in read_review(review_env["review"]) if r["pair_id"] == PAIR)
    assert row["decision"] == "DISTINCT" and row["reviewer"] == "Jane Doe"


def test_all_candidates_refuses_without_embeddings(review_env):
    with pytest.raises(RuntimeError, match="all-candidates"):
        export(review_env["manifest"], review_env["db"], review_env["tmp"] / "x.csv", None, None,
               all_candidates=True)
    with pytest.raises(RuntimeError, match="all-candidates"):
        export(review_env["manifest"], review_env["db"], review_env["tmp"] / "x.csv",
               review_env["tmp"] / "missing", None, all_candidates=True)


def test_cli_all_candidates_flag(review_env, capsys):
    manifest, emb = _all_candidates_env(review_env)
    write_manifest(manifest, review_env["manifest_path"])
    out = review_env["tmp"] / "cli.csv"
    code = main(["export", "--all-candidates", "--review", str(out), "--db", str(review_env["db"]),
                 "--manifest", str(review_env["manifest_path"]), "--embeddings", str(emb)])
    assert code == 0 and '"scope": "all_candidates"' in capsys.readouterr().out
    assert PAIR in {r["pair_id"] for r in read_review(out)}
