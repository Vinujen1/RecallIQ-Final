import json

import pytest

from evaluation import DEFAULT_DB, DEFAULT_MANIFEST, SPLITS
from evaluation.splits import (
    assign_splits,
    build_groups,
    build_manifest,
    extract_brand,
    find_violations,
    load_split_records,
    split_lookup,
)
from retrieval.corpus import corpus_fingerprint, load_corpus


def test_split_is_reproducible_with_the_same_seed(eval_env):
    again = build_manifest(eval_env["db"], seed=4110)
    assert again["assignments"] == eval_env["manifest"]["assignments"]
    assert again["groups"] == eval_env["manifest"]["groups"]


def test_different_seed_changes_assignment(eval_env):
    other = build_manifest(eval_env["db"], seed=1)
    assert other["assignments"] != eval_env["manifest"]["assignments"]


def test_every_record_assigned_once_and_targets_roughly_met(eval_env):
    manifest = eval_env["manifest"]
    corpus_ids = {r.recall_id for r in load_corpus(eval_env["db"])}
    assert set(map(int, manifest["assignments"])) == corpus_ids
    counts = manifest["counts"]
    assert sum(counts.values()) == len(corpus_ids)
    assert counts["train"] > counts["validation"] > 0 and counts["test"] > 0
    assert abs(manifest["achieved_fractions"]["train"] - 0.6) < 0.1


def test_near_duplicates_and_same_brand_share_a_split(eval_env):
    split_of = split_lookup(eval_env["manifest"])
    assert split_of[9001] == split_of[9002]  # product-key Jaccard 0.75
    assert split_of[9003] == split_of[9004]  # same brand 'ruffles'
    rationale = " ".join(r for g in eval_env["manifest"]["groups"] for r in g["rationale"])
    assert "Jaccard" in rationale and "same brand 'ruffles'" in rationale


def test_ambiguous_pairs_are_reported_not_merged(eval_env):
    pairs = eval_env["manifest"]["ambiguous_pairs"]
    cheddar = [p for p in pairs if sorted(p["recall_ids"]) == [9005, 9006]]
    assert cheddar and cheddar[0]["jaccard"] == 0.5
    assert len(cheddar[0]["splits"]) == 2
    assert eval_env["manifest"]["checks"]["ambiguous_pairs_total"] >= 1


def test_manifest_records_provenance(eval_env):
    manifest = eval_env["manifest"]
    assert manifest["seed"] == 4110
    assert manifest["corpus_fingerprint"] == corpus_fingerprint(load_corpus(eval_env["db"]))
    assert manifest["grouping"]["uses_heuristic_product_family"] is False
    assert manifest["checks"]["title_groups_crossing_splits"] == 0
    assert manifest["checks"]["merged_groups_crossing_splits"] == 0
    assert eval_env["manifest_path"].with_suffix(".csv").is_file()


def test_leakage_detection_flags_split_groups(eval_env):
    records = load_split_records(eval_env["db"])
    groups, _ = build_groups(records)
    split_of = split_lookup(eval_env["manifest"])
    assert find_violations(split_of, records, groups) == []

    leaked = dict(split_of)
    leaked[9002] = next(s for s in SPLITS if s != split_of[9001])  # separate the fish cakes
    problems = find_violations(leaked, records, groups)
    assert any("spans splits" in p for p in problems)

    missing = dict(split_of)
    missing.pop(9001)
    assert any("exactly once" in p for p in find_violations(missing, records, groups))


def test_title_group_leak_is_detected(eval_env):
    records = load_split_records(eval_env["db"])
    twin = records[0]
    clone = type(twin)(recall_id=1, title=twin.title, title_group=twin.title_group,
                       domain=twin.domain, primary_category=twin.primary_category,
                       needs_review=False, key_tokens=twin.key_tokens)
    split_of = {r.recall_id: "train" for r in records} | {1: "test"}
    problems = find_violations(split_of, records + [clone], [])
    assert any(twin.title_group in p for p in problems)


@pytest.mark.parametrize(
    ("title", "brand"),
    [
        ("Ruffles brand Regular Potato Chips recalled", "ruffles"),
        ("Undeclared milk in certain Kakao brand Dark Chocolate", "kakao"),
        ("Certain Central-Epicure brand smoked salmon", "centralepicure"),
        ("Various brands of oysters recalled", None),
        ("Plush Hippopotamus Toy", None),
    ],
)
def test_extract_brand(title, brand):
    assert extract_brand(title) == brand


def test_split_does_not_depend_on_record_order(eval_env):
    import random

    from evaluation.splits import assign_splits

    records = load_split_records(eval_env["db"])
    shuffled = records[:]
    random.Random(7).shuffle(shuffled)
    a = assign_splits(build_groups(records)[0], seed=4110)
    b = assign_splits(build_groups(shuffled)[0], seed=4110)
    assert a == b


def test_multi_record_groups_are_not_all_sent_to_train(eval_env):
    per_split = eval_env["manifest"]["multi_record_groups_per_split"]
    assert sum(per_split.values()) == 2  # fish cakes + Ruffles
    assert per_split["train"] < 2


def test_splits_cli_refuses_to_change_an_existing_assignment(eval_env, capsys):
    from evaluation.splits import main

    common = ["--db", str(eval_env["db"]), "--out", str(eval_env["manifest_path"]),
              "--embeddings", str(eval_env["tmp"] / "none")]
    assert main([*common, "--seed", "4110"]) == 0       # same assignment: fine
    assert main([*common, "--seed", "1"]) == 1          # different: refused
    assert "Refusing to overwrite" in capsys.readouterr().out
    assert main([*common, "--seed", "1", "--force"]) == 0


def test_semantic_audit_lists_cross_split_pairs(eval_env):
    from conftest import build_fake_embeddings
    from evaluation.splits import semantic_cross_split_pairs

    emb = build_fake_embeddings(eval_env["db"], eval_env["tmp"] / "emb")
    split_of = split_lookup(eval_env["manifest"])
    audit = semantic_cross_split_pairs(eval_env["db"], emb, split_of, threshold=0.0, limit=5)
    assert audit["available"] and audit["pairs_crossing_splits"] > 0
    assert all(len(set(p["splits"])) == 2 for p in audit["top_pairs"])
    missing = semantic_cross_split_pairs(eval_env["db"], eval_env["tmp"] / "x", split_of)
    assert missing["available"] is False


def test_assign_splits_rejects_bad_targets():
    with pytest.raises(ValueError):
        assign_splits([], targets={"train": 0.5, "validation": 0.5})


@pytest.mark.skipif(not DEFAULT_MANIFEST.is_file() or not DEFAULT_DB.is_file(),
                    reason="real split manifest not built")
def test_real_manifest_is_consistent_and_leak_checked():
    manifest = json.loads(DEFAULT_MANIFEST.read_text("utf-8"))
    corpus = load_corpus(DEFAULT_DB)
    assert manifest["corpus_fingerprint"] == corpus_fingerprint(corpus)
    assert sum(manifest["counts"].values()) == len(corpus) == 1000
    assert manifest["seed"] == 4110
    records = load_split_records(DEFAULT_DB)
    # check against the links the manifest was built with: after `split_review apply` these
    # are the human-reviewed links (a DISTINCT review removes a default manual link)
    links = tuple(tuple(link) for link in manifest["grouping"]["manual_links"])
    groups, _ = build_groups(records, manual_links=links)
    assert find_violations(split_lookup(manifest), records, groups) == []
