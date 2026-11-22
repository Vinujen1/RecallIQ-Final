"""Training-pair generation: split isolation, hard negatives, determinism (synthetic data)."""

import pytest

from evaluation.annotations import Annotation, Judgment
from evaluation.splits import split_lookup
from retrieval.corpus import CorpusRecord, load_corpus
from training.config import TrainingConfig
from training.data import (
    LeakageError,
    TrainingPair,
    assert_no_leakage,
    build_pairs,
    pseudo_queries,
    require_frozen,
    summarize,
    title_core,
)

CONFIG = TrainingConfig(hard_negatives_per_pair=3, bm25_candidates=20, semantic_candidates=0)


@pytest.fixture
def env(eval_env):
    """Synthetic corpus; groups (9001/9002 merged, 9003/9004 merged) put in train, the
    ambiguous cheddar pair split train/test, singletons spread over the three splits."""
    manifest = eval_env["manifest"]
    assignments = manifest["assignments"]
    for i, rid in enumerate(sorted(int(k) for k in assignments)):
        assignments[str(rid)] = ("train", "train", "validation", "test")[i % 4]
    for rid in (9001, 9002, 9003, 9004, 9005):
        assignments[str(rid)] = "train"
    assignments["9006"] = "test"
    manifest["frozen"] = True
    corpus = load_corpus(eval_env["db"])
    return {**eval_env, "manifest": manifest, "corpus": corpus,
            "split_of": split_lookup(manifest)}


def test_title_core_strips_recall_boilerplate():
    assert title_core("Archived - Certain Sawmill Bay brand Gallo Mussels may contain Marine Biotoxin") \
        == "Sawmill Bay brand Gallo Mussels"
    assert title_core("Naturally Good brand Hulled Sesame Seeds recalled due to Salmonella") \
        == "Naturally Good brand Hulled Sesame Seeds"
    assert title_core("Health Canada warns that relight candles previously available on "
                      "Amazon.ca pose a fire hazard") == "relight candles"


def test_pseudo_query_views_are_distinct_and_labelled():
    record = CorpusRecord(1, "Naturally Good brand Hulled Sesame Seeds recalled due to Salmonella",
                          "Nuts", "Salmonella", "https://x", "full text")
    views = dict(pseudo_queries(record, "Hulled Sesame Seeds", CONFIG.pseudo_query_views))
    assert views == {"title_core": "Naturally Good brand Hulled Sesame Seeds",
                     "generic_issue": "Hulled Sesame Seeds Salmonella"}  # product_issue duplicate
    one_word = CorpusRecord(2, "Widgets recalled", None, None, "https://x", "Widgets")
    assert pseudo_queries(one_word, None, CONFIG.pseudo_query_views) == []


def test_pairs_use_only_their_split(env):
    for split in ("train", "validation"):
        pairs = build_pairs(env["corpus"], env["manifest"], split, CONFIG)
        assert pairs
        for p in pairs:
            assert env["split_of"][p.positive_id] == split
            assert all(env["split_of"][n] == split for n in p.negative_ids)
    test_ids = {r for r, s in env["split_of"].items() if s == "test"}
    train = build_pairs(env["corpus"], env["manifest"], "train", CONFIG)
    assert not test_ids & {i for p in train for i in (p.positive_id, *p.negative_ids)}


def test_test_split_is_refused(env):
    with pytest.raises(ValueError, match="only be built for"):
        build_pairs(env["corpus"], env["manifest"], "test", CONFIG)


def test_hard_negatives_exclude_same_group_and_near_duplicates(env):
    pairs = build_pairs(env["corpus"], env["manifest"], "train", CONFIG)
    fish = [p for p in pairs if p.positive_id == 9002]
    assert fish and all(9001 not in p.negative_ids for p in fish)      # merged group
    chips = [p for p in pairs if p.positive_id == 9003]
    assert chips and all(9004 not in p.negative_ids for p in chips)    # brand group
    assert any(p.negative_ids for p in pairs)
    assert all(p.positive_id not in p.negative_ids for p in pairs)
    loose = TrainingConfig(hard_negatives_per_pair=3, bm25_candidates=20, semantic_candidates=0,
                           false_negative_jaccard=1.01)
    strict = TrainingConfig(hard_negatives_per_pair=3, bm25_candidates=20, semantic_candidates=0,
                            false_negative_jaccard=0.0)
    assert any(p.negative_ids for p in build_pairs(env["corpus"], env["manifest"], "train", loose))
    assert not any(p.negative_ids for p in build_pairs(env["corpus"], env["manifest"], "train", strict))


def test_human_train_queries_never_use_held_out_documents(env):
    query = Annotation("VD-0001", "is my cheddar cheese safe", "train",
                       [Judgment(9005, 3), Judgment(9006, 2)], reviewer_status="reviewed",
                       annotator="VD", reviewer="AB", anchor_recall_id=9005)
    draft = Annotation("VD-0002", "frozen fish cake egg", "train", [Judgment(9001, 3)],
                       reviewer_status="draft", annotator="VD")
    pairs = build_pairs(env["corpus"], env["manifest"], "train", CONFIG,
                        annotations=[query, draft])
    human = [p for p in pairs if p.source == "human"]
    assert [p.query_id for p in human] == ["VD-0001"]       # draft ignored
    assert human[0].positive_id == 9005                      # 9006 is in test: not a positive
    assert 9006 not in human[0].negative_ids                 # ... and never a negative


def test_leakage_guard_and_frozen_guard(env):
    test_id = next(r for r, s in env["split_of"].items() if s == "test")
    bad = [TrainingPair("q", 9001, (test_id,), "pseudo:title_core", "x")]
    with pytest.raises(LeakageError):
        assert_no_leakage(bad, env["split_of"], "train")
    with pytest.raises(RuntimeError, match="not frozen"):
        require_frozen({**env["manifest"], "frozen": False})


def test_pairs_are_deterministic(env):
    a = summarize(build_pairs(env["corpus"], env["manifest"], "train", CONFIG))
    b = summarize(build_pairs(env["corpus"], env["manifest"], "train", CONFIG))
    assert a == b and a["pairs"] > 0


def test_config_round_trip_and_validation():
    config = TrainingConfig(epochs=2, pseudo_query_views=("title_core",))
    assert TrainingConfig.from_json(config.to_json()) == config
    with pytest.raises(ValueError):
        TrainingConfig.from_json({"epochs": 0})
    with pytest.raises(ValueError, match="unknown"):
        TrainingConfig.from_json({"learning_rat": 1e-5})


def test_train_query_repeating_a_held_out_query_is_detected():
    from training.data import heldout_query_collisions

    train_q = Annotation("VD-0001", "Is my  CHEDDAR cheese safe?", "train", [Judgment(9005)])
    val_q = Annotation("AB-0001", "is my cheddar cheese safe", "validation", [Judgment(9100)])
    other = Annotation("VD-0002", "frozen fish cake egg", "train", [Judgment(9001)])
    assert heldout_query_collisions([train_q, val_q, other]) == ["VD-0001"]
