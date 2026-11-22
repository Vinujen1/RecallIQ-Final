"""Model B vs Model C configuration, consumer-style views, leakage guards, proxy selection
(synthetic corpus and a tiny random BERT; no real data, no download)."""

import json
from pathlib import Path

import pytest

from retrieval.corpus import CorpusRecord
from training.config import TrainingConfig
from training.data import (
    TrainingPair,
    brand_of,
    build_pairs,
    drop_text_collisions,
    normalize_query,
    plain_hazard,
    pseudo_queries,
    reviewed_distinct_pairs,
    title_core,
)
from training.train import proxy_queries_from

from tests.training.test_trainer import DOCS, GROUPS, _pairs, tiny_model  # noqa: F401
from tests.training.test_training_data import env  # noqa: F401

REPO = Path(__file__).resolve().parents[2]
VIEWS = ("title_core", "generic_issue", "product_issue", "consumer_question", "brand_product")


def _cfg(**kw):
    return TrainingConfig(bm25_candidates=20, semantic_candidates=0, pseudo_query_views=VIEWS, **kw)


def test_shipped_b_and_c_configs_differ_only_in_hard_negatives():
    b = json.loads((REPO / "ml" / "configs" / "model_b.json").read_text("utf-8"))
    c = json.loads((REPO / "ml" / "configs" / "model_c.json").read_text("utf-8"))
    diff = {k for k in b if b[k] != c[k]}
    assert diff == {"hard_negatives_per_pair", "experiment_label"}
    assert b["hard_negatives_per_pair"] == 0 and c["hard_negatives_per_pair"] > 0
    for data in (b, c):
        TrainingConfig.from_json(data)  # valid


def test_model_b_pairs_have_no_negatives_but_same_queries_as_c(env):  # noqa: F811
    b = build_pairs(env["corpus"], env["manifest"], "train", _cfg(hard_negatives_per_pair=0))
    c = build_pairs(env["corpus"], env["manifest"], "train", _cfg(hard_negatives_per_pair=3))
    assert all(not p.negative_ids for p in b)
    assert any(p.negative_ids for p in c)
    assert [(p.query, p.positive_id, p.source) for p in b] == \
           [(p.query, p.positive_id, p.source) for p in c]


def test_validation_pairs_identical_for_both_variants(env):  # noqa: F811
    pairs = [build_pairs(env["corpus"], env["manifest"], "validation",
                         _cfg(hard_negatives_per_pair=h), num_negatives=3) for h in (0, 3)]
    assert pairs[0] == pairs[1]


def test_reviewed_distinct_negative_comes_first_and_respects_split(env):  # noqa: F811
    rows = [{"recall_id_a": "9002", "recall_id_b": "9005", "decision": "DISTINCT", "reviewer": "VD"},
            {"recall_id_a": "9002", "recall_id_b": "9006", "decision": "DISTINCT", "reviewer": "VD"},
            {"recall_id_a": "9003", "recall_id_b": "9005", "decision": "", "reviewer": ""}]
    distinct = reviewed_distinct_pairs(rows)
    assert distinct == {9002: {9005, 9006}, 9005: {9002}, 9006: {9002}}
    pairs = build_pairs(env["corpus"], env["manifest"], "train", _cfg(hard_negatives_per_pair=3),
                        verified_distinct=distinct)
    fish = [p for p in pairs if p.positive_id == 9002]
    assert fish and all(p.negative_ids[0] == 9005 for p in fish)       # 9005: train, verified
    assert all(9006 not in p.negative_ids for p in pairs)               # 9006 is in test
    off = build_pairs(env["corpus"], env["manifest"], "train", _cfg(hard_negatives_per_pair=0),
                      verified_distinct=distinct)
    assert all(not p.negative_ids for p in off)                          # Model B: none at all


def test_text_collisions_with_held_out_queries_are_dropped():
    pairs = [TrainingPair("Is my  Cheddar safe?", 1, (), "pseudo:consumer_question", "a"),
             TrainingPair("frozen fish cake", 2, (), "pseudo:title_core", "b")]
    kept, dropped = drop_text_collisions(pairs, {normalize_query("is my cheddar safe")})
    assert dropped == 1 and [p.query_id for p in kept] == ["b"]


def test_consumer_views_paraphrase_without_copying_title():
    record = CorpusRecord(42, "Naturally Good brand Hulled Sesame Seeds recalled due to Salmonella",
                          "Nuts", "Salmonella", "https://x", "full text")
    views = dict(pseudo_queries(record, None, VIEWS))
    assert "naturally good" not in views["consumer_question"]          # brand dropped
    assert "hulled sesame seeds" in views["consumer_question"]
    assert views["brand_product"] == "Naturally Good hulled sesame seeds"
    assert views == dict(pseudo_queries(record, None, VIEWS))           # deterministic
    assert plain_hazard("Fire hazard - Burn hazard", "1").count(" and ") == 1
    assert plain_hazard("Unlisted thing", "1") == "unlisted thing"
    assert brand_of("Cheddar cheese") == ""


@pytest.mark.parametrize("title,core", [
    ("Archived � Health Canada Reminds Canadians About Garage Sale Safety",
     "Health Canada Reminds Canadians About Garage Sale Safety"),
    ("Expanded recall: NicPro brand Epoxy Resin Kits recalled due to improper labelling",
     "NicPro brand Epoxy Resin Kits"),
    ("London Drugs Limited recalls Apex Premium LED Lighting Set", "Apex Premium LED Lighting Set"),
])
def test_title_core_handles_prefixes_and_company_recalls(title, core):
    assert title_core(title) == core


def test_proxy_queries_merge_duplicates_include_group_and_skip_titles():
    pairs = [TrainingPair("cheese listeria", 1, (), "pseudo:generic_issue", "v-1"),
             TrainingPair("Cheese  Listeria", 2, (), "pseudo:generic_issue", "v-2"),
             TrainingPair("Brand cheese", 1, (), "pseudo:title_core", "v-3")]
    proxy = proxy_queries_from(pairs, {1: "g1", 2: "g2", 3: "g1"})
    assert len(proxy) == 1
    _, _, rel = proxy[0]
    assert rel == {1: 1, 3: 1, 2: 1}                    # group member 3 + merged doc 2


def test_proxy_selection_and_both_validation_losses_logged(tiny_model, tmp_path):  # noqa: F811
    from training.trainer import train

    config = TrainingConfig(epochs=2, batch_size=4, learning_rate=1e-3, warmup_ratio=0.0,
                            hard_negatives_per_pair=0)
    proxy = [(f"p{i}", text, {i: 1}) for i, text in
             {1: "fish cake egg", 2: "cheese listeria", 4: "high chair fall"}.items()]
    train(tiny_model, _pairs(), _pairs()[:4], DOCS, GROUPS, config, tmp_path / "run",
          proxy_queries=proxy)
    history = json.loads((tmp_path / "run" / "history.json").read_text("utf-8"))
    info = json.loads((tmp_path / "run" / "run_info.json").read_text("utf-8"))
    assert history["selection_metric"] == "proxy_mrr@10" == info["selection_metric_used"]
    for row in history["epochs"]:
        assert {"validation_loss", "validation_loss_in_batch_only", "proxy"} <= set(row)
        assert 0 <= row["proxy"]["mrr@10"] <= 1
    assert "in-batch negatives only" in info["loss"]


def test_selection_falls_back_to_loss_without_proxy_and_says_so(tiny_model, tmp_path):  # noqa: F811
    from training.trainer import train

    config = TrainingConfig(epochs=1, batch_size=4, learning_rate=1e-3, warmup_ratio=0.0)
    train(tiny_model, _pairs(), _pairs()[:4], DOCS, GROUPS, config, tmp_path / "run")
    info = json.loads((tmp_path / "run" / "run_info.json").read_text("utf-8"))
    assert info["selection_metric_requested"] == "proxy_mrr@10"
    assert info["selection_metric_used"] == "validation_loss"


def test_early_stopping_stops_after_patience(tiny_model, tmp_path):  # noqa: F811
    from training.trainer import train

    # learning rate ~0 -> the selection score cannot improve after epoch 0
    config = TrainingConfig(epochs=6, batch_size=4, learning_rate=1e-12, warmup_ratio=0.0,
                            early_stopping_patience=2, selection_metric="validation_loss")
    train(tiny_model, _pairs(), _pairs()[:4], DOCS, GROUPS, config, tmp_path / "run")
    summary = json.loads((tmp_path / "run" / "summary.json").read_text("utf-8"))
    assert summary["stopped_early"] is True and summary["epochs_run"] < 6


def test_invalid_variant_settings_are_rejected():
    with pytest.raises(ValueError):
        TrainingConfig(selection_metric="test_mrr").validate()
    with pytest.raises(ValueError):
        TrainingConfig(early_stopping_patience=-1).validate()
    with pytest.raises(ValueError):
        TrainingConfig(validation_hard_negatives=-1).validate()


# --- duplicate-text masking (fix found by the ECC mle-reviewer, 2026-11-16) ---------------

def test_duplicate_text_mask_marks_only_other_queries_with_same_text():
    from training.trainer import duplicate_text_mask

    mask = duplicate_text_mask(["Cheddar  cheese listeria", "cheddar cheese listeria", "egg"], 5)
    assert mask.tolist() == [[False, True, False, False, False],
                             [True, False, False, False, False],
                             [False, False, False, False, False]]


def test_same_text_positive_is_removed_from_the_softmax():
    import math

    import torch

    from training.trainer import contrastive_loss, duplicate_text_mask

    q = torch.tensor([[1.0, 0.0], [1.0, 0.0]])          # two queries, same text
    c = torch.tensor([[1.0, 0.0], [1.0, 0.0]])          # their (different-group) positives
    groups = ["g1", "g2"]
    unmasked = contrastive_loss(q, c, groups, groups, scale=1.0)
    masked = contrastive_loss(q, c, groups, groups, scale=1.0,
                              extra_mask=duplicate_text_mask(["a b", "a b"], 2))
    assert unmasked.item() == pytest.approx(math.log(2))   # each positive punished as negative
    assert masked.item() == pytest.approx(0.0, abs=1e-6)   # only the own positive remains


def test_v1_configs_keep_old_behaviour_and_v2_pairs_differ_only_in_hard_negatives():
    for name in ("model_b.json", "model_c.json"):
        config = TrainingConfig.load(REPO / "ml" / "configs" / name)
        assert config.mask_duplicate_query_texts is False      # v1 runs stay reproducible
    for seed in (4110, 4111, 4112):
        b = json.loads((REPO / "ml" / "configs" / f"model_b_v2_seed{seed}.json").read_text("utf-8"))
        c = json.loads((REPO / "ml" / "configs" / f"model_c_v2_seed{seed}.json").read_text("utf-8"))
        assert b["seed"] == c["seed"] == seed and b["mask_duplicate_query_texts"] is True
        assert {k for k in b if b[k] != c[k]} == {"hard_negatives_per_pair", "experiment_label"}
