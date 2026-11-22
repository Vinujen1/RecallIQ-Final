"""Contrastive loss and training loop on a TINY randomly initialised BERT (no download).

Losses here are from synthetic toy data and only check mechanics; they say nothing about
retrieval quality.
"""

import json
import math

import pytest
import torch

from training.config import TrainingConfig
from training.data import TrainingPair
from training.model import MeanPoolEncoder
from training.trainer import contrastive_loss, evaluate_loss, set_seed, train

DOCS = {
    1: "frozen fish cake recalled due to undeclared egg",
    2: "cheddar cheese recalled due to listeria",
    3: "potato chips recalled due to undeclared milk",
    4: "baby high chair recalled due to fall hazard",
    5: "oysters recalled due to salmonella",
    6: "bicycle fork recalled due to crash hazard",
    7: "sesame seeds recalled due to salmonella",
    8: "chocolate milk recalled due to spoilage",
}
QUERIES = {1: "fish cake egg", 2: "cheese listeria", 3: "chips milk", 4: "high chair fall",
           5: "oysters salmonella", 6: "bike fork crash", 7: "sesame salmonella",
           8: "chocolate milk spoiled"}


@pytest.fixture
def tiny_model(tmp_path):
    from transformers import BertConfig, BertModel, BertTokenizer

    words = sorted({w for t in (*DOCS.values(), *QUERIES.values()) for w in t.split()})
    vocab = tmp_path / "vocab.txt"
    vocab.write_text("\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", *words]), "utf-8")
    tokenizer = BertTokenizer(str(vocab))
    torch.manual_seed(0)
    model = BertModel(BertConfig(vocab_size=len(words) + 5, hidden_size=32, num_hidden_layers=1,
                                 num_attention_heads=2, intermediate_size=64,
                                 max_position_embeddings=64))
    return MeanPoolEncoder(model, tokenizer, max_seq_length=32)


def _pairs():
    return [TrainingPair(QUERIES[i], i, (i % 8 + 1,), "pseudo:title_core", f"q{i}") for i in DOCS]


GROUPS = {i: f"g{i}" for i in DOCS}


def test_contrastive_loss_matches_hand_computation():
    q = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    c = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.6, 0.8]])  # 2 positives + 1 hard negative
    loss = contrastive_loss(q, c, ["a", "b"], ["a", "b", "x"], scale=1.0)
    row0 = -math.log(math.e / (math.e + 1 + math.exp(0.6)))
    row1 = -math.log(math.e / (1 + math.e + math.exp(0.8)))
    assert loss.item() == pytest.approx((row0 + row1) / 2, rel=1e-5)


def test_same_group_candidates_are_masked():
    q = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    c = torch.tensor([[1.0, 0.0], [1.0, 0.0]])   # two views of the SAME recall in one batch
    masked = contrastive_loss(q, c, ["g", "g"], ["g", "g"], scale=1.0)
    unmasked = contrastive_loss(q, c, ["g", "h"], ["g", "h"], scale=1.0)
    assert masked.item() == pytest.approx(0.0, abs=1e-6)
    assert unmasked.item() == pytest.approx(math.log(2), rel=1e-5)


def test_encoder_outputs_unit_vectors(tiny_model):
    vectors = tiny_model.encode(list(DOCS.values()))
    assert vectors.shape == (8, 32)
    assert vectors.dtype.name == "float32"
    assert (abs((vectors ** 2).sum(1) - 1) < 1e-5).all()


def test_training_writes_history_and_reloadable_checkpoints(tiny_model, tmp_path):
    config = TrainingConfig(epochs=2, batch_size=4, learning_rate=1e-3, warmup_ratio=0.0)
    result = train(tiny_model, _pairs(), _pairs()[:4], DOCS, GROUPS, config, tmp_path / "run",
                   {"note": "synthetic"})
    run = tmp_path / "run"
    for name in ("run_info.json", "history.json", "summary.json"):
        assert (run / name).is_file()
    history = json.loads((run / "history.json").read_text("utf-8"))
    assert [e["epoch"] for e in history["epochs"]] == [0, 1, 2]
    assert all(math.isfinite(e["validation_loss"]) for e in history["epochs"])
    assert all(math.isfinite(s["loss"]) for s in history["step_loss"]) and history["step_loss"]
    info = json.loads((run / "run_info.json").read_text("utf-8"))
    assert info["config"]["seed"] == config.seed and info["data"]["train_pairs"] == 8
    assert result.best_epoch in (0, 1, 2)          # epoch 0 = the base model is a candidate
    assert info["truncated_smoke_run"] is False and info["scheduled_steps"] == 4
    assert info["device"] == "cpu" and "git_dirty" in info["environment"]
    reloaded = MeanPoolEncoder.load(run / "checkpoints" / "last")
    texts = list(DOCS.values())
    assert torch.allclose(torch.from_numpy(reloaded.encode(texts)),
                          torch.from_numpy(tiny_model.encode(texts)), atol=1e-5)
    assert (run / "checkpoints" / "best" / "recalliq_encoder.json").is_file()


def test_training_reduces_training_loss_on_toy_data(tiny_model, tmp_path):
    config = TrainingConfig(epochs=15, batch_size=4, learning_rate=5e-3, warmup_ratio=0.0)
    before = evaluate_loss(tiny_model, _pairs(), DOCS, GROUPS, config)
    train(tiny_model, _pairs(), _pairs(), DOCS, GROUPS, config, tmp_path / "run")
    after = evaluate_loss(tiny_model, _pairs(), DOCS, GROUPS, config)
    assert after < before


def test_same_seed_gives_identical_runs(tiny_model, tmp_path):
    import copy

    config = TrainingConfig(epochs=1, batch_size=4, learning_rate=1e-3, warmup_ratio=0.0)
    first, second = copy.deepcopy(tiny_model), copy.deepcopy(tiny_model)
    set_seed(1)
    train(first, _pairs(), _pairs()[:4], DOCS, GROUPS, config, tmp_path / "a")
    train(second, _pairs(), _pairs()[:4], DOCS, GROUPS, config, tmp_path / "b")
    a = json.loads((tmp_path / "a" / "history.json").read_text("utf-8"))["step_loss"]
    b = json.loads((tmp_path / "b" / "history.json").read_text("utf-8"))["step_loss"]
    assert [s["loss"] for s in a] == [s["loss"] for s in b]


@pytest.mark.parametrize(("n", "bs"), [(8, 4), (9, 4), (10, 4), (3, 4), (1, 4), (33, 32)])
def test_scheduled_steps_match_actual_batches(n, bs):
    from training.trainer import _batches, batches_per_epoch

    pairs = [TrainingPair("q", 1, (), "pseudo:title_core", str(i)) for i in range(n)]
    assert batches_per_epoch(n, bs) == sum(1 for _ in _batches(pairs, bs, None))


def test_max_steps_marks_a_truncated_run(tiny_model, tmp_path):
    config = TrainingConfig(epochs=3, batch_size=4, learning_rate=1e-3, warmup_ratio=0.0)
    train(tiny_model, _pairs(), _pairs()[:4], DOCS, GROUPS, config, tmp_path / "run", max_steps=1)
    info = json.loads((tmp_path / "run" / "run_info.json").read_text("utf-8"))
    summary = json.loads((tmp_path / "run" / "summary.json").read_text("utf-8"))
    assert info["truncated_smoke_run"] is True and summary["total_steps"] == 1
