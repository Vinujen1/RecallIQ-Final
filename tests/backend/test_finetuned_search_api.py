"""Fine-tuned models (B/C) through POST /search, GET /search/models and POST /explain.

Synthetic DB + a TINY random BERT saved as a RecallIQ checkpoint (no download): exercises
the real checkpoint loading, weights-hash compatibility check, index building and cosine
ranking. A final group of tests uses the real trained checkpoints when they exist.
"""

import json
import shutil

import pytest
from fastapi.testclient import TestClient

from app import search_service
from app.config import Settings, get_settings
from app.main import app
from conftest import build_fake_embeddings
from recall_data.build import run
from retrieval.finetuned import build_index, check_compatible
from retrieval.semantic import StaleIndexError, load_embeddings

MODELS = ("bm25", "minilm", "minilm_finetuned", "minilm_hard_negative")


def _tiny_checkpoint(directory, seed):
    import torch
    from transformers import BertConfig, BertModel, BertTokenizer

    from training.model import MeanPoolEncoder

    words = ["generated", "test", "recall", "cheese", "listeria", "toy", "choking", "milk",
             "model", "upc", "lot", "fire", "hazard", "product", "food"]
    vocab = directory.parent / f"vocab-{seed}.txt"
    vocab.write_text("\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", *words]), "utf-8")
    torch.manual_seed(seed)
    model = BertModel(BertConfig(vocab_size=len(words) + 5, hidden_size=24, num_hidden_layers=1,
                                 num_attention_heads=2, intermediate_size=32,
                                 max_position_embeddings=64))
    MeanPoolEncoder(model, BertTokenizer(str(vocab)), 32).save(directory, {"epoch": 1})
    return directory


@pytest.fixture
def env(raw_csv, tmp_path):
    out = tmp_path / "processed"
    run(raw_csv, out, size=30, seed=11)
    db = out / "recalliq.db"
    minilm = build_fake_embeddings(db, tmp_path / "emb")
    paths = {}
    for name, seed in (("b", 1), ("c", 2)):
        ckpt = _tiny_checkpoint(tmp_path / f"ckpt-{name}", seed)
        label = "minilm_finetuned" if name == "b" else "minilm_hard_negative"
        build_index(db, ckpt, tmp_path / f"emb-{name}", label)
        paths[name] = (ckpt, tmp_path / f"emb-{name}")
    return {"db": db, "minilm": minilm, "b": paths["b"], "c": paths["c"], "tmp": tmp_path}


def _settings(env, **overrides):
    values = dict(database_path=env["db"], embeddings_dir=env["minilm"],
                  model_b_checkpoint=env["b"][0], model_b_embeddings_dir=env["b"][1],
                  model_c_checkpoint=env["c"][0], model_c_embeddings_dir=env["c"][1])
    values.update(overrides)
    return Settings(**values)


@pytest.fixture
def client_for(env):
    def make(**overrides):
        search_service._cache.clear()
        settings = _settings(env, **overrides)
        app.dependency_overrides[get_settings] = lambda: settings
        return TestClient(app)
    yield make
    app.dependency_overrides.clear()
    search_service._cache.clear()


def test_each_model_is_selectable_and_labelled(client_for):
    client = client_for()
    for model in ("minilm_finetuned", "minilm_hard_negative"):
        response = client.post("/search", json={"query": "Generated test recall", "model": model, "top_k": 5})
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["model"] == model and body["score_type"] == "cosine_similarity"
        assert "Model B" in body["score_description"] or "Model C" in body["score_description"]
        scores = [h["score"] for h in body["results"]]
        assert len(scores) == 5 and scores == sorted(scores, reverse=True)
        assert all(-1.0001 <= s <= 1.0001 for s in scores)


def test_fine_tuned_rankings_are_deterministic_and_indexes_are_model_specific(client_for, env):
    client = client_for()
    body = {"query": "Generated test recall", "top_k": 10}
    for model in ("minilm_finetuned", "minilm_hard_negative"):
        first = client.post("/search", json={**body, "model": model}).json()["results"]
        assert first and first == client.post("/search", json={**body, "model": model}).json()["results"]
    # each index records the weights hash of the checkpoint that built it
    b, c = (load_embeddings(env[k][1]).metadata for k in ("b", "c"))
    assert b["checkpoint_sha256"] != c["checkpoint_sha256"]
    assert b["label"] == "minilm_finetuned" and c["label"] == "minilm_hard_negative"


def test_models_endpoint_reports_availability(client_for, env):
    status = {m["model"]: m for m in client_for().get("/search/models").json()["models"]}
    assert set(status) == set(MODELS)
    assert status["bm25"]["available"] and status["minilm_finetuned"]["available"]
    assert status["minilm_hard_negative"]["available"]
    # the fixture's MiniLM index is a fake (model_id "fake"): correctly reported unavailable
    assert not status["minilm"]["available"] and status["minilm"]["reason"]
    missing = client_for(model_c_checkpoint=env["tmp"] / "nope").get("/search/models").json()
    by = {m["model"]: m for m in missing["models"]}
    assert by["minilm_finetuned"]["available"] and not by["minilm_hard_negative"]["available"]
    assert "minilm_hard_negative" in by["minilm_hard_negative"]["reason"]


def test_missing_checkpoint_or_index_gives_503_with_reason(client_for, env):
    client = client_for(model_b_checkpoint=env["tmp"] / "missing-ckpt",
                        model_c_embeddings_dir=env["tmp"] / "missing-index")
    for model in ("minilm_finetuned", "minilm_hard_negative"):
        response = client.post("/search", json={"query": "cheese", "model": model})
        assert response.status_code == 503
        assert "missing" in response.json()["detail"].lower()
    assert client.post("/search", json={"query": "cheese", "model": "bm25"}).status_code == 200


def test_index_built_by_another_checkpoint_is_refused(client_for, env):
    # serve Model B's index with Model C's checkpoint -> weights hash mismatch
    with pytest.raises(StaleIndexError, match="different checkpoint"):
        check_compatible(load_embeddings(env["b"][1]), env["c"][0])
    client = client_for(model_b_embeddings_dir=env["c"][1])
    response = client.post("/search", json={"query": "cheese", "model": "minilm_finetuned"})
    assert response.status_code == 503 and "different checkpoint" in response.json()["detail"]


def test_tampered_index_corpus_fingerprint_is_refused(client_for, env):
    bad = env["tmp"] / "emb-bad"
    shutil.copytree(env["b"][1], bad)
    meta = json.loads((bad / "metadata.json").read_text("utf-8"))
    meta["corpus_fingerprint"] = "0" * 64
    (bad / "metadata.json").write_text(json.dumps(meta), "utf-8")
    client = client_for(model_b_embeddings_dir=bad)
    response = client.post("/search", json={"query": "cheese", "model": "minilm_finetuned"})
    assert response.status_code == 503 and "corpus" in response.json()["detail"].lower()


def test_invalid_model_and_request_are_rejected(client_for):
    client = client_for()
    assert client.post("/search", json={"query": "x", "model": "model_d"}).status_code == 422
    assert client.post("/search", json={"query": "  ", "model": "minilm_finetuned"}).status_code == 422
    assert client.post("/search", json={"query": "x", "model": "bm25", "top_k": 0}).status_code == 422
    assert client.post("/explain", json={"query": "x", "top_k": 6}).status_code == 422


def test_identifier_checks_accompany_every_result(client_for):
    client = client_for()
    plain = client.post("/search", json={"query": "Generated test recall", "model": "bm25"}).json()
    assert plain["results"], "fixture query must return results"
    assert plain["identifiers_in_query"] is False and "not a determination" in plain["notice"]
    assert [c["recall_id"] for c in plain["identifier_checks"]] == [h["recall_id"] for h in plain["results"]]
    assert all(c["verdict"] == "UNKNOWN" for c in plain["identifier_checks"])
    coded = client.post("/search", json={"query": "Generated test recall lot 036000291452",
                                         "model": "bm25"}).json()
    assert coded["identifiers_in_query"] is True
    first = coded["identifier_checks"][0]["by_type"]
    assert first["lot"]["query_values"] == ["036000291452"] and first["upc"]["query_values"] == []


def test_explain_returns_grounded_template_answer(client_for):
    response = client_for().post("/explain", json={"query": "Generated test recall", "model": "minilm_finetuned"})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["evidence_status"] == "evidence_found"
    assert body["explanation"]["generator"] == "template"
    assert body["sources"] == [f["url"] for f in body["retrieved_facts"]]
    assert set(body["explanation"]["cited_recall_ids"]) <= {f["recall_id"] for f in body["retrieved_facts"]}
    assert str(body["retrieved_facts"][0]["recall_id"]) in body["explanation"]["text"]
    assert "not confirmation" in body["disclaimer"]


def test_explain_without_results_says_so(client_for):
    body = client_for().post("/explain", json={"query": "zzqx", "model": "bm25"}).json()
    assert body["evidence_status"] == "no_results" and body["retrieved_facts"] == []
    assert "No recall notice" in body["explanation"]["text"] and body["sources"] == []


# --- real trained checkpoints (skipped until trained and indexed) ------------------------

REAL = Settings()
REAL_READY = REAL.database_path.is_file() and all(
    (ckpt / "model.safetensors").is_file() and (index / "metadata.json").is_file()
    for ckpt, index in REAL.finetuned_models.values())


@pytest.mark.skipif(not REAL_READY, reason="trained checkpoints / their indexes not built")
@pytest.mark.parametrize("model", ["minilm_finetuned", "minilm_hard_negative"])
def test_real_trained_model_search_and_explain(model):
    app.dependency_overrides.clear()
    search_service._cache.clear()
    with TestClient(app) as real:
        response = real.post("/search", json={"query": "cheese with listeria", "model": model,
                                              "top_k": 5})
        assert response.status_code == 200, response.text
        results = response.json()["results"]
        assert len(results) == 5
        assert all(r["url"].startswith("https://recalls-rappels.canada.ca/") for r in results)
        again = real.post("/search", json={"query": "cheese with listeria", "model": model,
                                           "top_k": 5}).json()["results"]
        assert again == results
        explained = real.post("/explain", json={"query": "cheese with listeria", "model": model})
        assert explained.status_code == 200 and explained.json()["sources"]
    search_service._cache.clear()


# --- regressions from the ecc:fastapi-reviewer pass (2026-11-16) -------------------------

def test_corrupt_checkpoint_load_is_a_model_unavailable_error_not_a_crash(tmp_path, env):
    from retrieval.finetuned import CheckpointEncoder
    from retrieval.semantic import ModelUnavailableError

    broken = tmp_path / "broken"
    shutil.copytree(env["b"][0], broken)
    (broken / "model.safetensors").write_bytes(b"not a safetensors file")
    encoder = CheckpointEncoder(broken)
    for _ in range(2):  # second call: cached failure, fails fast
        with pytest.raises(ModelUnavailableError):
            encoder.encode(["cheese"])


def test_inference_failure_becomes_503(client_for, monkeypatch):
    from retrieval.finetuned import CheckpointEncoder

    client = client_for()

    class Exploding:
        def encode(self, texts, batch_size=64):
            raise RuntimeError("simulated torch failure")

    monkeypatch.setattr(CheckpointEncoder, "_load", lambda self: Exploding())
    response = client.post("/search", json={"query": "Generated test recall", "model": "minilm_finetuned"})
    assert response.status_code == 503 and "failed" in response.json()["detail"]


def test_checkpoint_is_loaded_eagerly_when_the_engine_is_built(env):
    from retrieval.loader import build_engine

    engine = build_engine(env["db"], env["minilm"], finetuned={"minilm_finetuned": env["b"]})
    _, encoder, error = engine.dense["minilm_finetuned"]
    assert error is None and encoder._model is not None   # loaded right after the hash check
