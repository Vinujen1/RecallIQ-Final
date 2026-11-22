import pytest
from fastapi.testclient import TestClient

from app import search_service
from app.config import Settings, get_settings
from app.main import app
from app.search_service import get_search_engine
from conftest import FakeEncoder, build_fake_embeddings
from recall_data.build import run
from retrieval.embed import DEFAULT_DB, DEFAULT_OUT
from retrieval.loader import build_engine


@pytest.fixture
def paths(raw_csv, tmp_path):
    out = tmp_path / "processed"
    run(raw_csv, out, size=30, seed=11)
    db = out / "recalliq.db"
    return db, build_fake_embeddings(db, tmp_path / "emb")


@pytest.fixture
def client(paths):
    db, emb = paths
    engine = build_engine(db, emb, encoder=FakeEncoder())
    app.dependency_overrides[get_settings] = lambda: Settings(database_path=db, embeddings_dir=emb)
    app.dependency_overrides[get_search_engine] = lambda: engine
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.fixture
def client_without_embeddings(paths, tmp_path):
    db, _ = paths
    search_service._cache.clear()
    app.dependency_overrides[get_settings] = lambda: Settings(
        database_path=db, embeddings_dir=tmp_path / "missing"
    )
    yield TestClient(app)
    app.dependency_overrides.clear()
    search_service._cache.clear()


def _post(client, **body):
    return client.post("/search", json=body)


def test_bm25_search_response_shape_and_labels(client):
    response = _post(client, query="Generated test recall", model="bm25", top_k=5)
    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "bm25" and body["score_type"] == "bm25"
    assert "not comparable with cosine" in body["score_description"]
    assert 1 <= len(body["results"]) <= 5
    hit = body["results"][0]
    assert set(hit) == {"rank", "recall_id", "title", "category", "issue", "url", "score",
                        "domain", "last_updated", "archived"}
    assert body["domain"] is None and body["category"] is None
    scores = [h["score"] for h in body["results"]]
    assert scores == sorted(scores, reverse=True)
    assert [h["rank"] for h in body["results"]] == list(range(1, len(scores) + 1))


def test_results_map_to_existing_recall_records(client):
    hits = _post(client, query="Generated test recall", model="bm25", top_k=10).json()["results"]
    assert hits
    for hit in hits:
        record = client.get(f"/recalls/{hit['recall_id']}").json()
        assert (record["title"], record["url"]) == (hit["title"], hit["url"])


def test_minilm_search_uses_cosine_labels(client):
    body = _post(client, query="cheese listeria", model="minilm", top_k=7).json()
    assert body["score_type"] == "cosine_similarity"
    assert "[-1, 1]" in body["score_description"]
    assert len(body["results"]) == 7
    assert all(-1.0 <= h["score"] <= 1.0 for h in body["results"])


def test_defaults_are_bm25_and_top_10(client):
    body = _post(client, query="Generated test recall").json()
    assert (body["model"], body["top_k"]) == ("bm25", 10)
    assert len(body["results"]) == 10


@pytest.mark.parametrize("query", ["zzzz qqqq", "the and of", "a a a"])
def test_query_without_usable_terms_returns_empty_bm25_results(client, query):
    response = _post(client, query=query, model="bm25")
    assert response.status_code == 200
    assert response.json()["results"] == []


def test_empty_recall_table_returns_503_not_500(paths, tmp_path):
    import shutil
    import sqlite3

    db, emb = paths
    empty = tmp_path / "empty.db"
    shutil.copy(db, empty)
    connection = sqlite3.connect(empty)
    connection.execute("DELETE FROM recall_categories")
    connection.execute("DELETE FROM recalls")
    connection.commit()
    connection.close()
    search_service._cache.clear()
    app.dependency_overrides[get_settings] = lambda: Settings(database_path=empty, embeddings_dir=emb)
    try:
        response = _post(TestClient(app), query="cheese")
        assert response.status_code == 503
        assert "could not be built" in response.json()["detail"]
    finally:
        app.dependency_overrides.clear()
        search_service._cache.clear()


@pytest.mark.parametrize(
    "body",
    [
        {"query": ""},
        {"query": "   "},
        {"query": "x" * 501},
        {},
        {"query": "cheese", "model": "tfidf"},
        {"query": "cheese", "model": "MINILM"},
        {"query": "cheese", "top_k": 0},
        {"query": "cheese", "top_k": 51},
        {"query": "cheese", "top_k": "ten"},
    ],
)
def test_invalid_requests_are_rejected(client, body):
    assert client.post("/search", json=body).status_code == 422


def test_search_requires_post(client):
    assert client.get("/search").status_code == 405


def test_minilm_returns_503_when_embeddings_missing_but_bm25_works(client_without_embeddings):
    minilm = _post(client_without_embeddings, query="cheese", model="minilm")
    assert minilm.status_code == 503
    assert "retrieval.embed" in minilm.json()["detail"]
    assert _post(client_without_embeddings, query="Generated", model="bm25").status_code == 200


def test_missing_database_returns_503(tmp_path):
    search_service._cache.clear()
    app.dependency_overrides[get_settings] = lambda: Settings(
        database_path=tmp_path / "none.db", embeddings_dir=tmp_path
    )
    try:
        assert _post(TestClient(app), query="cheese").status_code == 503
    finally:
        app.dependency_overrides.clear()


# --- filters and per-hit metadata (Phase 5) ---------------------------------------------

def _categories(client):
    return client.get("/categories").json()["items"]


def test_hits_carry_domain_date_and_archived_from_the_record(client):
    hits = _post(client, query="Generated test recall", top_k=10).json()["results"]
    assert hits
    for hit in hits:
        record = client.get(f"/recalls/{hit['recall_id']}").json()
        assert (hit["domain"], hit["last_updated"], hit["archived"]) == (
            record["domain"], record["last_updated"], record["archived"])


@pytest.mark.parametrize("model", ["bm25", "minilm"])
def test_domain_filter_is_applied_before_top_k(client, model):
    for domain in ("food", "consumer_product"):
        body = _post(client, query="Generated test recall", model=model, top_k=5,
                     domain=domain).json()
        assert body["domain"] == domain
        assert body["results"], domain
        assert all(h["domain"] == domain for h in body["results"])
        if model == "minilm":  # filled from allowed records, not post-filtered from a top 5
            assert len(body["results"]) == 5
        assert [h["rank"] for h in body["results"]] == list(range(1, len(body["results"]) + 1))


@pytest.mark.parametrize("model", ["bm25", "minilm"])
def test_category_filter_matches_any_listed_category(client, model):
    category = _categories(client)[0]
    body = _post(client, query="Generated test recall", model=model, top_k=50,
                 category=category["category"]).json()
    assert body["category"] == category["category"]
    assert 0 < len(body["results"]) <= category["count"]
    for hit in body["results"]:
        record = client.get(f"/recalls/{hit['recall_id']}").json()
        assert category["category"] in record["categories"]
    if model == "minilm":  # dense models rank every allowed record
        assert len(body["results"]) == category["count"]


def test_filters_combine_and_unknown_category_returns_no_results(client):
    assert _post(client, query="Generated test recall", category="No such category").json()[
        "results"] == []
    assert _post(client, query="Generated test recall", domain="food",
                 category="   ").json()["category"] is None  # blank = no filter


def test_filtered_ranking_preserves_unfiltered_order(client):
    full = _post(client, query="Generated test recall", model="minilm", top_k=50).json()["results"]
    food = _post(client, query="Generated test recall", model="minilm", top_k=50,
                 domain="food").json()["results"]
    assert [h["recall_id"] for h in food] == [h["recall_id"] for h in full if h["domain"] == "food"]


@pytest.mark.parametrize("body", [
    {"query": "x", "domain": "drugs"},
    {"query": "x", "category": "c" * 201},
])
def test_invalid_filters_are_rejected(client, body):
    assert _post(client, **body).status_code == 422


def test_explain_accepts_the_same_filters(client):
    body = client.post("/explain", json={"query": "Generated test recall", "top_k": 3,
                                         "domain": "food"}).json()
    ids = [f["recall_id"] for f in body["retrieved_facts"]]
    assert ids and all(client.get(f"/recalls/{i}").json()["domain"] == "food" for i in ids)


# --- real data (skipped until DB + embeddings are built) -------------------------------

REAL_READY = DEFAULT_DB.is_file() and (DEFAULT_OUT / "metadata.json").is_file()


@pytest.mark.skipif(not REAL_READY, reason="real database / embeddings not built")
@pytest.mark.parametrize("model", ["bm25", "minilm"])
def test_real_search_endpoint(model):
    search_service._cache.clear()
    with TestClient(app) as real:
        response = real.post("/search", json={"query": "cheese with listeria", "model": model,
                                              "top_k": 5})
    assert response.status_code == 200
    results = response.json()["results"]
    assert len(results) == 5
    assert all(r["url"].startswith("https://recalls-rappels.canada.ca/") for r in results)
