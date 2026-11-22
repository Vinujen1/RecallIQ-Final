from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import Settings, get_settings
from app.main import app
from recall_data.build import run

TOTAL = 30


@pytest.fixture
def client(raw_csv, tmp_path):
    out = tmp_path / "processed"
    run(raw_csv, out, size=TOTAL, seed=11)
    app.dependency_overrides[get_settings] = lambda: Settings(database_path=out / "recalliq.db")
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.fixture
def client_without_db(tmp_path):
    app.dependency_overrides[get_settings] = lambda: Settings(database_path=tmp_path / "none.db")
    yield TestClient(app)
    app.dependency_overrides.clear()


def _all_ids(client: TestClient) -> list[int]:
    ids, offset = [], 0
    while True:
        page = client.get("/recalls", params={"limit": 7, "offset": offset}).json()
        ids += [item["id"] for item in page["items"]]
        offset += 7
        if offset >= page["total"]:
            return ids


# --- /recalls --------------------------------------------------------------------------

def test_list_defaults(client):
    response = client.get("/recalls")
    assert response.status_code == 200
    body = response.json()
    assert body["total"] == TOTAL
    assert (body["limit"], body["offset"]) == (20, 0)
    assert len(body["items"]) == 20
    item = body["items"][0]
    assert {"id", "title", "url", "domain", "categories", "archived", "last_updated"} <= item.keys()
    assert isinstance(item["categories"], list)


def test_pagination_pages_do_not_overlap_and_match_a_single_query(client):
    first = client.get("/recalls", params={"limit": 5, "offset": 0}).json()["items"]
    second = client.get("/recalls", params={"limit": 5, "offset": 5}).json()["items"]
    combined = client.get("/recalls", params={"limit": 10, "offset": 0}).json()["items"]

    assert {i["id"] for i in first}.isdisjoint({i["id"] for i in second})
    assert [i["id"] for i in first + second] == [i["id"] for i in combined]


def test_pagination_covers_every_record_exactly_once(client):
    ids = _all_ids(client)
    assert len(ids) == TOTAL
    assert len(set(ids)) == TOTAL  # no duplicate ids across pages


def test_listing_is_newest_first_with_missing_dates_last(client):
    items = client.get("/recalls", params={"limit": 100}).json()["items"]
    dates = [i["last_updated"] for i in items]
    dated = [d for d in dates if d is not None]
    assert dated == sorted(dated, reverse=True)
    assert dates[: len(dated)] == dated


def test_offset_past_end_returns_empty_page(client):
    body = client.get("/recalls", params={"offset": 1000}).json()
    assert body["items"] == [] and body["total"] == TOTAL


@pytest.mark.parametrize(
    "params",
    [{"limit": 0}, {"limit": 101}, {"offset": -1}, {"limit": "abc"},
     {"offset": 99999999999999999999}],  # beyond SQLite's 64-bit range: 422, not 500
)
def test_invalid_pagination_is_rejected(client, params):
    assert client.get("/recalls", params=params).status_code == 422


def test_recalls_is_read_only(client):
    assert client.post("/recalls").status_code == 405


# --- /recalls/{id} ---------------------------------------------------------------------

def test_get_recall_returns_full_record(client):
    recall_id = client.get("/recalls", params={"limit": 1}).json()["items"][0]["id"]
    response = client.get(f"/recalls/{recall_id}")
    assert response.status_code == 200
    body = response.json()
    assert body["id"] == recall_id
    assert body["url"] == f"https://example.test/recall/{recall_id}"
    assert body["family_confidence"] in {"high", "medium", "low"}
    assert body["title_group"].startswith("tg_")


def test_get_recall_not_found(client):
    assert client.get("/recalls/999999").status_code == 404


@pytest.mark.parametrize("bad_id", ["abc", "0", "-5", "99999999999999999999"])
def test_get_recall_invalid_id(client, bad_id):
    assert client.get(f"/recalls/{bad_id}").status_code == 422


# --- /categories -----------------------------------------------------------------------

def test_categories_counts_match_records(client):
    body = client.get("/categories").json()
    assert body["total"] == len(body["items"]) > 0
    assert all(item["domain"] in {"food", "consumer_product"} for item in body["items"])
    counts = [item["count"] for item in body["items"]]
    assert counts == sorted(counts, reverse=True)

    listed = client.get("/recalls", params={"limit": 100}).json()["items"]
    assert sum(counts) == sum(len(item["categories"]) for item in listed)


# --- availability ----------------------------------------------------------------------

def test_missing_database_returns_503_but_health_still_works(client_without_db):
    assert client_without_db.get("/recalls").status_code == 503
    assert client_without_db.get("/categories").status_code == 503
    assert client_without_db.get("/health").status_code == 200


# --- real dataset (skipped until the pipeline has been run) ----------------------------

REAL_DB = Settings().database_path


@pytest.mark.skipif(not Path(REAL_DB).is_file(), reason="real database not built")
def test_real_database_serves_selected_records():
    with TestClient(app) as real_client:
        body = real_client.get("/recalls", params={"limit": 5}).json()
        assert body["total"] == 1000
        first = body["items"][0]
        assert first["url"].startswith("https://recalls-rappels.canada.ca/")
        assert real_client.get(f"/recalls/{first['id']}").status_code == 200
