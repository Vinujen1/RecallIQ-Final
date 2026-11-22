import json
import sqlite3
from pathlib import Path

import pytest

from conftest import (
    ELIGIBLE_HAND_WRITTEN,
    EXCLUDED_HAND_WRITTEN,
    GENERATED_COUNT,
    HAND_WRITTEN_ROWS,
    write_csv,
)
from recall_data.build import DEFAULT_OUTPUT_DIR, run
from recall_data.clean import filter_eligible, normalize
from recall_data.grouping import add_product_families, add_title_groups, assign_family, title_group_id
from recall_data.load import RAW_COLUMNS, DataValidationError, load_raw
from recall_data.sample import allocate, stratified_sample
from recall_data.storage import to_records, write_sqlite


def _prepared(path: Path):
    return add_product_families(add_title_groups(normalize(filter_eligible(load_raw(path)))))


# --- loading ---------------------------------------------------------------------------

def test_load_raw_keeps_every_row_and_column(raw_csv):
    frame = load_raw(raw_csv)
    assert len(frame) == len(HAND_WRITTEN_ROWS) + GENERATED_COUNT
    assert tuple(frame.columns) == RAW_COLUMNS
    assert (frame == "").any().any()  # empty cells are kept as "", not NaN


def test_load_raw_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_raw(tmp_path / "nope.csv")


def test_load_raw_rejects_missing_columns(tmp_path):
    path = write_csv(tmp_path / "bad.csv", HAND_WRITTEN_ROWS, columns=RAW_COLUMNS[:-1])
    with pytest.raises(DataValidationError, match="Archived"):
        load_raw(path)


def test_load_raw_rejects_duplicate_ids(tmp_path):
    rows = HAND_WRITTEN_ROWS + [{**HAND_WRITTEN_ROWS[0], "Title": "Another title"}]
    with pytest.raises(DataValidationError, match="Duplicate NID"):
        load_raw(write_csv(tmp_path / "dup.csv", rows))


def test_load_raw_rejects_non_numeric_ids(tmp_path):
    rows = [{**HAND_WRITTEN_ROWS[0], "NID": "abc"}]
    with pytest.raises(DataValidationError, match="Non-numeric"):
        load_raw(write_csv(tmp_path / "nan.csv", rows))


# --- filtering & normalization ---------------------------------------------------------

def test_filter_keeps_only_food_and_consumer_products(small_csv):
    raw = load_raw(small_csv)
    eligible = filter_eligible(raw)
    assert len(eligible) == ELIGIBLE_HAND_WRITTEN
    assert len(raw) - len(eligible) == EXCLUDED_HAND_WRITTEN
    assert set(eligible["Organization"]) == {"CFIA", "Consumer product safety"}


def test_normalize_handles_missing_values_without_inventing_data(small_csv):
    records = {r["id"]: r for r in to_records(_prepared(small_csv))}

    assert records[2]["title"] == "Undeclared milk in Test Brand B Cookies"  # whitespace only
    assert records[2]["recall_class"] is None          # "" -> None
    assert records[2]["last_updated"] is None          # "" -> None
    assert records[4]["recall_class"] is None          # "--" placeholder -> None
    assert records[5]["last_updated"] is None          # unparseable -> None, not guessed
    assert records[1]["last_updated"] == "2024-01-02"
    assert records[2]["archived"] is True and records[1]["archived"] is False
    assert records[1]["domain"] == "food" and records[4]["domain"] == "consumer_product"


def test_normalize_preserves_original_fields(small_csv):
    records = {r["id"]: r for r in to_records(_prepared(small_csv))}
    assert records[1]["url"] == "https://example.test/recall/1"
    assert records[1]["issue"] == "Microbiological - Listeria"
    assert records[4]["category"] == "Toys and games - Household items"
    assert records[4]["categories"] == ["Toys and games", "Household items"]


# --- grouping --------------------------------------------------------------------------

def test_exact_duplicate_titles_share_a_stable_group(small_csv):
    records = {r["id"]: r for r in to_records(_prepared(small_csv))}
    assert records[2]["title_group"] == records[3]["title_group"]
    assert records[2]["title_group_size"] == 2
    assert records[1]["title_group"] != records[2]["title_group"]
    assert records[2]["title_group"] == title_group_id("Undeclared milk in Test Brand B Cookies")


def test_title_group_is_case_sensitive_exact_match():
    assert title_group_id("Cheese recall") != title_group_id("cheese recall")


@pytest.mark.parametrize(
    ("title", "product", "categories", "domain", "family", "confidence"),
    [
        ("X recalled", None, ["Dairy"], "food", "Dairy", "high"),
        ("X recalled", None, ["Household items", "Toys and games"], "consumer_product",
         "Household items", "medium"),
        ("Test Brand Cheddar Cheese recalled due to Listeria", None, ["Food"], "food",
         "Dairy", "medium"),
        # the allergen named in the hazard clause must not decide the family
        ("Undeclared milk in Test Brand B Cookies", None, ["Food"], "food",
         "Candy, confectionary, snacks and sweeteners", "medium"),
        ("Test widget recalled", None, ["Other"], "consumer_product", None, "low"),
        ("Shrimp Fried Rice recalled", None, ["Food"], "food", None, "low"),
    ],
)
def test_assign_family(title, product, categories, domain, family, confidence):
    result = assign_family(title, product, categories, domain)
    assert result["product_family"] == family
    assert result["family_confidence"] == confidence
    assert result["needs_review"] is (confidence != "high")
    assert (result["review_reason"] is None) is (confidence == "high")


# --- sampling --------------------------------------------------------------------------

def test_allocate_sums_to_target_and_respects_stratum_size():
    sizes = {"a": 50, "b": 30, "c": 3}
    allocation = allocate(sizes, 10)
    assert sum(allocation.values()) == 10
    assert all(allocation[k] <= size for k, size in sizes.items())
    assert allocate({"a": 2, "b": 1}, 10) == {"a": 2, "b": 1}


def test_sample_is_deterministic_unique_and_deduplicated(raw_csv):
    records = _prepared(raw_csv)
    first, report = stratified_sample(records, size=20, seed=7)
    second, _ = stratified_sample(records, size=20, seed=7)
    other, _ = stratified_sample(records, size=20, seed=8)

    assert len(first) == 20
    assert first["id"].tolist() == second["id"].tolist()
    assert first["id"].tolist() != other["id"].tolist()
    assert first["id"].is_unique
    assert first["title_group"].is_unique
    assert sum(s["selected"] for s in report.values()) == 20


def test_sample_keeps_latest_member_of_duplicate_title_group(small_csv):
    sample, _ = stratified_sample(_prepared(small_csv), size=100, seed=1)
    ids = set(sample["id"])
    assert 3 in ids and 2 not in ids  # id 3 is dated 2024-05-01; id 2 has no date


def test_sample_rejects_non_positive_size(small_csv):
    with pytest.raises(ValueError):
        stratified_sample(_prepared(small_csv), size=0)


# --- export & database import ----------------------------------------------------------

def test_run_writes_jsonl_sqlite_and_manifest(raw_csv, tmp_path):
    out = tmp_path / "out"
    manifest = run(raw_csv, out, size=30, seed=3)

    assert manifest["raw_records"] == len(HAND_WRITTEN_ROWS) + GENERATED_COUNT
    assert manifest["eligible_records"] == ELIGIBLE_HAND_WRITTEN + GENERATED_COUNT
    assert manifest["selected_records"] == 30

    lines = (out / "recalls_sample.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 30
    assert all(json.loads(line)["id"] for line in lines)

    db = sqlite3.connect(out / "recalliq.db")
    try:
        assert db.execute("SELECT COUNT(*) FROM recalls").fetchone()[0] == 30
        assert db.execute("SELECT COUNT(DISTINCT id) FROM recalls").fetchone()[0] == 30
        orphans = db.execute(
            "SELECT COUNT(*) FROM recall_categories rc LEFT JOIN recalls r ON r.id = rc.recall_id "
            "WHERE r.id IS NULL"
        ).fetchone()[0]
        assert orphans == 0
        assert db.execute("SELECT value FROM metadata WHERE key = 'seed'").fetchone()[0] == "3"
    finally:
        db.close()


def test_sqlite_import_rejects_duplicate_ids_and_keeps_existing_db(small_csv, tmp_path):
    records = to_records(_prepared(small_csv))
    db_path = tmp_path / "recalliq.db"
    write_sqlite(records, db_path, metadata={"version": "good"})

    with pytest.raises(sqlite3.IntegrityError):
        write_sqlite(records + [records[0]], db_path, metadata={"version": "bad"})

    db = sqlite3.connect(db_path)
    try:
        assert db.execute("SELECT COUNT(*) FROM recalls").fetchone()[0] == len(records)
        assert db.execute("SELECT value FROM metadata").fetchone()[0] == "good"
    finally:
        db.close()
    assert not db_path.with_suffix(".db.tmp").exists()


def test_locked_database_raises_clear_error_and_cleans_temp_file(small_csv, tmp_path, monkeypatch):
    from recall_data import storage

    def locked(*_args):
        raise PermissionError(13, "The process cannot access the file")

    monkeypatch.setattr(storage.os, "replace", locked)
    db_path = tmp_path / "recalliq.db"
    with pytest.raises(storage.OutputLockedError, match="stop it and run the pipeline again"):
        write_sqlite(to_records(_prepared(small_csv)), db_path, metadata={})
    assert not db_path.exists()
    assert not db_path.with_suffix(".db.tmp").exists()


def test_run_fails_clearly_when_nothing_is_eligible(tmp_path):
    rows = [r for r in HAND_WRITTEN_ROWS if r["Organization"] not in {"CFIA", "Consumer product safety"}]
    with pytest.raises(ValueError, match="No eligible records"):
        run(write_csv(tmp_path / "none.csv", rows), tmp_path / "out")


def test_unexpected_archived_value_fails(tmp_path):
    rows = [{**HAND_WRITTEN_ROWS[0], "Archived": "yes"}]
    with pytest.raises(ValueError, match="Archived"):
        normalize(filter_eligible(load_raw(write_csv(tmp_path / "a.csv", rows))))


def test_text_normalization_removes_invisible_characters_only():
    from recall_data.clean import normalize_text

    assert normalize_text("Brand\u200b X\xa0 Cookies ") == "Brand X Cookies"
    assert normalize_text("Café – Crème") == "Café – Crème"


# --- real dataset (skipped until the pipeline has been run) ----------------------------

REAL_MANIFEST = DEFAULT_OUTPUT_DIR / "manifest.json"


@pytest.mark.skipif(not REAL_MANIFEST.is_file(), reason="real pipeline output not built")
def test_real_pipeline_output_is_consistent():
    manifest = json.loads(REAL_MANIFEST.read_text(encoding="utf-8"))
    assert manifest["selected_records"] == manifest["target_size"]
    assert manifest["eligible_records"] <= manifest["raw_records"]

    db = sqlite3.connect(DEFAULT_OUTPUT_DIR / "recalliq.db")
    try:
        count, distinct = db.execute("SELECT COUNT(*), COUNT(DISTINCT id) FROM recalls").fetchone()
        domains = {row[0] for row in db.execute("SELECT DISTINCT domain FROM recalls")}
    finally:
        db.close()
    assert count == distinct == manifest["selected_records"]
    assert domains == {"food", "consumer_product"}
