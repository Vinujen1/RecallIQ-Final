"""Shared fixtures.

The CSV built here is SYNTHETIC test data that mimics the structure of the official
HCRSAMOpenData.csv (same column names). It is written to a pytest temp directory only and
is never mixed with the real dataset in data/.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from recall_data.load import RAW_COLUMNS

HAND_WRITTEN_ROWS: list[dict[str, str]] = [
    # id 1: food, specific official category
    {"NID": "1", "Title": "Test Brand A Cheese recalled due to Listeria", "Organization": "CFIA",
     "Category": "Dairy", "Recall class": "Class 1", "Last updated": "2024-01-02", "Archived": "0",
     "Product": "Cheese", "Issue": "Microbiological - Listeria"},
    # ids 2 & 3: exact duplicate titles (after whitespace normalization), generic category
    {"NID": "2", "Title": "  Undeclared milk in Test Brand B Cookies  ", "Organization": "CFIA",
     "Category": "Food", "Recall class": "", "Last updated": "", "Archived": "1"},
    {"NID": "3", "Title": "Undeclared milk in Test Brand B   Cookies", "Organization": "CFIA",
     "Category": "Food", "Recall class": "Class 2", "Last updated": "2024-05-01", "Archived": "0"},
    # id 4: consumer product, multi-valued category, "--" placeholder class
    {"NID": "4", "Title": "Test Company recalls toy blocks", "Organization": "Consumer product safety",
     "Category": "Toys and games - Household items", "Recall class": "--",
     "Last updated": "2023-07-15", "Archived": "0"},
    # id 5: unparseable date, generic category, no keyword match
    {"NID": "5", "Title": "Test widget recalled", "Organization": "Consumer product safety",
     "Category": "Other", "Recall class": "", "Last updated": "not-a-date", "Archived": "1"},
    # ids 6-8: organizations that must be excluded
    {"NID": "6", "Title": "Test SUV airbag", "Organization": "TC", "Category": "SUV",
     "Last updated": "2024-02-02", "Archived": "0"},
    {"NID": "7", "Title": "Test infusion pump", "Organization": "Medical devices",
     "Category": "Medical devices", "Last updated": "2024-02-02", "Archived": "0"},
    {"NID": "8", "Title": "Test tablet recall", "Organization": "Drugs and health products",
     "Category": "Drugs", "Last updated": "2024-02-02", "Archived": "0"},
]

ELIGIBLE_HAND_WRITTEN = 5  # ids 1-5
EXCLUDED_HAND_WRITTEN = 3  # ids 6-8


def _generated_rows(count: int) -> list[dict[str, str]]:
    """Bulk eligible rows across a few strata, for sampling tests."""
    plan = [("CFIA", "Dairy"), ("CFIA", "Fresh"), ("Consumer product safety", "Toys and games"),
            ("Consumer product safety", "Electronics")]
    rows = []
    for i in range(count):
        organization, category = plan[i % len(plan)]
        nid = 100 + i
        rows.append({
            "NID": str(nid), "Title": f"Generated test recall {nid}", "Organization": organization,
            "Category": category, "Recall class": "Class 3" if organization == "CFIA" else "",
            "Last updated": f"2022-{(i % 12) + 1:02d}-{(i % 27) + 1:02d}", "Archived": str(i % 2),
        })
    return rows


GENERATED_COUNT = 80


def write_csv(path: Path, rows: list[dict[str, str]], columns=RAW_COLUMNS) -> Path:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            full = {column: "" for column in columns}
            full["URL"] = f"https://example.test/recall/{row['NID']}"
            full.update({k: v for k, v in row.items() if k in columns})
            writer.writerow(full)
    return path


class FakeEncoder:
    """Deterministic stand-in for MiniLM in unit tests (hashed bag of words, unit length).

    It has none of MiniLM's semantics; it only lets tests exercise indexing, id mapping
    and cosine ranking without downloading a model.
    """

    dimension = 16

    def encode(self, texts: list[str]) -> "np.ndarray":
        import zlib

        import numpy as np

        from retrieval.bm25 import tokenize

        matrix = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            matrix[row, 0] = 0.1  # bias so no vector is all-zero
            for token in tokenize(text):
                matrix[row, 1 + zlib.crc32(token.encode()) % (self.dimension - 1)] += 1.0
        return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


def build_fake_embeddings(db_path: Path, out_dir: Path) -> Path:
    """Write embedding artifacts for the fixture DB using FakeEncoder."""
    from retrieval.corpus import corpus_fingerprint, load_corpus
    from retrieval.semantic import save_embeddings

    corpus = load_corpus(db_path)
    matrix = FakeEncoder().encode([r.search_text for r in corpus])
    save_embeddings(
        out_dir,
        [r.recall_id for r in corpus],
        matrix,
        {"model_id": "fake", "model_revision": None, "dimension": FakeEncoder.dimension,
         "corpus_fingerprint": corpus_fingerprint(corpus)},
    )
    return out_dir


#: Evaluation fixture rows (SYNTHETIC): pairs with known grouping behaviour.
NEAR_DUPLICATE_ROWS = [
    # same product, Jaccard 0.75 -> merged
    {"NID": "9001", "Title": "Frozen Fish Cake recalled due to undeclared egg",
     "Organization": "CFIA", "Category": "Fish and seafood"},
    {"NID": "9002", "Title": "ChoripDong brand Frozen Fish Cake recalled due to undeclared egg",
     "Organization": "CFIA", "Category": "Fish and seafood"},
    # same brand, Jaccard 0.5 -> merged by the brand rule
    {"NID": "9003", "Title": "Ruffles brand Regular Potato Chips recalled due to undeclared milk",
     "Organization": "CFIA", "Category": "Candy, confectionary, snacks and sweeteners"},
    {"NID": "9004", "Title": "Ruffles brand Hot Wings Potato Chips recalled due to undeclared milk",
     "Organization": "CFIA", "Category": "Candy, confectionary, snacks and sweeteners"},
    # different brands, Jaccard 0.5 -> ambiguous, NOT merged
    {"NID": "9005", "Title": "Coaticook brand Cheddar Cheese recalled due to Listeria",
     "Organization": "CFIA", "Category": "Dairy"},
    {"NID": "9006", "Title": "Avonlea brand Cheddar Cheese recalled due to Listeria",
     "Organization": "CFIA", "Category": "Dairy"},
]


def _eval_singletons(count: int) -> list[dict[str, str]]:
    plan = [("CFIA", "Dairy"), ("Consumer product safety", "Toys and games"),
            ("CFIA", "Fresh"), ("Consumer product safety", "Electronics")]
    return [
        {"NID": str(9100 + i), "Title": f"Zq{i} widget{i}", "Organization": plan[i % 4][0],
         "Category": plan[i % 4][1]}
        for i in range(count)
    ]


@pytest.fixture
def eval_env(tmp_path: Path) -> dict:
    """Synthetic corpus (66 records) + DB + split manifest for evaluation tests."""
    from evaluation.splits import build_manifest, write_manifest
    from recall_data.build import run

    rows = [{**r, "Last updated": "2024-01-01", "Archived": "0"}
            for r in NEAR_DUPLICATE_ROWS + _eval_singletons(60)]
    out = tmp_path / "processed"
    run(write_csv(tmp_path / "eval.csv", rows), out, size=1000, seed=1)  # keep all
    db = out / "recalliq.db"
    manifest_path = out / "splits" / "splits.json"
    manifest = build_manifest(db, seed=4110)
    write_manifest(manifest, manifest_path)
    return {"db": db, "manifest_path": manifest_path, "manifest": manifest, "tmp": tmp_path}


@pytest.fixture
def raw_csv(tmp_path: Path) -> Path:
    return write_csv(tmp_path / "fixture.csv", HAND_WRITTEN_ROWS + _generated_rows(GENERATED_COUNT))


@pytest.fixture
def small_csv(tmp_path: Path) -> Path:
    return write_csv(tmp_path / "small.csv", HAND_WRITTEN_ROWS)
