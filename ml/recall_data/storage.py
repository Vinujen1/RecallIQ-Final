"""Export cleaned records to JSONL and to a SQLite database (rebuilt atomically)."""

from __future__ import annotations

import json
import os
import sqlite3
from collections.abc import Iterable, Mapping
from pathlib import Path

import pandas as pd

RECORD_FIELDS: tuple[str, ...] = (
    "id", "title", "url", "organization", "domain", "category", "categories", "product",
    "issue", "what_you_should_do", "recall_class", "last_updated", "archived",
    "title_group", "title_group_size", "product_family", "family_source",
    "family_confidence", "needs_review", "review_reason",
)

SCHEMA = """
CREATE TABLE recalls (
    id                 INTEGER PRIMARY KEY,           -- official NID
    title              TEXT    NOT NULL,
    url                TEXT    NOT NULL,
    organization       TEXT    NOT NULL,
    domain             TEXT    NOT NULL CHECK (domain IN ('food', 'consumer_product')),
    category           TEXT,                          -- original (possibly multi-valued)
    product            TEXT,
    issue              TEXT,
    what_you_should_do TEXT,
    recall_class       TEXT,
    last_updated       TEXT,                          -- ISO yyyy-mm-dd or NULL
    archived           INTEGER NOT NULL CHECK (archived IN (0, 1)),
    title_group        TEXT    NOT NULL,
    title_group_size   INTEGER NOT NULL CHECK (title_group_size >= 1),
    product_family     TEXT,
    family_source      TEXT    NOT NULL,
    family_confidence  TEXT    NOT NULL CHECK (family_confidence IN ('high', 'medium', 'low')),
    needs_review       INTEGER NOT NULL CHECK (needs_review IN (0, 1)),
    review_reason      TEXT
);

CREATE TABLE recall_categories (
    recall_id INTEGER NOT NULL REFERENCES recalls(id) ON DELETE CASCADE,
    category  TEXT    NOT NULL,
    position  INTEGER NOT NULL,
    PRIMARY KEY (recall_id, category)
);

CREATE TABLE metadata (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE INDEX idx_recalls_listing ON recalls (last_updated DESC, id DESC);
CREATE INDEX idx_recall_categories_category ON recall_categories (category);
"""


def to_records(frame: pd.DataFrame) -> list[dict[str, object]]:
    """Plain-Python dicts with None for missing values (no NaN leaks into JSON/SQL)."""
    records = []
    for raw in frame[list(RECORD_FIELDS)].to_dict(orient="records"):
        record: dict[str, object] = {}
        for key, value in raw.items():
            if isinstance(value, float) and pd.isna(value):
                value = None
            elif hasattr(value, "item"):  # numpy scalar -> Python scalar
                value = value.item()
            record[key] = value
        records.append(record)
    return records


class OutputLockedError(RuntimeError):
    """The destination file is open elsewhere (on Windows, typically the running API)."""


def _swap_into_place(tmp: Path, path: Path) -> None:
    """Atomically replace `path` with `tmp`; never leave the temp file behind."""
    try:
        os.replace(tmp, path)
    except OSError as error:
        tmp.unlink(missing_ok=True)
        raise OutputLockedError(
            f"Could not replace {path} ({error.strerror}). If the API server is running, "
            "stop it and run the pipeline again."
        ) from error


def write_jsonl(records: Iterable[Mapping[str, object]], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    count = 0
    try:
        with tmp.open("w", encoding="utf-8", newline="\n") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    _swap_into_place(tmp, path)
    return count


def write_json(data: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _swap_into_place(tmp, path)


def write_sqlite(
    records: list[Mapping[str, object]], path: Path, metadata: Mapping[str, object]
) -> None:
    """Build the database in a temp file and swap it in, so readers never see a half-built DB.

    A duplicate id raises sqlite3.IntegrityError and leaves any existing DB untouched.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.unlink(missing_ok=True)

    connection = sqlite3.connect(tmp)
    try:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.executescript(SCHEMA)
        with connection:
            connection.executemany(
                f"INSERT INTO recalls ({', '.join(f for f in RECORD_FIELDS if f != 'categories')}) "
                f"VALUES ({', '.join('?' for f in RECORD_FIELDS if f != 'categories')})",
                [
                    tuple(_sql_value(r[f]) for f in RECORD_FIELDS if f != "categories")
                    for r in records
                ],
            )
            connection.executemany(
                "INSERT INTO recall_categories (recall_id, category, position) VALUES (?, ?, ?)",
                [
                    (r["id"], category, position)
                    for r in records
                    for position, category in enumerate(r["categories"] or [])
                ],
            )
            connection.executemany(
                "INSERT INTO metadata (key, value) VALUES (?, ?)",
                [(k, json.dumps(v) if not isinstance(v, str) else v) for k, v in metadata.items()],
            )
    except Exception:
        connection.close()
        tmp.unlink(missing_ok=True)
        raise
    connection.close()
    _swap_into_place(tmp, path)


def _sql_value(value: object) -> object:
    if isinstance(value, bool):
        return int(value)
    return value
