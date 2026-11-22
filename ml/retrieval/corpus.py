"""The searchable corpus: one document per recall record in the SQLite database.

`search_text` is derived here, deterministically, from verbatim recall fields. It
deliberately excludes the heuristic `product_family` (see docs/RETRIEVAL.md) and the
boilerplate "What you should do" text.
"""

from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass
from pathlib import Path

# Bump when build_search_text changes; it is part of the corpus fingerprint, so stale
# embeddings are detected instead of silently mis-ranking.
SEARCH_TEXT_VERSION = "1"


@dataclass(frozen=True)
class CorpusRecord:
    recall_id: int
    title: str
    category: str | None
    issue: str | None
    url: str
    search_text: str


def build_search_text(
    title: str, product: str | None, issue: str | None, category: str | None
) -> str:
    """Title, then product, issue and category (when present), joined with '. '."""
    parts = [title]
    for value in (product, issue, category):
        if value and value not in parts:
            parts.append(value)
    return ". ".join(parts)


def load_corpus(db_path: Path) -> list[CorpusRecord]:
    """Read every recall, ordered by official id, from the read-only database."""
    if not db_path.is_file():
        raise FileNotFoundError(f"Recall database not found: {db_path}")
    connection = sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            "SELECT id, title, product, issue, category, url FROM recalls ORDER BY id"
        ).fetchall()
    finally:
        connection.close()
    return [
        CorpusRecord(
            recall_id=recall_id,
            title=title,
            category=category,
            issue=issue,
            url=url,
            search_text=build_search_text(title, product, issue, category),
        )
        for recall_id, title, product, issue, category, url in rows
    ]


def corpus_fingerprint(records: list[CorpusRecord]) -> str:
    """Hash of (id, search_text) pairs in order, plus the search-text version."""
    digest = hashlib.sha256(f"search_text_v{SEARCH_TEXT_VERSION}\n".encode())
    for record in records:
        digest.update(f"{record.recall_id}\t{record.search_text}\n".encode("utf-8"))
    return digest.hexdigest()
