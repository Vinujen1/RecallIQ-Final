"""Export records whose heuristic product_family needs manual review.

    python -m recall_data.family_review [--db PATH] [--out CSV]

Writes a CSV with empty `reviewer_family` / `reviewer_notes` / `reviewed_by` columns for a
human reviewer. Nothing in RecallIQ treats these families as verified until reviewed.
"""

from __future__ import annotations

import argparse
import csv
import sqlite3
import sys
from collections import Counter
from pathlib import Path

from recall_data.build import DEFAULT_OUTPUT_DIR

COLUMNS = (
    "id", "domain", "category", "title", "product", "product_family", "family_source",
    "family_confidence", "review_reason", "url",
)
REVIEW_COLUMNS = ("reviewer_family", "reviewer_notes", "reviewed_by")


def fetch_flagged(db_path: Path) -> list[dict]:
    connection = sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute(
            f"SELECT {', '.join(COLUMNS)} FROM recalls WHERE needs_review = 1 "
            "ORDER BY family_confidence DESC, review_reason, id"
        ).fetchall()
    finally:
        connection.close()
    return [dict(row) for row in rows]


def summarize(rows: list[dict]) -> dict[str, Counter]:
    return {
        "by_reason": Counter((r["review_reason"] or "").split(":")[0] for r in rows),
        "by_confidence": Counter(r["family_confidence"] for r in rows),
        "by_domain": Counter(r["domain"] for r in rows),
        "by_family": Counter(r["product_family"] or "(none)" for r in rows),
    }


def write_csv(rows: list[dict], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(COLUMNS + REVIEW_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, **{c: "" for c in REVIEW_COLUMNS}})


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Export product families needing review.")
    parser.add_argument("--db", type=Path, default=DEFAULT_OUTPUT_DIR / "recalliq.db")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_DIR / "family_review.csv")
    args = parser.parse_args(argv)

    rows = fetch_flagged(args.db)
    write_csv(rows, args.out)
    print(f"{len(rows)} records need review -> {args.out.resolve()}")
    for name, counts in summarize(rows).items():
        print(f"\n{name}:")
        for value, count in counts.most_common():
            print(f"  {count:4}  {value}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
