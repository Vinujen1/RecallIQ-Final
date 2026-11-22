"""Command-line smoke test: run realistic consumer queries through both models.

    python -m retrieval.smoke [--top-k 5] [--query "..."]

This only shows that search runs end-to-end on real data. It is NOT an evaluation: there
are no relevance judgments, so no accuracy is measured or claimed.
"""

from __future__ import annotations

import argparse
import sys

from retrieval.embed import DEFAULT_DB, DEFAULT_OUT
from retrieval.loader import build_engine
from retrieval.search import SCORE_TYPES, SearchUnavailableError

SMOKE_QUERIES: tuple[str, ...] = (
    "peanut butter contaminated with salmonella",
    "my child's hoodie has a drawstring",
    "baby crib that can collapse",
    "phone charger overheating fire",
    "cheese with listeria",
    "chocolate with undeclared milk allergy",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="RecallIQ retrieval smoke test")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--query", action="append", help="custom query (repeatable)")
    args = parser.parse_args(argv)

    engine = build_engine(DEFAULT_DB, DEFAULT_OUT)
    print(f"Corpus: {len(engine.corpus)} recall records")
    failures = 0
    for query in args.query or SMOKE_QUERIES:
        print(f"\n=== {query}")
        for model in ("bm25", "minilm"):
            try:
                hits = engine.search(query, model=model, top_k=args.top_k)
            except SearchUnavailableError as error:
                print(f"  [{model}] UNAVAILABLE: {error}")
                failures += 1
                continue
            print(f"  [{model}] score = {SCORE_TYPES[model]}")
            if not hits:
                print("    (no results)")
            for hit in hits:
                print(f"    {hit.rank}. {hit.score:7.4f}  #{hit.recall_id}  {hit.title[:90]}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
