"""Interactive terminal tool for writing consumer queries and judging relevant recalls.

    python -m evaluation.annotate --annotator AB

Workflow per query:
  1. Type the query FIRST, the way a consumer would phrase it, before looking at results.
  2. The tool shows a candidate pool (BM25 + MiniLM top results over all 1,000 recalls).
     Use `/find words` to search further and `/show ID` to read a record.
  3. Enter EVERY relevant recall id you find, optionally with grades: `ID:3 ID:2 ID`.
     Recalls may come from any split. List the recall the consumer most likely means
     first (or give it the highest grade): it is the query's anchor, and the query
     inherits the anchor's split.
New queries are saved as reviewer_status "draft"; a second team member reviews them later.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from evaluation import DEFAULT_ANNOTATIONS, DEFAULT_DB, DEFAULT_MANIFEST
from evaluation.annotations import (
    GRADES,
    Annotation,
    Judgment,
    append_jsonl,
    choose_anchor,
    corpus_context,
    read_jsonl,
)
from retrieval.embed import DEFAULT_OUT as DEFAULT_EMBEDDINGS
from retrieval.loader import build_engine
from retrieval.search import SearchEngine, SearchUnavailableError

POOL_DEPTH = 8
_ANNOTATOR = re.compile(r"^[A-Za-z0-9]{1,16}$")


def parse_relevance(text: str) -> list[Judgment]:
    """'50609:3 50874, 50656' -> judgments (grade 1 by default); rejects duplicates."""
    judgments: list[Judgment] = []
    seen: set[int] = set()
    for token in re.split(r"[\s,;]+", text.strip()):
        if not token:
            continue
        match = re.fullmatch(r"(\d+)(?::(\d+))?", token)
        if not match:
            raise ValueError(f"Cannot read {token!r}; use ID or ID:GRADE")
        rid, grade = int(match.group(1)), int(match.group(2) or 1)
        if grade not in GRADES:
            raise ValueError(f"Grade for {rid} must be one of {GRADES}")
        if rid in seen:
            raise ValueError(f"Recall {rid} listed twice")
        seen.add(rid)
        judgments.append(Judgment(rid, grade))
    if not judgments:
        raise ValueError("Enter at least one recall id")
    return judgments


def derive_split(judgments: list[Judgment], split_of: dict[int, str]) -> tuple[str, int]:
    """(query split, anchor recall). Relevant recalls may come from any split; the query
    takes the split of its anchor (highest grade, first listed)."""
    unknown = [j.recall_id for j in judgments if j.recall_id not in split_of]
    if unknown:
        raise ValueError(f"Not in the corpus: {unknown}")
    anchor = choose_anchor(judgments)
    return split_of[anchor], anchor


def next_query_id(existing: set[str], annotator: str) -> str:
    number = 1
    while f"{annotator}-{number:04d}" in existing:
        number += 1
    return f"{annotator}-{number:04d}"


def candidate_pool(engine: SearchEngine, query: str) -> list[tuple[int, str]]:
    """BM25 and MiniLM results interleaved and de-duplicated: (recall_id, source)."""
    pools = {}
    for model in ("bm25", "minilm"):
        try:
            pools[model] = [h.recall_id for h in engine.search(query, model=model, top_k=POOL_DEPTH)]
        except SearchUnavailableError:
            pools[model] = []
    seen, merged = set(), []
    for i in range(POOL_DEPTH):
        for model, ids in pools.items():
            if i < len(ids) and ids[i] not in seen:
                seen.add(ids[i])
                merged.append((ids[i], model))
    return merged


def run_session(
    engine: SearchEngine,
    split_of: dict[int, str],
    out_path: Path,
    annotator: str,
    split_fingerprint: str = "",
    read: Callable[[str], str] = input,
    write: Callable[[str], None] = print,
) -> int:
    records = {r.recall_id: r for r in engine.corpus}
    existing = {str(raw.get("query_id")) for raw in read_jsonl(out_path)} if out_path.is_file() else set()
    saved = 0

    def show(rid: int, note: str = "") -> None:
        # Split labels are deliberately hidden while judging; the split is derived on save.
        write(f"  #{rid:<6} {records[rid].title[:100]}{note}")

    while True:
        query = " ".join(read("\nConsumer query (blank to finish): ").split())
        if not query:
            write(f"Saved {saved} new draft queries to {out_path}")
            return saved
        write("Candidates (pooled from BM25 + MiniLM; also try /find):")
        for rid, source in candidate_pool(engine, query):
            show(rid, f"   <{source}>")
        while True:
            answer = read("Relevant ids (ID or ID:GRADE), /find WORDS, /show ID, /skip: ").strip()
            if answer == "/skip":
                break
            if answer.startswith("/find "):
                for hit in engine.search(answer[6:], model="bm25", top_k=15):
                    show(hit.recall_id)
                continue
            if answer.startswith("/show "):
                rid = int(answer[6:]) if answer[6:].strip().isdigit() else -1
                if rid in records:
                    r = records[rid]
                    write(f"  #{rid} {r.title}\n  category: {r.category}\n  issue: {r.issue}\n  {r.url}")
                else:
                    write("  unknown id")
                continue
            try:
                judgments = parse_relevance(answer)
                split, anchor = derive_split(judgments, split_of)
            except ValueError as error:
                write(f"  {error}")
                continue
            query_id = next_query_id(existing, annotator)
            existing.add(query_id)
            append_jsonl(Annotation(
                query_id=query_id, query=query, split=split, relevant=judgments,
                reviewer_status="draft", annotator=annotator, split_fingerprint=split_fingerprint,
                created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                anchor_recall_id=anchor,
            ), out_path)
            saved += 1
            write(f"  saved {query_id} ({split}, anchor #{anchor}, {len(judgments)} relevant, draft)")
            break


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write consumer queries and relevance labels.")
    parser.add_argument("--annotator", required=True, help="your initials, e.g. AB")
    parser.add_argument("--out", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--embeddings", type=Path, default=DEFAULT_EMBEDDINGS)
    args = parser.parse_args(argv)
    if not _ANNOTATOR.match(args.annotator):
        parser.error("--annotator must be 1-16 letters/digits")

    split_of, _, manifest = corpus_context(args.db, args.manifest)
    if not manifest.get("frozen"):
        print("WARNING: the split manifest is NOT frozen yet (split review pending). Queries "
              "written now may need re-validation if the split assignment changes.\n")
    engine = build_engine(args.db, args.embeddings)
    print(__doc__)
    run_session(engine, split_of, args.out, args.annotator,
                split_fingerprint=manifest.get("assignment_fingerprint", ""))
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
